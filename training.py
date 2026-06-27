import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("data") / "basenp"
DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_CTB_DIR = Path("CTB")
DEFAULT_CTB_ARCHIVE = Path("LDC2013T21.tgz")
DEFAULT_BASENP_LABEL = "NP"
NP_HEAD_POS = {"NN", "NR", "FW"}
OPTIONAL_NP_HEAD_POS = {"NT", "PN"}
FORBIDDEN_PHRASE_LABELS = {"IP", "VP", "CP", "PP", "LCP", "UCP", "FRAG"}
FORBIDDEN_FUNCTION_TAGS = {"TMP"}
FORBIDDEN_INTERNAL_POS = {"DEC", "DEG", "DEV", "DER"}
SINGLE_TOKEN_EXCLUDED_POS = {"NT", "DT", "CD", "OD", "M", "LC"}


def resolve_project_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def to_project_relative_path(path_like: Union[str, Path]) -> Path:
    absolute = resolve_project_path(path_like)
    try:
        return absolute.relative_to(PACKAGE_ROOT)
    except ValueError:
        return absolute


def display_project_path(path_like: Union[str, Path]) -> str:
    return to_project_relative_path(path_like).as_posix()


def default_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_project_arg(path_like: Union[str, Path]) -> str:
    return display_project_path(resolve_project_path(path_like))


def coarse_label(label: str) -> str:
    return label.split("-", 1)[0]


def split_tree_label(label: str) -> Tuple[str, List[str]]:
    parts = label.split("-")
    return parts[0], parts[1:]


def is_dataset_ready(data_dir: Union[str, Path]) -> bool:
    root = resolve_project_path(data_dir)
    return all((root / f"{split}.tsv").is_file() for split in ("train", "dev", "test"))


def ensure_ctb_ready(ctb_dir: Union[str, Path], archive_path: Union[str, Path], logger: "PipelineLogger") -> Path:
    ctb_root = resolve_project_path(ctb_dir)
    required_files = [ctb_root / "par" / f"{split}.noempty.txt" for split in ("train", "dev", "test")]
    if all(path.is_file() for path in required_files):
        logger.log(f"发现现成 CTB 预处理结果: {display_project_path(ctb_root)}")
        return ctb_root

    archive = resolve_project_path(archive_path)
    if not archive.is_file():
        raise FileNotFoundError(
            f"缺少 CTB 预处理输入，未找到数据包: {display_project_path(archive)}"
        )

    logger.log("未发现完整 CTB 预处理结果，开始执行 `preprocess.py` 生成 CTB 任务数据")
    import preprocess

    preprocess.ARCHIVE_PATH = archive
    preprocess.OUTPUT_ROOT = ctb_root
    preprocess.main()

    if not all(path.is_file() for path in required_files):
        raise RuntimeError(f"CTB 预处理未生成完整的 `par/*.noempty.txt`: {display_project_path(ctb_root)}")
    return ctb_root


def extract_base_np_spans(tree) -> List[Tuple[int, int]]:
    from phrasetree.tree import Tree

    def collect_terminal_tags(node: Tree) -> List[str]:
        tags: List[str] = []
        for subtree in node.subtrees():
            if is_preterminal(subtree):
                tags.append(coarse_label(subtree.label()))
        return tags

    def child_phrase_labels(node: Tree) -> List[str]:
        labels: List[str] = []
        for child in node:
            if isinstance(child, Tree):
                labels.append(coarse_label(child.label()))
        return labels

    def is_valid_base_np(node: Tree, start: int, end: int) -> bool:
        base_label, function_tags = split_tree_label(node.label())
        if base_label != "NP" or end <= start:
            return False
        if FORBIDDEN_FUNCTION_TAGS & set(function_tags):
            return False

        terminal_tags = collect_terminal_tags(node)
        if not terminal_tags:
            return False
        if any(tag in FORBIDDEN_INTERNAL_POS for tag in terminal_tags):
            return False

        direct_child_labels = child_phrase_labels(node)
        if any(label in FORBIDDEN_PHRASE_LABELS for label in direct_child_labels):
            return False

        has_nominal_head = any(tag in NP_HEAD_POS for tag in terminal_tags)
        if not has_nominal_head:
            has_nominal_head = any(tag in OPTIONAL_NP_HEAD_POS for tag in terminal_tags)
        if not has_nominal_head:
            return False

        if end - start == 1 and terminal_tags[0] in SINGLE_TOKEN_EXCLUDED_POS:
            return False
        return True

    def is_preterminal(node: Tree) -> bool:
        return len(node) > 0 and all(not isinstance(child, Tree) for child in node)

    def walk(node, cursor: int):
        if not isinstance(node, Tree):
            return cursor, [], False
        if is_preterminal(node):
            return cursor + 1, [], False

        spans: List[Tuple[int, int]] = []
        start = cursor
        contains_nested_np = False
        for child in node:
            cursor, child_spans, child_contains_np = walk(child, cursor)
            spans.extend(child_spans)
            contains_nested_np = contains_nested_np or child_contains_np

        end = cursor
        is_np = coarse_label(node.label()) == "NP"
        if is_valid_base_np(node, start, end):
            return cursor, [(start, end)], True
        return cursor, spans, is_np or contains_nested_np

    _, spans, _ = walk(tree, 0)
    return spans


def span_to_tags(length: int, label: str) -> List[str]:
    if length == 1:
        return [f"S-{label}"]
    tags = [f"B-{label}"]
    if length > 2:
        tags.extend([f"I-{label}"] * (length - 2))
    tags.append(f"E-{label}")
    return tags


def tree_to_basenp_sentence(tree_text: str, label: str = DEFAULT_BASENP_LABEL) -> Tuple[List[str], List[str], int]:
    from phrasetree.tree import Tree

    tree = Tree.fromstring(tree_text)
    tokens = [word for word, pos in tree.pos() if pos and pos != "-NONE-"]
    tags = ["O"] * len(tokens)
    chunk_count = 0
    for start, end in extract_base_np_spans(tree):
        if not (0 <= start < end <= len(tokens)):
            continue
        if any(tag != "O" for tag in tags[start:end]):
            continue
        tags[start:end] = span_to_tags(end - start, label)
        chunk_count += 1
    return tokens, tags, chunk_count


def convert_ctb_parse_to_basenp(src_path: Path, dst_path: Path, label: str, logger: "PipelineLogger") -> Dict[str, int]:
    sentence_count = 0
    token_count = 0
    chunk_count = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with src_path.open("r", encoding="utf-8") as src, dst_path.open("w", encoding="utf-8", newline="\n") as dst:
        for raw_line in src:
            tree_text = raw_line.strip()
            if not tree_text:
                continue
            tokens, tags, sent_chunk_count = tree_to_basenp_sentence(tree_text, label=label)
            if not tokens:
                continue
            for token, tag in zip(tokens, tags):
                dst.write(f"{token}\t{tag}\n")
            dst.write("\n")
            sentence_count += 1
            token_count += len(tokens)
            chunk_count += sent_chunk_count

    logger.log(
        f"已生成 {display_project_path(dst_path)} | sentences={sentence_count} tokens={token_count} chunks={chunk_count}"
    )
    return {"sentences": sentence_count, "tokens": token_count, "chunks": chunk_count}


def ensure_basenp_dataset(
    data_dir: Union[str, Path],
    ctb_dir: Union[str, Path],
    archive_path: Union[str, Path],
    label: str,
    logger: "PipelineLogger",
    force_rebuild: bool = False,
) -> Path:
    data_root = resolve_project_path(data_dir)
    if is_dataset_ready(data_root) and not force_rebuild:
        logger.log(f"发现现成 BaseNP 数据集: {display_project_path(data_root)}")
        return data_root
    if force_rebuild and data_root.exists():
        logger.log(f"按要求重建 BaseNP 数据集: {display_project_path(data_root)}")

    ctb_root = ensure_ctb_ready(ctb_dir, archive_path, logger)
    parse_root = ctb_root / "par"
    required_splits = {split: parse_root / f"{split}.noempty.txt" for split in ("train", "dev", "test")}
    missing = [display_project_path(path) for path in required_splits.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少 BaseNP 转换所需的 CTB parse 文件: {missing}")

    logger.log(f"开始从 {display_project_path(parse_root)} 自动生成 BaseNP 数据集 -> {display_project_path(data_root)}")
    summaries = {}
    for split, src_path in required_splits.items():
        dst_path = data_root / f"{split}.tsv"
        summaries[split] = convert_ctb_parse_to_basenp(src_path, dst_path, label, logger)
    logger.save_json("dataset_summary.json", {"data_dir": display_project_path(data_root), "splits": summaries})
    return data_root


class PipelineLogger:
    def __init__(self, output_root: Union[str, Path], run_name: str):
        self.output_root = resolve_project_path(output_root)
        self.run_name = run_name
        self.run_dir = self.output_root / "pipeline" / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "pipeline.log"

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(message)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(line + "\n")

    def save_json(self, file_name: str, payload: Dict[str, Any]) -> None:
        with (self.run_dir / file_name).open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)


@dataclass
class BaseTrainConfig:
    data_dir: str
    output_root: str
    model_name: str
    run_name: Optional[str] = None
    remove_o: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HMMTrainConfig(BaseTrainConfig):
    @classmethod
    def default(cls) -> "HMMTrainConfig":
        return cls(
            data_dir=str(DEFAULT_DATA_DIR),
            output_root=str(DEFAULT_OUTPUT_ROOT),
            model_name="hmm",
        )


@dataclass
class CRFTrainConfig(BaseTrainConfig):
    algorithm: str = "lbfgs"
    c1: float = 0.1
    c2: float = 0.1
    max_iterations: int = 100
    all_possible_transitions: bool = False

    @classmethod
    def default(cls) -> "CRFTrainConfig":
        return cls(
            data_dir=str(DEFAULT_DATA_DIR),
            output_root=str(DEFAULT_OUTPUT_ROOT),
            model_name="crf",
        )


@dataclass
class LSTMTrainConfig(BaseTrainConfig):
    batch_size: int = 64
    lr: float = 0.001
    epoches: int = 30
    print_step: int = 5
    emb_size: int = 128
    hidden_size: int = 128

    @classmethod
    def default(cls, model_name: str) -> "LSTMTrainConfig":
        return cls(
            data_dir=str(DEFAULT_DATA_DIR),
            output_root=str(DEFAULT_OUTPUT_ROOT),
            model_name=model_name,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一键完成 BaseNP 数据准备、全部模型训练与对比实验")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="BaseNP 数据目录")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="统一输出根目录")
    parser.add_argument("--ctb-dir", default=str(DEFAULT_CTB_DIR), help="CTB 预处理输出目录")
    parser.add_argument("--ctb-archive", default=str(DEFAULT_CTB_ARCHIVE), help="原始 CTB 压缩包路径")
    parser.add_argument("--run-name", default=None, help="本次流水线运行名称")
    parser.add_argument("--skip-data-prepare", action="store_true", help="跳过数据自动准备，要求 `data-dir` 已存在")
    parser.add_argument("--rebuild-data", action="store_true", help="即使 `data-dir` 已存在也强制重建 BaseNP 数据")
    parser.add_argument("--skip-compare", action="store_true", help="训练完成后跳过对比实验")
    parser.add_argument("--remove-o", action="store_true", help="token 级评估时移除 O 标签")
    parser.add_argument("--crf-algorithm", default="lbfgs", help="CRF 优化算法")
    parser.add_argument("--crf-c1", type=float, default=0.1, help="CRF L1 正则")
    parser.add_argument("--crf-c2", type=float, default=0.1, help="CRF L2 正则")
    parser.add_argument("--crf-max-iterations", type=int, default=100, help="CRF 最大迭代次数")
    parser.add_argument("--crf-all-possible-transitions", action="store_true", help="CRF 是否启用所有转移")
    parser.add_argument("--batch-size", type=int, default=64, help="BiLSTM 系列 batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="BiLSTM 系列学习率")
    parser.add_argument("--lstm-epochs", type=int, default=30, help="BiLSTM 系列训练轮数")
    parser.add_argument("--print-step", type=int, default=5, help="BiLSTM 系列日志打印步数")
    parser.add_argument("--emb-size", type=int, default=128, help="BiLSTM 系列向量维度")
    parser.add_argument("--hidden-size", type=int, default=128, help="BiLSTM 系列隐层维度")
    return parser


def run_pipeline(args) -> Dict[str, Any]:
    from config import run_bilstm_training, run_crf_training, run_hmm_training

    run_name = args.run_name or default_run_name()
    logger = PipelineLogger(args.output_root, run_name)
    logger.log(f"流水线开始，run_name={run_name}")
    logger.log(f"项目根目录: {PACKAGE_ROOT.as_posix()}")

    data_dir = normalize_project_arg(args.data_dir)
    output_root = normalize_project_arg(args.output_root)

    if args.skip_data_prepare:
        if not is_dataset_ready(data_dir):
            raise FileNotFoundError(f"数据目录不完整，无法跳过数据准备: {display_project_path(data_dir)}")
        logger.log(f"跳过数据准备，直接使用: {display_project_path(data_dir)}")
    else:
        ensure_basenp_dataset(
            args.data_dir,
            args.ctb_dir,
            args.ctb_archive,
            DEFAULT_BASENP_LABEL,
            logger,
            force_rebuild=args.rebuild_data,
        )

    hmm_config = HMMTrainConfig(
        data_dir=data_dir,
        output_root=output_root,
        model_name="hmm",
        run_name=run_name,
        remove_o=args.remove_o,
    )
    crf_config = CRFTrainConfig(
        data_dir=data_dir,
        output_root=output_root,
        model_name="crf",
        run_name=run_name,
        remove_o=args.remove_o,
        algorithm=args.crf_algorithm,
        c1=args.crf_c1,
        c2=args.crf_c2,
        max_iterations=args.crf_max_iterations,
        all_possible_transitions=args.crf_all_possible_transitions,
    )
    bilstm_config = LSTMTrainConfig(
        data_dir=data_dir,
        output_root=output_root,
        model_name="bilstm",
        run_name=run_name,
        remove_o=args.remove_o,
        batch_size=args.batch_size,
        lr=args.lr,
        epoches=args.lstm_epochs,
        print_step=args.print_step,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
    )
    bilstm_crf_config = LSTMTrainConfig(
        data_dir=data_dir,
        output_root=output_root,
        model_name="bilstm_crf",
        run_name=run_name,
        remove_o=args.remove_o,
        batch_size=args.batch_size,
        lr=args.lr,
        epoches=args.lstm_epochs,
        print_step=args.print_step,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
    )

    completed: Dict[str, str] = {}
    failed: Dict[str, str] = {}

    train_jobs = [
        ("hmm", lambda: run_hmm_training(hmm_config)),
        ("crf", lambda: run_crf_training(crf_config)),
        ("bilstm", lambda: run_bilstm_training(bilstm_config, use_crf=False)),
        ("bilstm_crf", lambda: run_bilstm_training(bilstm_crf_config, use_crf=True)),
    ]

    for model_name, runner in train_jobs:
        logger.log(f"开始训练 {model_name}")
        try:
            result = runner()
            ctx = result[-1]
            completed[model_name] = display_project_path(ctx.run_dir)
            logger.log(f"{model_name} 训练完成，输出目录: {completed[model_name]}")
        except Exception as exc:
            failed[model_name] = str(exc)
            logger.log(f"{model_name} 训练失败: {exc}")

    compare_output = None
    if not args.skip_compare:
        if completed:
            logger.log("开始运行对比实验")
            from compare_models import run_compare

            compare_output = run_compare(data_dir, output_root, run_name)
            compare_output["run_dir"] = normalize_project_arg(compare_output["run_dir"])
            logger.log("对比实验完成")
        else:
            logger.log("没有成功训练的模型，跳过对比实验")

    summary = {
        "run_name": run_name,
        "data_dir": data_dir,
        "output_root": output_root,
        "completed_models": completed,
        "failed_models": failed,
        "compare_output": compare_output,
    }
    logger.save_json("summary.json", summary)
    if not completed:
        raise RuntimeError("所有模型训练均失败，请检查 `outputs/pipeline` 下的流水线日志。")
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
