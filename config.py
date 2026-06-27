import json
import os
import pickle
import time
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from bilstm import BiLSTM
from bilstm_crf import BiLSTM_CRF
from crf import CRFModel
from hmm import HMM


PACKAGE_ROOT = Path(__file__).resolve().parent
SPLIT_SUFFIXES = (".tsv", ".txt", ".bio", ".bmes")


def resolve_project_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def to_project_relative_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    absolute = resolve_project_path(path)
    try:
        return absolute.relative_to(PACKAGE_ROOT)
    except ValueError:
        return absolute


def display_project_path(path_like: Union[str, Path]) -> str:
    relative = to_project_relative_path(path_like)
    return relative.as_posix()


def resolve_split_path(data_dir: Union[str, Path], split: str) -> Path:
    root = resolve_project_path(data_dir)
    for suffix in SPLIT_SUFFIXES:
        candidate = root / f"{split}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"未找到 {split} 数据文件: {display_project_path(root)}")


def relativize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: relativize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [relativize_payload(value) for value in payload]
    if isinstance(payload, tuple):
        return [relativize_payload(value) for value in payload]
    if isinstance(payload, Path):
        return display_project_path(payload)
    if isinstance(payload, str):
        candidate = Path(payload)
        if candidate.is_absolute():
            return display_project_path(candidate)
        return payload.replace("\\", "/")
    return payload


def flatten_lists(lists_):
    flatten_list = []
    for item in lists_:
        if isinstance(item, list):
            flatten_list.extend(item)
        else:
            flatten_list.append(item)
    return flatten_list


def save_pickle(obj, file_name):
    file_path = Path(file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(file_name):
    with Path(file_name).open("rb") as f:
        return pickle.load(f)


def build_map(lists_: Sequence[Sequence[str]]) -> Dict[str, int]:
    maps: Dict[str, int] = {}
    for list_ in lists_:
        for elem in list_:
            if elem not in maps:
                maps[elem] = len(maps)
    return maps


def build_corpus(split: str, make_vocab: bool = True, data_dir: str = ""):
    assert split in ["train", "dev", "test"]
    file_path = resolve_split_path(data_dir, split)

    word_lists: List[List[str]] = []
    tag_lists: List[List[str]] = []
    with file_path.open("r", encoding="utf-8") as f:
        word_list: List[str] = []
        tag_list: List[str] = []
        for raw_line in f:
            line = raw_line.strip()
            if line:
                columns = line.split()
                if len(columns) < 2:
                    raise ValueError(f"数据格式错误，至少需要 token 和 tag 两列: {line}")
                word = columns[0]
                tag = columns[-1]
                word_list.append(word)
                tag_list.append(tag)
            else:
                if word_list:
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                word_list = []
                tag_list = []

        if word_list:
            word_lists.append(word_list)
            tag_lists.append(tag_list)

    if make_vocab:
        return word_lists, tag_lists, build_map(word_lists), build_map(tag_lists)
    return word_lists, tag_lists


def extend_maps(word2id: Dict[str, int], tag2id: Dict[str, int], for_crf: bool = True):
    word2id = dict(word2id)
    tag2id = dict(tag2id)

    word2id["<unk>"] = len(word2id)
    word2id["<pad>"] = len(word2id)
    tag2id["<unk>"] = len(tag2id)
    tag2id["<pad>"] = len(tag2id)

    if for_crf:
        word2id["<start>"] = len(word2id)
        word2id["<end>"] = len(word2id)
        tag2id["<start>"] = len(tag2id)
        tag2id["<end>"] = len(tag2id)
    return word2id, tag2id


def preprocess_data_for_lstmcrf(word_lists, tag_lists, test: bool = False):
    assert len(word_lists) == len(tag_lists)
    new_word_lists = []
    new_tag_lists = []
    for words, tags in zip(word_lists, tag_lists):
        words = list(words) + ["<end>"]
        tags = list(tags)
        if not test:
            tags.append("<end>")
        new_word_lists.append(words)
        new_tag_lists.append(tags)
    return new_word_lists, new_tag_lists


def tensorized(batch, maps):
    pad_id = maps.get("<pad>")
    unk_id = maps.get("<unk>")

    max_len = len(batch[0])
    batch_size = len(batch)
    batch_tensor = torch.ones(batch_size, max_len).long() * pad_id
    for i, seq in enumerate(batch):
        for j, elem in enumerate(seq):
            batch_tensor[i][j] = maps.get(elem, unk_id)
    lengths = [len(seq) for seq in batch]
    return batch_tensor, lengths


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)), key=lambda idx: len(pairs[idx][0]), reverse=True)
    pairs = [pairs[idx] for idx in indices]
    word_lists, tag_lists = list(zip(*pairs))
    return word_lists, tag_lists, indices


def cal_loss(logits, targets, tag2id):
    pad_id = tag2id.get("<pad>")
    mask = targets != pad_id
    targets = targets[mask]
    out_size = logits.size(2)
    logits = logits.masked_select(mask.unsqueeze(2).expand(-1, -1, out_size)).contiguous().view(-1, out_size)
    return F.cross_entropy(logits, targets)


def indexed(targets, tagset_size, start_id):
    batch_size, max_len = targets.size()
    for col in range(max_len - 1, 0, -1):
        targets[:, col] += targets[:, col - 1] * tagset_size
    targets[:, 0] += start_id * tagset_size
    return targets


def cal_lstm_crf_loss(crf_scores, targets, tag2id):
    pad_id = tag2id.get("<pad>")
    start_id = tag2id.get("<start>")
    end_id = tag2id.get("<end>")
    device = crf_scores.device

    batch_size, max_len = targets.size()
    target_size = len(tag2id)
    mask = targets != pad_id
    lengths = mask.sum(dim=1)
    targets = indexed(targets, target_size, start_id)
    targets = targets.masked_select(mask)

    flatten_scores = crf_scores.masked_select(
        mask.view(batch_size, max_len, 1, 1).expand_as(crf_scores)
    ).view(-1, target_size * target_size).contiguous()
    golden_scores = flatten_scores.gather(dim=1, index=targets.unsqueeze(1)).sum()

    scores_upto_t = torch.zeros(batch_size, target_size).to(device)
    for t in range(max_len):
        batch_size_t = (lengths > t).sum().item()
        if t == 0:
            scores_upto_t[:batch_size_t] = crf_scores[:batch_size_t, t, start_id, :]
        else:
            scores_upto_t[:batch_size_t] = torch.logsumexp(
                crf_scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                dim=1,
            )
    all_path_scores = scores_upto_t[:, end_id].sum()
    return (all_path_scores - golden_scores) / batch_size


class Metrics(object):
    def __init__(self, golden_tags, predict_tags, remove_O: bool = False):
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)

        if remove_O:
            self._remove_Otags()

        self.tagset = sorted(set(self.golden_tags))
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)
        self.precision_scores = self.cal_precision()
        self.recall_scores = self.cal_recall()
        self.f1_scores = self.cal_f1()

    def cal_precision(self):
        scores = {}
        for tag in self.tagset:
            denom = self.predict_tags_counter[tag]
            scores[tag] = self.correct_tags_number.get(tag, 0) / denom if denom else 0.0
        return scores

    def cal_recall(self):
        scores = {}
        for tag in self.tagset:
            denom = self.golden_tags_counter[tag]
            scores[tag] = self.correct_tags_number.get(tag, 0) / denom if denom else 0.0
        return scores

    def cal_f1(self):
        scores = {}
        for tag in self.tagset:
            precision = self.precision_scores[tag]
            recall = self.recall_scores[tag]
            scores[tag] = 2 * precision * recall / (precision + recall + 1e-10)
        return scores

    def count_correct_tags(self):
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                correct_dict[gold_tag] = correct_dict.get(gold_tag, 0) + 1
        return correct_dict

    def weighted_average(self):
        weighted = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        total = len(self.golden_tags)
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted["precision"] += self.precision_scores[tag] * size
            weighted["recall"] += self.recall_scores[tag] * size
            weighted["f1_score"] += self.f1_scores[tag] * size
        for metric in weighted:
            weighted[metric] /= total
        return weighted

    def confusion_matrix(self):
        matrix = []
        for _ in self.tagset:
            matrix.append([0] * len(self.tagset))
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = self.tagset.index(golden_tag)
                col = self.tagset.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:
                continue
        return matrix

    def as_dict(self) -> Dict[str, object]:
        per_tag = {}
        for tag in self.tagset:
            per_tag[tag] = {
                "precision": self.precision_scores[tag],
                "recall": self.recall_scores[tag],
                "f1_score": self.f1_scores[tag],
                "support": self.golden_tags_counter[tag],
            }
        return {
            "per_tag": per_tag,
            "avg_total": self.weighted_average(),
            "support_total": len(self.golden_tags),
            "confusion_matrix": {
                "labels": self.tagset,
                "matrix": self.confusion_matrix(),
            },
        }

    def render_scores(self) -> str:
        lines: List[str] = []
        header_format = "{:>9s}  {:>9} {:>9} {:>9} {:>9}"
        row_format = "{:>9s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>9}"
        lines.append(header_format.format("", "precision", "recall", "f1-score", "support"))
        for tag in self.tagset:
            lines.append(
                row_format.format(
                    tag,
                    self.precision_scores[tag],
                    self.recall_scores[tag],
                    self.f1_scores[tag],
                    self.golden_tags_counter[tag],
                )
            )
        avg = self.weighted_average()
        lines.append(
            row_format.format(
                "avg/total",
                avg["precision"],
                avg["recall"],
                avg["f1_score"],
                len(self.golden_tags),
            )
        )
        return "\n".join(lines)

    def render_confusion_matrix(self) -> str:
        lines = ["Confusion Matrix:"]
        row_format = "{:>7} " * (len(self.tagset) + 1)
        lines.append(row_format.format("", *self.tagset))
        for i, row in enumerate(self.confusion_matrix()):
            lines.append(row_format.format(self.tagset[i], *row))
        return "\n".join(lines)

    def render_report(self) -> str:
        return self.render_scores() + "\n\n" + self.render_confusion_matrix()

    def _remove_Otags(self):
        indices = [i for i, tag in enumerate(self.golden_tags) if tag == "O"]
        self.golden_tags = [tag for i, tag in enumerate(self.golden_tags) if i not in indices]
        self.predict_tags = [tag for i, tag in enumerate(self.predict_tags) if i not in indices]


def split_chunk_tag(tag: str) -> Tuple[str, str]:
    if tag == "O":
        return "O", ""
    if "-" in tag:
        prefix, label = tag.split("-", 1)
        return prefix.upper(), label
    return tag.upper(), ""


def extract_chunks(tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    chunks: List[Tuple[int, int, str]] = []
    active_start: Optional[int] = None
    active_label = ""

    def close_chunk(end_index: int) -> None:
        nonlocal active_start, active_label
        if active_start is not None:
            chunks.append((active_start, end_index, active_label))
            active_start = None
            active_label = ""

    for index, tag in enumerate(tags):
        prefix, label = split_chunk_tag(tag)

        if prefix == "O":
            close_chunk(index)
            continue

        if prefix in {"S", "U"}:
            close_chunk(index)
            chunks.append((index, index + 1, label))
            continue

        if prefix == "B":
            close_chunk(index)
            active_start = index
            active_label = label
            continue

        if prefix in {"I", "M"}:
            if active_start is None or label != active_label:
                close_chunk(index)
                active_start = index
                active_label = label
            continue

        if prefix == "E":
            if active_start is None or label != active_label:
                close_chunk(index)
                chunks.append((index, index + 1, label))
            else:
                chunks.append((active_start, index + 1, active_label))
                active_start = None
                active_label = ""
            continue

        close_chunk(index)
        chunks.append((index, index + 1, label or prefix))

    close_chunk(len(tags))
    return chunks


def chunking_report(gold_tag_lists: Sequence[Sequence[str]], pred_tag_lists: Sequence[Sequence[str]]) -> Dict[str, float]:
    gold_total = 0
    pred_total = 0
    correct_total = 0
    exact_match = 0

    for gold_tags, pred_tags in zip(gold_tag_lists, pred_tag_lists):
        gold_chunks = set(extract_chunks(gold_tags))
        pred_chunks = set(extract_chunks(pred_tags))
        gold_total += len(gold_chunks)
        pred_total += len(pred_chunks)
        correct_total += len(gold_chunks & pred_chunks)
        if list(gold_tags) == list(pred_tags):
            exact_match += 1

    precision = correct_total / pred_total if pred_total else 0.0
    recall = correct_total / gold_total if gold_total else 0.0
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    sentence_acc = exact_match / len(gold_tag_lists) if gold_tag_lists else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "gold_chunks": gold_total,
        "pred_chunks": pred_total,
        "correct_chunks": correct_total,
        "sentence_accuracy": sentence_acc,
        "sentences": len(gold_tag_lists),
    }


def render_chunking_report(metrics: Dict[str, float]) -> str:
    return "\n".join(
        [
            "Chunk Metrics:",
            "precision={:.4f} recall={:.4f} f1-score={:.4f} sent-acc={:.4f}".format(
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["sentence_accuracy"],
            ),
            "gold_chunks={} pred_chunks={} correct_chunks={} sentences={}".format(
                int(metrics["gold_chunks"]),
                int(metrics["pred_chunks"]),
                int(metrics["correct_chunks"]),
                int(metrics["sentences"]),
            ),
        ]
    )


def render_chunking_table(results: Dict[str, Dict[str, float]]) -> str:
    header = "{:>12s}  {:>9} {:>9} {:>9} {:>10} {:>11}".format(
        "model", "precision", "recall", "f1-score", "sent-acc", "pred-chunks"
    )
    rows = [header]
    for model_name, metrics in results.items():
        rows.append(
            "{:>12s}  {:>9.4f} {:>9.4f} {:>9.4f} {:>10.4f} {:>11}".format(
                model_name,
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["sentence_accuracy"],
                int(metrics["pred_chunks"]),
            )
        )
    return "\n".join(rows)


def save_chunk_metrics(ctx, metrics: Dict[str, float], file_stem: str = "test_chunk_metrics") -> None:
    ctx.save_json(ctx.reports_dir / f"{file_stem}.json", metrics)
    ctx.save_text(ctx.reports_dir / f"{file_stem}.txt", render_chunking_report(metrics) + "\n")


class RunContext(object):
    def __init__(self, model_type: str, output_root: str, model_name: str, run_name: Optional[str] = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_name = f"{model_name}_{timestamp}"

        self.model_type = model_type
        self.model_name = model_name
        self.output_root = resolve_project_path(output_root)
        self.run_dir = self.output_root / model_type / run_name
        self.config_dir = self.run_dir / "config"
        self.logs_dir = self.run_dir / "logs"
        self.artifacts_dir = self.run_dir / "artifacts"
        self.reports_dir = self.run_dir / "reports"
        for dir_path in [self.config_dir, self.logs_dir, self.artifacts_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / "train.log"

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(message)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    def save_json(self, path: Path, payload: Any):
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_text(self, path: Path, text: str):
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)

    def save_config(self, config: Any):
        if is_dataclass(config):
            payload = asdict(config)
        elif hasattr(config, "to_dict"):
            payload = config.to_dict()
        else:
            payload = dict(config)
        payload = relativize_payload(payload)
        self.save_json(self.config_dir / "train_config.json", payload)

    def save_model(self, model):
        save_pickle(model, self.artifacts_dir / f"{self.model_name}.pkl")

    def save_metrics(self, metrics, file_stem: str = "test_metrics"):
        self.save_json(self.reports_dir / f"{file_stem}.json", metrics.as_dict())
        self.save_text(self.reports_dir / f"{file_stem}.txt", metrics.render_report())

    def save_summary(self, payload: Any, name: str = "summary.json"):
        self.save_json(self.reports_dir / name, payload)


def find_latest_artifact(output_root: str, model_type: str, model_name: str) -> Path:
    root = resolve_project_path(output_root) / model_type
    candidates = sorted(root.glob(f"*/artifacts/{model_name}.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"未找到 {model_type} 模型产物: {root}")
    return candidates[0]


def run_crf_training(config):
    ctx = RunContext("crf", config.output_root, config.model_name, config.run_name)
    ctx.save_config(config)
    ctx.log(f"读取 BaseNP 数据集: {display_project_path(config.data_dir)}")

    train_word_lists, train_tag_lists, _, _ = build_corpus("train", data_dir=config.data_dir)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=config.data_dir)

    start = time.time()
    stage_bar = tqdm(total=2, desc="CRF 训练流程")
    model = CRFModel(
        algorithm=config.algorithm,
        c1=config.c1,
        c2=config.c2,
        max_iterations=config.max_iterations,
        all_possible_transitions=config.all_possible_transitions,
    )
    model.train(train_word_lists, train_tag_lists)
    stage_bar.update(1)
    elapsed = int(time.time() - start)
    ctx.log(f"CRF 训练完成，用时 {elapsed} 秒")

    stage_bar.set_description("CRF 测试集预测")
    pred_tag_lists = model.test(test_word_lists)
    stage_bar.update(1)
    stage_bar.close()

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=config.remove_o)
    chunk_metrics = chunking_report(test_tag_lists, pred_tag_lists)
    ctx.log(metrics.render_scores())
    ctx.log(metrics.render_confusion_matrix())
    ctx.log(render_chunking_report(chunk_metrics))

    ctx.save_model(model)
    ctx.save_metrics(metrics)
    save_chunk_metrics(ctx, chunk_metrics)
    ctx.save_summary(
        {
            "model_type": "crf",
            "time_seconds": elapsed,
            "train_sentences": len(train_word_lists),
            "test_sentences": len(test_word_lists),
            "chunk_metrics": chunk_metrics,
        }
    )
    return pred_tag_lists, metrics, ctx


def run_hmm_training(config):
    ctx = RunContext("hmm", config.output_root, config.model_name, config.run_name)
    ctx.save_config(config)
    ctx.log(f"读取 BaseNP 数据集: {display_project_path(config.data_dir)}")

    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", data_dir=config.data_dir)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=config.data_dir)

    start = time.time()
    stage_bar = tqdm(total=2, desc="HMM 训练流程")
    stage_bar.set_postfix(sentences=len(train_word_lists))
    model = HMM(len(tag2id), len(word2id))
    model.train(train_word_lists, train_tag_lists, word2id, tag2id)
    stage_bar.update(1)
    elapsed = int(time.time() - start)
    ctx.log(f"HMM 训练完成，用时 {elapsed} 秒")

    stage_bar.set_description("HMM 测试集预测")
    pred_tag_lists = []
    for word_list in tqdm(test_word_lists, desc="HMM 测试集预测", leave=False):
        pred_tag_lists.append(model.decoding(word_list, word2id, tag2id))
    stage_bar.update(1)
    stage_bar.close()

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=config.remove_o)
    chunk_metrics = chunking_report(test_tag_lists, pred_tag_lists)
    ctx.log(metrics.render_scores())
    ctx.log(metrics.render_confusion_matrix())
    ctx.log(render_chunking_report(chunk_metrics))

    ctx.save_model(model)
    ctx.save_metrics(metrics)
    save_chunk_metrics(ctx, chunk_metrics)
    ctx.save_summary(
        {
            "model_type": "hmm",
            "time_seconds": elapsed,
            "train_sentences": len(train_word_lists),
            "test_sentences": len(test_word_lists),
            "chunk_metrics": chunk_metrics,
        }
    )
    return pred_tag_lists, metrics, ctx


class SequenceTaggerTrainer(object):
    def __init__(self, vocab_size, out_size, config, use_crf: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.use_crf = use_crf

        if use_crf:
            self.model = BiLSTM_CRF(vocab_size, config.emb_size, config.hidden_size, out_size).to(self.device)
            self.loss_fn = cal_lstm_crf_loss
        else:
            self.model = BiLSTM(vocab_size, config.emb_size, config.hidden_size, out_size).to(self.device)
            self.loss_fn = cal_loss

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.best_model = None
        self.best_val_loss = 1e18
        self.history = []

    def train(self, word_lists, tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id, logger):
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

        for epoch in range(1, self.config.epoches + 1):
            step = 0
            loss_bucket = 0.0
            total_step = len(word_lists) // self.config.batch_size + 1
            progress = tqdm(
                range(0, len(word_lists), self.config.batch_size),
                desc=f"Epoch {epoch}/{self.config.epoches}",
                leave=False,
                total=total_step,
            )
            for ind in progress:
                step += 1
                batch_sents = word_lists[ind : ind + self.config.batch_size]
                batch_tags = tag_lists[ind : ind + self.config.batch_size]
                batch_loss = self.train_step(batch_sents, batch_tags, word2id, tag2id)
                loss_bucket += batch_loss
                progress.set_postfix(loss=f"{batch_loss:.4f}")

                if step % self.config.print_step == 0:
                    logger(
                        "Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                            epoch,
                            step,
                            total_step,
                            100.0 * step / total_step,
                            loss_bucket / self.config.print_step,
                        )
                    )
                    loss_bucket = 0.0
            progress.close()

            val_loss = self.validate(dev_word_lists, dev_tag_lists, word2id, tag2id)
            self.history.append({"epoch": epoch, "val_loss": val_loss})
            logger("Epoch {}, Val Loss:{:.4f}".format(epoch, val_loss))

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, _ = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        scores = self.model(tensorized_sents, lengths)
        self.optimizer.zero_grad()
        loss = self.loss_fn(scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_steps = 0
            progress = tqdm(
                range(0, len(dev_word_lists), self.config.batch_size),
                desc="验证中",
                leave=False,
                total=(len(dev_word_lists) // self.config.batch_size + 1),
            )
            for ind in progress:
                total_steps += 1
                batch_sents = dev_word_lists[ind : ind + self.config.batch_size]
                batch_tags = dev_tag_lists[ind : ind + self.config.batch_size]
                tensorized_sents, lengths = tensorized(batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, _ = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)
                scores = self.model(tensorized_sents, lengths)
                batch_loss = self.loss_fn(scores, targets, tag2id).item()
                total_loss += batch_loss
                progress.set_postfix(loss=f"{batch_loss:.4f}")
            progress.close()

            val_loss = total_loss / total_steps
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = deepcopy(self.model)
            return val_loss

    def predict(self, word_lists, tag_lists, word2id, tag2id):
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        model = self.best_model if self.best_model is not None else self.model
        model.eval()
        with torch.no_grad():
            batch_tagids = model.test(tensorized_sents, lengths, tag2id)

        id2tag = {id_: tag for tag, id_ in tag2id.items()}
        pred_tag_lists = []
        for i, ids in enumerate(batch_tagids):
            valid_length = lengths[i] - 1 if self.use_crf else lengths[i]
            pred_tag_lists.append([id2tag[ids[j].item()] for j in range(valid_length)])

        ind_maps = sorted(list(enumerate(indices)), key=lambda elem: elem[1])
        recovered_indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in recovered_indices]
        tag_lists = [tag_lists[i] for i in recovered_indices]
        return pred_tag_lists, tag_lists


def run_bilstm_training(config, use_crf: bool = False):
    model_type = "bilstm_crf" if use_crf else "bilstm"
    ctx = RunContext(model_type, config.output_root, config.model_name, config.run_name)
    ctx.save_config(config)
    ctx.log(f"读取 BaseNP 数据集: {display_project_path(config.data_dir)}")

    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", data_dir=config.data_dir)
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False, data_dir=config.data_dir)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=config.data_dir)

    if use_crf:
        word2id, tag2id = extend_maps(word2id, tag2id, for_crf=True)
        train_word_lists, train_tag_lists = preprocess_data_for_lstmcrf(train_word_lists, train_tag_lists)
        dev_word_lists, dev_tag_lists = preprocess_data_for_lstmcrf(dev_word_lists, dev_tag_lists)
        test_word_lists, test_tag_lists = preprocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)
    else:
        word2id, tag2id = extend_maps(word2id, tag2id, for_crf=False)

    start = time.time()
    trainer = SequenceTaggerTrainer(len(word2id), len(tag2id), config, use_crf=use_crf)
    trainer.train(train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id, ctx.log)
    elapsed = int(time.time() - start)

    test_predict_bar = tqdm(total=1, desc=f"{model_type} 测试集预测")
    pred_tag_lists, recovered_test_tags = trainer.predict(test_word_lists, test_tag_lists, word2id, tag2id)
    test_predict_bar.update(1)
    test_predict_bar.close()

    metrics = Metrics(recovered_test_tags, pred_tag_lists, remove_O=config.remove_o)
    chunk_metrics = chunking_report(recovered_test_tags, pred_tag_lists)
    ctx.log(metrics.render_scores())
    ctx.log(metrics.render_confusion_matrix())
    ctx.log(render_chunking_report(chunk_metrics))

    ctx.save_model(trainer)
    ctx.save_metrics(metrics)
    save_chunk_metrics(ctx, chunk_metrics)
    ctx.save_summary(
        {
            "model_type": model_type,
            "time_seconds": elapsed,
            "train_sentences": len(train_word_lists),
            "dev_sentences": len(dev_word_lists),
            "test_sentences": len(test_word_lists),
            "best_val_loss": trainer.best_val_loss,
            "history": trainer.history,
            "chunk_metrics": chunk_metrics,
        }
    )
    return pred_tag_lists, recovered_test_tags, metrics, ctx
