import argparse
from collections import Counter
from typing import Iterable, List, Optional, Sequence, Tuple

from config import (
    RunContext,
    build_corpus,
    chunking_report,
    extend_maps,
    find_latest_artifact,
    load_pickle,
    preprocess_data_for_lstmcrf,
    render_chunking_table,
)
from training import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_ROOT


NP_HEAD_PREFIXES = ("n", "nr", "ns", "nt", "nz", "nl", "ng", "f", "s", "r")
NP_MODIFIER_PREFIXES = ("a", "ad", "an", "b", "m", "mq", "q")


def _load_jieba_posseg():
    try:
        import jieba.posseg as pseg
    except ImportError as exc:
        raise ImportError("jieba 基线依赖 jieba，请先安装 requirements.txt 中的依赖。") from exc
    return pseg


def _maybe_flatten_bilstm(model):
    try:
        model.model.bilstm.flatten_parameters()
    except AttributeError:
        pass
    try:
        model.model.bilstm.bilstm.flatten_parameters()
    except AttributeError:
        pass


def _normalize_flag(flag: str) -> str:
    return (flag or "").lower()


def _is_np_head(flag: str) -> bool:
    flag = _normalize_flag(flag)
    return any(flag.startswith(prefix) for prefix in NP_HEAD_PREFIXES)


def _is_np_modifier(flag: str) -> bool:
    flag = _normalize_flag(flag)
    return any(flag.startswith(prefix) for prefix in NP_MODIFIER_PREFIXES)


def _candidate_np_flags(flag: str) -> bool:
    return _is_np_head(flag) or _is_np_modifier(flag)


def _build_token_spans(tokens: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        next_cursor = cursor + len(token)
        spans.append((cursor, next_cursor))
        cursor = next_cursor
    return spans


def _extract_jieba_np_spans(tokens: Sequence[str]) -> List[Tuple[int, int]]:
    raw_text = "".join(tokens)
    if not raw_text:
        return []

    pseg = _load_jieba_posseg()
    spans: List[Tuple[int, int]] = []
    cursor = 0
    active_start: Optional[int] = None
    active_end = 0
    active_has_head = False

    def close_active() -> None:
        nonlocal active_start, active_end, active_has_head
        if active_start is not None and active_has_head and active_end > active_start:
            spans.append((active_start, active_end))
        active_start = None
        active_end = 0
        active_has_head = False

    for pair in pseg.cut(raw_text, HMM=True):
        word = pair.word
        flag = _normalize_flag(pair.flag)
        start = cursor
        end = cursor + len(word)
        cursor = end

        if _candidate_np_flags(flag):
            if active_start is None:
                active_start = start
            active_end = end
            active_has_head = active_has_head or _is_np_head(flag)
        else:
            close_active()

    close_active()
    return spans


def _overlap_length(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> int:
    return max(0, min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]))


def _best_chunk_index(token_span: Tuple[int, int], chunk_spans: Sequence[Tuple[int, int]]) -> Optional[int]:
    best_index: Optional[int] = None
    best_overlap = 0
    token_len = max(1, token_span[1] - token_span[0])
    for index, chunk_span in enumerate(chunk_spans):
        overlap = _overlap_length(token_span, chunk_span)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    if best_index is None:
        return None
    if best_overlap * 2 < token_len:
        return None
    return best_index


def _chunk_ids_to_tags(chunk_ids: Sequence[Optional[int]], label: str) -> List[str]:
    tags = ["O"] * len(chunk_ids)
    index = 0
    while index < len(chunk_ids):
        chunk_id = chunk_ids[index]
        if chunk_id is None:
            index += 1
            continue
        end = index + 1
        while end < len(chunk_ids) and chunk_ids[end] == chunk_id:
            end += 1
        span_len = end - index
        if span_len == 1:
            tags[index] = f"S-{label}"
        else:
            tags[index] = f"B-{label}"
            for inner in range(index + 1, end - 1):
                tags[inner] = f"I-{label}"
            tags[end - 1] = f"E-{label}"
        index = end
    return tags


def _infer_primary_label(tag_lists: Iterable[Sequence[str]]) -> str:
    labels = Counter()
    for tag_list in tag_lists:
        for tag in tag_list:
            if tag == "O" or "-" not in tag:
                continue
            labels[tag.split("-", 1)[1]] += 1
    if not labels:
        return "NP"
    return labels.most_common(1)[0][0]


def _jieba_predict_tag_lists(word_lists: Sequence[Sequence[str]], label: str) -> List[List[str]]:
    pred_tag_lists: List[List[str]] = []
    for tokens in word_lists:
        token_spans = _build_token_spans(tokens)
        chunk_spans = _extract_jieba_np_spans(tokens)
        chunk_ids = [_best_chunk_index(token_span, chunk_spans) for token_span in token_spans]
        pred_tag_lists.append(_chunk_ids_to_tags(chunk_ids, label))
    return pred_tag_lists


def run_compare(data_dir: str, output_root: str, run_name: Optional[str] = None):
    ctx = RunContext("compare", output_root, "compare", run_name)
    ctx.save_config(
        {
            "data_dir": data_dir,
            "output_root": output_root,
            "models": ["jieba", "hmm", "crf", "bilstm", "bilstm_crf"],
        }
    )

    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", data_dir=data_dir)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=data_dir)
    primary_label = _infer_primary_label(list(train_tag_lists) + list(test_tag_lists))
    results = {}
    skipped = {}

    try:
        jieba_pred_tags = _jieba_predict_tag_lists(test_word_lists, primary_label)
        results["jieba"] = chunking_report(test_tag_lists, jieba_pred_tags)
    except Exception as exc:
        skipped["jieba"] = str(exc)

    try:
        crf_model = load_pickle(find_latest_artifact(output_root, "crf", "crf"))
        crf_pred_tags = crf_model.test(test_word_lists)
        results["crf"] = chunking_report(test_tag_lists, crf_pred_tags)
    except Exception as exc:
        skipped["crf"] = str(exc)

    try:
        hmm_model = load_pickle(find_latest_artifact(output_root, "hmm", "hmm"))
        hmm_pred_tags = hmm_model.test(test_word_lists, word2id, tag2id)
        results["hmm"] = chunking_report(test_tag_lists, hmm_pred_tags)
    except Exception as exc:
        skipped["hmm"] = str(exc)

    try:
        bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
        bilstm_model = load_pickle(find_latest_artifact(output_root, "bilstm", "bilstm"))
        _maybe_flatten_bilstm(bilstm_model)
        bilstm_pred_tags, _ = bilstm_model.predict(test_word_lists, test_tag_lists, bilstm_word2id, bilstm_tag2id)
        results["bilstm"] = chunking_report(test_tag_lists, bilstm_pred_tags)
    except Exception as exc:
        skipped["bilstm"] = str(exc)

    try:
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        bilstm_crf_model = load_pickle(find_latest_artifact(output_root, "bilstm_crf", "bilstm_crf"))
        _maybe_flatten_bilstm(bilstm_crf_model)
        test_words_crf, test_tags_crf = preprocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)
        bilstm_crf_pred_tags, _ = bilstm_crf_model.predict(test_words_crf, test_tags_crf, crf_word2id, crf_tag2id)
        results["bilstm_crf"] = chunking_report(test_tag_lists, bilstm_crf_pred_tags)
    except Exception as exc:
        skipped["bilstm_crf"] = str(exc)

    if not results:
        raise RuntimeError("没有可用于对比的结果，请先训练相关模型。")

    table = render_chunking_table(results)
    if skipped:
        skipped_lines = ["", "skipped:"]
        for model_name, reason in skipped.items():
            skipped_lines.append(f"- {model_name}: {reason}")
        table = table + "\n" + "\n".join(skipped_lines)

    ctx.log(table)
    ctx.save_json(ctx.reports_dir / "compare_metrics.json", results)
    ctx.save_text(ctx.reports_dir / "compare_metrics.txt", table + "\n")
    ctx.save_summary({"model_type": "compare", "models": list(results.keys()), "skipped": skipped})
    return {"results": results, "skipped": skipped, "table": table, "run_dir": str(ctx.run_dir)}


def main():
    parser = argparse.ArgumentParser(description="对比 jieba、CRF、HMM、BiLSTM、BiLSTM-CRF 的 BaseNP 效果")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="数据集目录")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--run-name", default=None, help="运行名称")
    args = parser.parse_args()
    run_compare(args.data_dir, args.output_root, args.run_name)



if __name__ == "__main__":
    main()
