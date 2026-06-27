import argparse

from config import (
    RunContext,
    build_corpus,
    extend_maps,
    find_latest_artifact,
    load_pickle,
    preprocess_data_for_lstmcrf,
    render_segmentation_table,
    segmentation_report,
    tags_to_words,
)
from training import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_ROOT


def _load_jieba():
    try:
        import jieba
    except ImportError as exc:
        raise ImportError("对比脚本依赖 jieba，请先安装 jieba。") from exc
    return jieba


def _maybe_flatten_bilstm(model):
    try:
        model.model.bilstm.flatten_parameters()
    except AttributeError:
        pass
    try:
        model.model.bilstm.bilstm.flatten_parameters()
    except AttributeError:
        pass


def main():
    parser = argparse.ArgumentParser(description="对比 jieba、CRF、HMM、BiLSTM、BiLSTM-CRF 的分词效果")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="数据集目录")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--run-name", default=None, help="运行名称")
    args = parser.parse_args()

    ctx = RunContext("compare", args.output_root, "compare", args.run_name)
    ctx.save_config(
        {
            "data_dir": args.data_dir,
            "output_root": args.output_root,
            "models": ["jieba", "hmm", "crf", "bilstm", "bilstm_crf"],
        }
    )

    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train", data_dir=args.data_dir)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False, data_dir=args.data_dir)
    gold_word_lists = [tags_to_words(chars, tags) for chars, tags in zip(test_word_lists, test_tag_lists)]
    results = {}
    skipped = {}

    try:
        jieba = _load_jieba()
        jieba_word_lists = [jieba.lcut("".join(chars), HMM=True) for chars in test_word_lists]
        results["jieba"] = segmentation_report(gold_word_lists, jieba_word_lists)
    except Exception as exc:
        skipped["jieba"] = str(exc)

    try:
        crf_model = load_pickle(find_latest_artifact(args.output_root, "crf", "crf"))
        crf_pred_tags = crf_model.test(test_word_lists)
        crf_word_lists = [tags_to_words(chars, tags) for chars, tags in zip(test_word_lists, crf_pred_tags)]
        results["crf"] = segmentation_report(gold_word_lists, crf_word_lists)
    except Exception as exc:
        skipped["crf"] = str(exc)

    try:
        hmm_model = load_pickle(find_latest_artifact(args.output_root, "hmm", "hmm"))
        hmm_pred_tags = hmm_model.test(test_word_lists, word2id, tag2id)
        hmm_word_lists = [tags_to_words(chars, tags) for chars, tags in zip(test_word_lists, hmm_pred_tags)]
        results["hmm"] = segmentation_report(gold_word_lists, hmm_word_lists)
    except Exception as exc:
        skipped["hmm"] = str(exc)

    try:
        bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=False)
        bilstm_model = load_pickle(find_latest_artifact(args.output_root, "bilstm", "bilstm"))
        _maybe_flatten_bilstm(bilstm_model)
        bilstm_pred_tags, _ = bilstm_model.predict(test_word_lists, test_tag_lists, bilstm_word2id, bilstm_tag2id)
        bilstm_word_lists = [tags_to_words(chars, tags) for chars, tags in zip(test_word_lists, bilstm_pred_tags)]
        results["bilstm"] = segmentation_report(gold_word_lists, bilstm_word_lists)
    except Exception as exc:
        skipped["bilstm"] = str(exc)

    try:
        crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
        bilstm_crf_model = load_pickle(find_latest_artifact(args.output_root, "bilstm_crf", "bilstm_crf"))
        _maybe_flatten_bilstm(bilstm_crf_model)
        test_words_crf, test_tags_crf = preprocess_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)
        bilstm_crf_pred_tags, _ = bilstm_crf_model.predict(test_words_crf, test_tags_crf, crf_word2id, crf_tag2id)
        bilstm_crf_word_lists = [tags_to_words(chars, tags) for chars, tags in zip(test_word_lists, bilstm_crf_pred_tags)]
        results["bilstm_crf"] = segmentation_report(gold_word_lists, bilstm_crf_word_lists)
    except Exception as exc:
        skipped["bilstm_crf"] = str(exc)

    if not results:
        raise RuntimeError("没有可用于对比的结果，请先训练相关模型或安装 jieba。")

    table = render_segmentation_table(results)
    if skipped:
        skipped_lines = ["", "skipped:"]
        for model_name, reason in skipped.items():
            skipped_lines.append(f"- {model_name}: {reason}")
        table = table + "\n" + "\n".join(skipped_lines)

    ctx.log(table)
    ctx.save_json(ctx.reports_dir / "compare_metrics.json", results)
    ctx.save_text(ctx.reports_dir / "compare_metrics.txt", table + "\n")
    ctx.save_summary({"model_type": "compare", "models": list(results.keys()), "skipped": skipped})


if __name__ == "__main__":
    main()
