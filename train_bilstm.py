import argparse

from config import run_bilstm_training
from training import LSTMTrainConfig


def main():
    defaults = LSTMTrainConfig.default("bilstm")
    parser = argparse.ArgumentParser(description="训练 BaseNP BiLSTM 模型")
    parser.add_argument("--data-dir", default=defaults.data_dir, help="BaseNP 数据集目录")
    parser.add_argument("--output-root", default=defaults.output_root, help="输出根目录")
    parser.add_argument("--model-name", default=defaults.model_name, help="模型名称")
    parser.add_argument("--run-name", default=None, help="运行名称")
    parser.add_argument("--remove-o", action="store_true", help="评估时移除 O 标签")
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="batch size")
    parser.add_argument("--lr", type=float, default=defaults.lr, help="学习率")
    parser.add_argument("--epoches", type=int, default=defaults.epoches, help="训练轮数")
    parser.add_argument("--print-step", type=int, default=defaults.print_step, help="日志打印步数")
    parser.add_argument("--emb-size", type=int, default=defaults.emb_size, help="词向量维度")
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size, help="隐层维度")
    args = parser.parse_args()

    config = LSTMTrainConfig(
        data_dir=args.data_dir,
        output_root=args.output_root,
        model_name=args.model_name,
        run_name=args.run_name,
        remove_o=args.remove_o,
        batch_size=args.batch_size,
        lr=args.lr,
        epoches=args.epoches,
        print_step=args.print_step,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
    )
    run_bilstm_training(config, use_crf=False)


if __name__ == "__main__":
    main()
