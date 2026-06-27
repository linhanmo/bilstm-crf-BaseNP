import argparse

from config import run_hmm_training
from training import HMMTrainConfig


def main():
    defaults = HMMTrainConfig.default()
    parser = argparse.ArgumentParser(description="训练 BaseNP HMM 模型")
    parser.add_argument("--data-dir", default=defaults.data_dir, help="BaseNP 数据集目录")
    parser.add_argument("--output-root", default=defaults.output_root, help="输出根目录")
    parser.add_argument("--model-name", default=defaults.model_name, help="模型名称")
    parser.add_argument("--run-name", default=None, help="运行名称")
    parser.add_argument("--remove-o", action="store_true", help="评估时移除 O 标签")
    args = parser.parse_args()

    config = HMMTrainConfig(
        data_dir=args.data_dir,
        output_root=args.output_root,
        model_name=args.model_name,
        run_name=args.run_name,
        remove_o=args.remove_o,
    )
    run_hmm_training(config)


if __name__ == "__main__":
    main()
