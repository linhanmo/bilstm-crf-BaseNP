import argparse

from config import run_crf_training
from training import CRFTrainConfig


def main():
    defaults = CRFTrainConfig.default()
    parser = argparse.ArgumentParser(description="训练 BaseNP CRF 模型")
    parser.add_argument("--data-dir", default=defaults.data_dir, help="BaseNP 数据集目录")
    parser.add_argument("--output-root", default=defaults.output_root, help="输出根目录")
    parser.add_argument("--model-name", default=defaults.model_name, help="模型名称")
    parser.add_argument("--run-name", default=None, help="运行名称")
    parser.add_argument("--remove-o", action="store_true", help="评估时移除 O 标签")
    parser.add_argument("--algorithm", default=defaults.algorithm, help="CRF 优化算法")
    parser.add_argument("--c1", type=float, default=defaults.c1, help="CRF L1 正则")
    parser.add_argument("--c2", type=float, default=defaults.c2, help="CRF L2 正则")
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations, help="CRF 最大迭代次数")
    parser.add_argument("--all-possible-transitions", action="store_true", help="是否启用所有转移")
    args = parser.parse_args()

    config = CRFTrainConfig(
        data_dir=args.data_dir,
        output_root=args.output_root,
        model_name=args.model_name,
        run_name=args.run_name,
        remove_o=args.remove_o,
        algorithm=args.algorithm,
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations,
        all_possible_transitions=args.all_possible_transitions,
    )
    run_crf_training(config)


if __name__ == "__main__":
    main()
