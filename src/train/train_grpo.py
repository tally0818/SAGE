from .common import build_parser, run_training


def main():
    parser = build_parser("Train plain GRPO with TRL")
    args = parser.parse_args()
    run_training(args, use_sage=False)


if __name__ == "__main__":
    main()
