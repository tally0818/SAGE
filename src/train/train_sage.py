from .common import build_parser, run_training


def main():
    parser = build_parser("Train SAGE with TRL")
    args = parser.parse_args()
    run_training(args, use_sage=True)


if __name__ == "__main__":
    main()
