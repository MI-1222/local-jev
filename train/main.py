"""Local-Jev 学習パイプラインの統一エントリーポイントモジュール。

CLI 引数を解析し、SFT (教師あり指示学習) パイプラインを実行する。
"""

import sys

from training.run_sft import build_config_from_args, parse_args, run_sft_pipeline


def main() -> None:
    """学習スクリプトのメイン実行関数。"""
    args = parse_args(sys.argv[1:])
    config = build_config_from_args(args)
    run_sft_pipeline(config)


if __name__ == "__main__":
    main()
