"""Hugging Face Hub へ sokuto モデル成果物をアップロードするスクリプト."""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

TIER_CONFIG = {
    "tier1": {
        "repo_id": "MI-1222/sokuto-ja-130m-int8",
        "local_dir": ROOT_DIR / "models" / "quantized",
        "title": "sokuto-ja-130m-int8 (Tier 1: 超低遅延 130M-INT8)",
        "backbone": "sbintuitions/modernbert-ja-130m",
        "ece": "6.29%",
    },
    "tier2": {
        "repo_id": "MI-1222/sokuto-ja-310m-int8",
        "local_dir": ROOT_DIR / "models" / "modernbert-310m-int8",
        "title": "sokuto-ja-310m-int8 (Tier 2: 標準高精度 310M-INT8)",
        "backbone": "sbintuitions/modernbert-ja-310m",
        "ece": "2.61%",
    },
}


def generate_model_card(tier: str) -> str:
    """指定 Tier のモデルカード (README.md) の内容を生成する."""
    conf = TIER_CONFIG[tier]
    return f"""---
language:
  - ja
license: mit
tags:
  - onnx
  - modernbert
  - zero-shot
  - classification
  - system-one
  - sokuto
pipeline_tag: text-classification
---

# {conf["title"]}

日本語特化・TypeSafe AI「Jev (System 1)」のローカル推論基盤 [`sokuto`](https://github.com/MI-1222/local-jev) 用の公式モデル成果物です。
文章生成を行わず、日本語ネイティブ (ModernBERT-ja) の単一フォワードパスで型付き確率決定 (Choice / Score / Noul) をミリ秒単位で返します。

## モデル仕様
- **バックボーン**: `{conf["backbone"]}`
- **量子化**: ハイブリッド動的 INT8 (バックボーンのみ INT8 化、ヘッドは FP32 保持)
- **較正誤差 (ECE)**: {conf["ece"]}

## 同梱ファイル
- `model.onnx`: INT8 量子化済み ONNX 推論モデル
- `tokenizer.json`, `tokenizer_config.json`: トークナイザー定義・設定
- `config.json`: モデルアーキテクチャ設定
- `calibration.json`: 事後温度スケーリング較正パラメータ
- `quantize_metadata.json`: 量子化メタデータ
"""


def upload_tier(tier: str, tag: str | None = None) -> None:
    """指定 Tier のモデル成果物を Hugging Face Hub にアップロードする."""
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    conf = TIER_CONFIG[tier]
    repo_id = conf["repo_id"]
    local_dir = conf["local_dir"]

    if not local_dir.exists():
        raise FileNotFoundError(f"モデルディレクトリが見つかりません: {local_dir}")

    print(f"[*] リポジトリを確認/作成中: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)

    # README.md (モデルカード) の自動配置
    readme_path = local_dir / "README.md"
    if not readme_path.exists():
        readme_path.write_text(generate_model_card(tier), encoding="utf-8")

    print(f"[*] {local_dir} 内のファイルを {repo_id} にアップロード中...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Release {tag or 'latest'} model artifacts",
    )

    if tag:
        print(f"[*] リリースタグ {tag} を作成中...")
        api.create_tag(repo_id=repo_id, tag=tag, repo_type="model")

    print(f"[✓] アップロード完了: https://huggingface.co/{repo_id}")


def main() -> None:
    """CLI エントリーポイント."""
    parser = argparse.ArgumentParser(
        description="Upload sokuto models to Hugging Face Hub"
    )
    parser.add_argument("--tier", choices=["tier1", "tier2", "all"], default="all")
    parser.add_argument(
        "--tag", type=str, help="Release tag (例: v0.3.4)", default=None
    )
    args = parser.parse_args()

    tiers = ["tier1", "tier2"] if args.tier == "all" else [args.tier]
    for t in tiers:
        upload_tier(t, tag=args.tag)


if __name__ == "__main__":
    main()
