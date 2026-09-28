#!/usr/bin/env bash
# ==============================================================================
# sokuto モデル成果物ダウンロードスクリプト
#
# Hugging Face Hub から最適化済み ONNX モデル・トークナイザー・較正設定を取得し,
# models/ ディレクトリ配下に自動配置する。
# ==============================================================================

set -euo pipefail

DEFAULT_TIER="tier2"
REVISION="main"
TIER="${1:-$DEFAULT_TIER}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# scripts/ サブディレクトリ内にある場合は親ディレクトリ、パッケージ直下等の場合は自身を ROOT_DIR とする
if [[ "$(basename "${SCRIPT_DIR}")" == "scripts" ]]; then
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  ROOT_DIR="${SCRIPT_DIR}"
fi
MODELS_DIR="${ROOT_DIR}/models"

case "${TIER}" in
  tier1|130m)
    REPO_ID="MI-1222/sokuto-ja-130m-int8"
    TARGET_DIR="${MODELS_DIR}/quantized"
    TIER_LABEL="Tier 1 (130M-INT8)"
    ;;
  tier2|310m)
    REPO_ID="MI-1222/sokuto-ja-310m-int8"
    TARGET_DIR="${MODELS_DIR}/modernbert-310m-int8"
    TIER_LABEL="Tier 2 (310M-INT8)"
    ;;
  all)
    "${BASH_SOURCE[0]}" tier1
    "${BASH_SOURCE[0]}" tier2
    exit 0
    ;;
  *)
    echo "[ERROR] 未知の Tier '${TIER}'。利用可能な値: tier1, tier2, all" >&2
    exit 1
    ;;
esac

echo "=========================================================="
echo " sokuto モデル成果物ダウンロード: ${TIER_LABEL}"
echo " リポジトリ : https://huggingface.co/${REPO_ID} (${REVISION})"
echo " 保存先     : ${TARGET_DIR}"
echo "=========================================================="

mkdir -p "${TARGET_DIR}"

FILES=(
  "model.onnx"
  "tokenizer.json"
  "tokenizer_config.json"
  "config.json"
  "calibration.json"
  "quantize_metadata.json"
)

# 1. hf / huggingface-cli による並列ダウンロード
if command -v hf &> /dev/null && hf download --help &> /dev/null; then
  echo "[*] hf cli を検出しました。並列ダウンロードを開始します..."
  hf download "${REPO_ID}" \
    --revision "${REVISION}" \
    --local-dir "${TARGET_DIR}"
elif command -v huggingface-cli &> /dev/null; then
  echo "[*] huggingface-cli を検出しました。ダウンロードを開始します..."
  huggingface-cli download "${REPO_ID}" \
    --revision "${REVISION}" \
    --local-dir "${TARGET_DIR}"
else
  # 2. curl による直接ダウンロード (Location リダイレクト追従のため -L 必須)
  echo "[*] curl を使用して Hugging Face Hub から直接ダウンロードします..."
  for file in "${FILES[@]}"; do
    TARGET_FILE="${TARGET_DIR}/${file}"
    URL="https://huggingface.co/${REPO_ID}/resolve/${REVISION}/${file}"
    echo "[-] ダウンロード中: ${file} ..."
    curl -L --fail --progress-bar -o "${TARGET_FILE}" "${URL}" || {
      echo "[WARN] ${file} の取得をスキップしました (リポジトリに存在しない可能性があります)。"
      rm -f "${TARGET_FILE}"
    }
  done
fi

echo "=========================================================="
echo "[SUCCESS] モデル成果物の配置が完了しました: ${TARGET_DIR}"
ls -lh "${TARGET_DIR}"
echo "=========================================================="
