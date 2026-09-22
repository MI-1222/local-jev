#!/usr/bin/env bash
# ==============================================================================
# Local-Jev エアギャップ (外部ネットワーク完全遮断) 起動検証スクリプト
#
# Docker の `--network none` フラグを用いてコンテナの外部通信を物理的に遮断し、
# 外部名前解決や Hugging Face Hub 等への暗黙の通信を一切行わずに
# モデル推論が 100% 自己完結して動作することを検証する。
# ==============================================================================

set -euo pipefail

IMAGE_NAME="${1:-local-jev:cpu}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models/default"

echo "=================================================================="
echo "Local-Jev エアギャップ検証テストを開始します"
echo "イメージ: ${IMAGE_NAME}"
echo "モデルパス: ${MODEL_DIR}"
echo "=================================================================="

# 1. モデルディレクトリの存在確認
if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "[ERROR] モデルディレクトリが見つかりません: ${MODEL_DIR}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/model.onnx" || ! -f "${MODEL_DIR}/tokenizer.json" ]]; then
    echo "[ERROR] 必須モデル成果物 (model.onnx, tokenizer.json) が不足しています。" >&2
    exit 1
fi

# 2. イメージの存在確認 (存在しない場合はビルドを案内)
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "[INFO] イメージ '${IMAGE_NAME}' がローカルに存在しないため、ビルドを実行します..."
    docker build -t "${IMAGE_NAME}" -f "${ROOT_DIR}/docker/Dockerfile.cpu" "${ROOT_DIR}"
fi

# 3. 外部ネットワーク完全遮断 (--network none) 下でのインプロセス推論テスト
echo ""
echo ">>> Step 1: --network none でのインプロセス推論ベンチマーク実行"
docker run --rm \
    --network none \
    -v "${MODEL_DIR}:/models/default:ro" \
    "${IMAGE_NAME}" \
    benchmark \
        --model-dir /models/default \
        --iterations 5 \
        --warmup 2 \
        --scenario single \
        --provider cpu

echo "[SUCCESS] エアギャップ環境でのインプロセス推論に成功しました。"

# 4. バックグラウンドサーバー起動とローカルヘルスチェック検証
echo ""
echo ">>> Step 2: 独立ネットワーク名前空間でのサーバー起動 & ヘルスチェック検証"
CONTAINER_NAME="local-jev-airgap-test-$$"

# コンテナ起動 (外部遮断ネットワーク)
docker run -d \
    --name "${CONTAINER_NAME}" \
    --network none \
    -v "${MODEL_DIR}:/models/default:ro" \
    "${IMAGE_NAME}" \
    serve \
        --host 127.0.0.1 \
        --port 3000 \
        --model-dir /models/default

cleanup() {
    echo "[INFO] テストコンテナ '${CONTAINER_NAME}' を停止・削除します..."
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# サーバーの立ち上がりを待機 (最大 15 秒)
echo "[INFO] サーバーの初期化完了を待機しています..."
ATTEMPTS=0
MAX_ATTEMPTS=15
HEALTHY=false

while [[ ${ATTEMPTS} -lt ${MAX_ATTEMPTS} ]]; do
    sleep 1
    ATTEMPTS=$((ATTEMPTS + 1))
    
    # コンテナ内部からヘルスチェックを実行 (外部ネットワークゼロを保証)
    if docker exec "${CONTAINER_NAME}" /usr/local/bin/local-jev healthcheck --url http://127.0.0.1:3000/ready >/dev/null 2>&1; then
        HEALTHY=true
        break
    fi
done

if [[ "${HEALTHY}" == "true" ]]; then
    echo "[SUCCESS] エアギャップ環境で サーバーの起動および /ready エンドポイントの正常応答を確認しました。"
else
    echo "[ERROR] サーバーの初期化がタイムアウトしました。コンテナログを出力します:" >&2
    docker logs "${CONTAINER_NAME}" >&2
    exit 1
fi

echo ""
echo "=================================================================="
echo "すべてのエアギャップ起動検証テストに合格しました。"
echo "=================================================================="
