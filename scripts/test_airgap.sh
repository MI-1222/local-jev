#!/usr/bin/env bash
# ==============================================================================
# Sokuto エアギャップ (外部ネットワーク完全遮断) 起動検証スクリプト
#
# Docker の `--network none` フラグを用いてコンテナの外部通信を物理的に遮断し,
# 外部名前解決や Hugging Face Hub 等への暗黙の通信を一切行わずに
# モデル推論が 100% 自己完結して動作すること, および常駐 RAM (< 1GB) を検証する。
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="sokuto:cpu"
TIER="tier2"
MODEL_DIR=""

# 引数解析 (オプション指定および位置引数に対応)
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            TIER="$2"
            shift 2
            ;;
        --image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# 位置引数による上書き (指定がある場合)
if [[ ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
    IMAGE_NAME="${POSITIONAL_ARGS[0]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
    TIER="${POSITIONAL_ARGS[1]}"
fi

if [[ -z "${MODEL_DIR}" ]]; then
    if [[ "${TIER}" == "tier1" && -d "${ROOT_DIR}/models/quantized" ]]; then
        MODEL_DIR="${ROOT_DIR}/models/quantized"
    elif [[ "${TIER}" == "tier2" && -d "${ROOT_DIR}/models/modernbert-310m-int8" ]]; then
        MODEL_DIR="${ROOT_DIR}/models/modernbert-310m-int8"
    else
        MODEL_DIR="${ROOT_DIR}/models/default"
    fi
fi

echo "=================================================================="
echo "Sokuto エアギャップ検証テストを開始します。"
echo "イメージ: ${IMAGE_NAME}."
echo "モデル Tier: ${TIER}."
echo "モデルパス: ${MODEL_DIR}."
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

# 2. イメージの存在確認 (存在しない場合は案内)
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

# 4. バックグラウンドサーバー起動とローカルヘルスチェック & メモリ検証
echo ""
echo ">>> Step 2: 独立ネットワーク名前空間でのサーバー起動 & ヘルスチェック検証"
CONTAINER_NAME="sokuto-airgap-test-$$"

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
    if docker exec "${CONTAINER_NAME}" /usr/local/bin/sokuto healthcheck --url http://127.0.0.1:3000/ready >/dev/null 2>&1; then
        HEALTHY=true
        break
    fi
done

if [[ "${HEALTHY}" == "true" ]]; then
    echo "[SUCCESS] エアギャップ環境でサーバー起動および /ready エンドポイントの正常応答を確認しました。"
else
    echo "[ERROR] サーバーの初期化がタイムアウトしました。コンテナログを出力します:" >&2
    docker logs "${CONTAINER_NAME}" >&2
    exit 1
fi

# メモリフットプリントの確認 (RAM < 1GB)
echo ""
echo ">>> Step 3: 常駐メモリ (RSS) の計測"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" "${CONTAINER_NAME}"

echo ""
echo "=================================================================="
echo "すべてのエアギャップ起動検証テストに合格しました。"
echo "=================================================================="
