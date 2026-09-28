#!/usr/bin/env bash
# ==============================================================================
# Sokuto オフライン配布パッケージ生成スクリプト
#
# エアギャップ環境やオンプレミス環境へ持ち込むための配布アーカイブを自動生成する。
# Dual-Tier モデル構成 (Tier 1: 130M-INT8 / Tier 2: 310M-INT8) に対応し,
# コンテナイメージ (tar.gz), モデル成果物, Compose 設定, デプロイスクリプト,
# チェックサムを同梱した自己完結型 tar.gz を出力する。
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_VERSION="v$(sed -n 's/^version = "\(.*\)"/\1/p' "${ROOT_DIR}/Cargo.toml" | head -n 1)"
VERSION="${DEFAULT_VERSION}"
FLAVOR="cpu"
OUTPUT_BASE_DIR="${ROOT_DIR}/dist"
TIER="tier2"

# 引数解析 (位置引数およびオプション指定の両方に対応)
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)
            TIER="$2"
            shift 2
            ;;
        --flavor)
            FLAVOR="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --out-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# 位置引数があれば上書き
if [[ ${#POSITIONAL_ARGS[@]} -ge 1 ]]; then
    VERSION="${POSITIONAL_ARGS[0]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 2 ]]; then
    FLAVOR="${POSITIONAL_ARGS[1]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 3 ]]; then
    OUTPUT_BASE_DIR="${POSITIONAL_ARGS[2]}"
fi
if [[ ${#POSITIONAL_ARGS[@]} -ge 4 ]]; then
    TIER="${POSITIONAL_ARGS[3]}"
fi

PACKAGE_NAME="sokuto-${VERSION}-${FLAVOR}-${TIER}"
PACKAGE_DIR="${OUTPUT_BASE_DIR}/${PACKAGE_NAME}"
IMAGE_TAG="sokuto:${FLAVOR}"
TIER_UPPER=$(echo "${TIER}" | tr '[:lower:]' '[:upper:]')

echo "=================================================================="
echo "Sokuto オフライン配布パッケージの生成を開始します。"
echo "バージョン: ${VERSION}."
echo "フレーバー: ${FLAVOR}."
echo "モデル Tier: ${TIER}."
echo "出力先: ${PACKAGE_DIR}."
echo "=================================================================="

# 1. モデル成果物の解決 (Tier 1 vs Tier 2)
SRC_MODEL_DIR=""
if [[ "${TIER}" == "tier1" ]]; then
    if [[ -d "${ROOT_DIR}/models/quantized" && -f "${ROOT_DIR}/models/quantized/model.onnx" ]]; then
        SRC_MODEL_DIR="${ROOT_DIR}/models/quantized"
    fi
elif [[ "${TIER}" == "tier2" ]]; then
    if [[ -d "${ROOT_DIR}/models/modernbert-310m-int8" && -f "${ROOT_DIR}/models/modernbert-310m-int8/model.onnx" ]]; then
        SRC_MODEL_DIR="${ROOT_DIR}/models/modernbert-310m-int8"
    fi
fi

# フォールバック
if [[ -z "${SRC_MODEL_DIR}" ]]; then
    if [[ -d "${ROOT_DIR}/models/default" && -f "${ROOT_DIR}/models/default/model.onnx" ]]; then
        SRC_MODEL_DIR="${ROOT_DIR}/models/default"
        echo "[WARN] 指定 Tier (${TIER}) の専用ディレクトリが見つからないため、models/default を使用します。"
    else
        echo "[ERROR] 有効なモデル成果物が見つかりません。train/export/quantize.py を実行してください。" >&2
        exit 1
    fi
fi

echo "[INFO] 採用モデルソース: ${SRC_MODEL_DIR}"

# 2. コンテナイメージの存在確認 (なければビルド)
if [[ "${FLAVOR}" != "binary" ]]; then
    if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
        echo "[INFO] イメージ '${IMAGE_TAG}' をビルドします..."
        if [[ "${FLAVOR}" == "allinone" ]]; then
            docker build \
                --build-arg MODEL_TIER="${TIER}" \
                -t "${IMAGE_TAG}" \
                -f "${ROOT_DIR}/docker/Dockerfile.allinone" \
                "${ROOT_DIR}"
        elif [[ "${FLAVOR}" == "cuda" || "${FLAVOR}" == "gpu" ]]; then
            docker build -t "${IMAGE_TAG}" -f "${ROOT_DIR}/docker/Dockerfile.cuda" "${ROOT_DIR}"
        else
            docker build -t "${IMAGE_TAG}" -f "${ROOT_DIR}/docker/Dockerfile.cpu" "${ROOT_DIR}"
        fi
    fi
fi

# 3. 配布ディレクトリの作成
mkdir -p "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/models/default"

# 4. コンテナイメージアーカイブのエクスポート (tar.gz) またはバイナリの配置
if [[ "${FLAVOR}" != "binary" ]]; then
    ARCHIVE_PATH="${PACKAGE_DIR}/${PACKAGE_NAME}-image.tar.gz"
    echo "[INFO] Docker イメージをエクスポート中: ${ARCHIVE_PATH}..."
    docker save "${IMAGE_TAG}" | gzip > "${ARCHIVE_PATH}"
else
    echo "[INFO] ネイティブバイナリを収集・配置中..."
    mkdir -p "${PACKAGE_DIR}/bin" "${PACKAGE_DIR}/lib"
    BIN_SRC=""
    if [[ -f "${ROOT_DIR}/target/release/sokuto" ]]; then
        BIN_SRC="${ROOT_DIR}/target/release/sokuto"
    elif [[ -f "${ROOT_DIR}/target/debug/sokuto" ]]; then
        BIN_SRC="${ROOT_DIR}/target/debug/sokuto"
    fi

    if [[ -n "${BIN_SRC}" ]]; then
        cp "${BIN_SRC}" "${PACKAGE_DIR}/bin/"
        chmod +x "${PACKAGE_DIR}/bin/sokuto"
    else
        echo "[WARN] target 配下に sokuto バイナリが見つかりません。cargo build --release を実行してください。" >&2
    fi

    # 共有ライブラリの収集
    find "${ROOT_DIR}/target" -name "libonnxruntime*.so*" -exec cp -a {} "${PACKAGE_DIR}/lib/" \; 2>/dev/null || true
    find "${ROOT_DIR}/target" -name "libonnxruntime*.dylib" -exec cp -a {} "${PACKAGE_DIR}/lib/" \; 2>/dev/null || true
fi

# 5. モデル成果物の同梱 (allinone 以外の場合、または検証用)
echo "[INFO] モデル成果物を同梱中..."
cp -r "${SRC_MODEL_DIR}"/* "${PACKAGE_DIR}/models/default/"

# 6. 配布用設定および起動スクリプトの配置
if [[ "${FLAVOR}" != "binary" ]]; then
    echo "[INFO] 配布用 Compose テンプレートを配置中..."
    cat << EOF > "${PACKAGE_DIR}/docker-compose.yml"
services:
  sokuto:
    image: ${IMAGE_TAG}
    container_name: sokuto
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - SOKUTO_HOST=0.0.0.0
      - SOKUTO_PORT=3000
      - SOKUTO_MODEL_DIR=/models/default
      - SOKUTO_POOL_SIZE=2
      - SOKUTO_INTRA_THREADS=2
      - SOKUTO_INTER_THREADS=1
      - RUST_LOG=sokuto_server=info,sokuto_cli=info,tower_http=info
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
    volumes:
      - ./models/default:/models/default:ro
    healthcheck:
      test: ["CMD", "/usr/local/bin/sokuto", "healthcheck", "--url", "http://127.0.0.1:3000/ready"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 5s
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
EOF

    # コンテナ版 deploy.sh
    cat << 'EOF' > "${PACKAGE_DIR}/deploy.sh"
#!/usr/bin/env bash
# ==============================================================================
# Sokuto オフラインコンテナデプロイスクリプト
# ==============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_FILE=$(find "${DIR}" -name "*-image.tar.gz" | head -n 1 || true)
if [[ -z "${IMAGE_FILE}" || ! -f "${IMAGE_FILE}" ]]; then
    echo "[ERROR] Docker イメージアーカイブ (*-image.tar.gz) が見つかりません。" >&2
    exit 1
fi

echo ">>> Docker イメージをロード中: ${IMAGE_FILE}..."
docker load -i "${IMAGE_FILE}"

echo ">>> サービスを起動中..."
docker compose -f "${DIR}/docker-compose.yml" up -d

echo "[SUCCESS] Sokuto サービスが正常に起動しました。"
echo "確認用エンドポイント: http://localhost:3000/ready"
EOF
    chmod +x "${PACKAGE_DIR}/deploy.sh"

    # コンテナ版 README.md
    cat << EOF > "${PACKAGE_DIR}/README.md"
# Sokuto オフライン配布パッケージ利用手順書 (${TIER_UPPER})

本パッケージは、外部インターネットから遮断されたエアギャップ環境やオンプレミス環境において、
Sokuto 推論サーバー (${TIER}) を即座に稼働させるための自己完結型配布パッケージです。

## 同梱内容
- \`*-image.tar.gz\`: Docker コンテナイメージアーカイブ。
- \`models/default/\`: 推論用モデル成果物 (${TIER}: ONNX グラフ, トークナイザー, 較正設定)。
- \`docker-compose.yml\`: 本番運用向け Compose 定義 (メモリ 1GB 制限適用)。
- \`deploy.sh\`: ワンクリック導入スクリプト。

## デプロイ手順

### 1. イメージの読み込み & 起動
付属の \`deploy.sh\` を実行します。
\`\`\`bash
./deploy.sh
\`\`\`

手動で起動する場合は以下の通りです。
\`\`\`bash
docker load -i sokuto-*-image.tar.gz
docker compose up -d
\`\`\`

### 2. 動作確認
推論エンジンの準備完了状態 (Readiness) を確認します。
\`\`\`bash
curl -i http://localhost:3000/ready
\`\`\`

### 3. 停止方法
\`\`\`bash
docker compose down
\`\`\`
EOF

else
    # ネイティブバイナリ版 deploy.sh (run.sh)
    cat << 'EOF' > "${PACKAGE_DIR}/deploy.sh"
#!/usr/bin/env bash
# ==============================================================================
# Sokuto ネイティブサーバー起動スクリプト
# ==============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LD_LIBRARY_PATH="${DIR}/lib:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="${DIR}/lib:${DYLD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo ">>> Sokuto ネイティブサーバーを起動します (ポート 3000)..."
exec "${DIR}/bin/sokuto" serve \
    --host 0.0.0.0 \
    --port 3000 \
    --model-dir "${DIR}/models/default"
EOF
    chmod +x "${PACKAGE_DIR}/deploy.sh"

    # ネイティブ版 README.md
    cat << EOF > "${PACKAGE_DIR}/README.md"
# Sokuto スタンドアロンバイナリパッケージ (${TIER_UPPER})

本パッケージには \`bin/sokuto\` 実行バイナリと、推論用モデル成果物 (${TIER}) が同梱されています。
Docker を使用せず、直接ネイティブプロセスとして実行可能です。

## 同梱内容
- \`bin/sokuto\`: CLI / サーバー実行バイナリ。
- \`lib/\`: ONNX Runtime 共有ライブラリ。
- \`models/default/\`: 推論用モデル成果物 (${TIER}: ONNX グラフ, トークナイザー, 較正設定)。
- \`deploy.sh\`: 起動スクリプト。

## 起動手順

### 1. サーバーの起動
\`\`\`bash
./deploy.sh
\`\`\`

または直接バイナリを実行:
\`\`\`bash
./bin/sokuto serve --model-dir ./models/default --port 3000
\`\`\`

### 2. 動作確認
\`\`\`bash
curl -i http://localhost:3000/ready
\`\`\`
EOF
fi

# 7. チェックサム (SHA256) の算出
echo "[INFO] チェックサムを生成中..."
(cd "${PACKAGE_DIR}" && find . -maxdepth 2 -type f ! -name "SHA256SUMS" -exec shasum -a 256 {} + > SHA256SUMS)

# 8. パッケージ全体の tar.gz アーカイブ作成
TARBALL_PATH="${OUTPUT_BASE_DIR}/${PACKAGE_NAME}.tar.gz"
echo "[INFO] パッケージ全体のアーカイブを生成中: ${TARBALL_PATH}..."
tar -C "${OUTPUT_BASE_DIR}" -czf "${TARBALL_PATH}" "${PACKAGE_NAME}"
(cd "${OUTPUT_BASE_DIR}" && shasum -a 256 "${PACKAGE_NAME}.tar.gz" > "${PACKAGE_NAME}.tar.gz.sha256")

echo "=================================================================="
echo "[SUCCESS] 配布パッケージが正常に生成されました: ${TARBALL_PATH}."
echo "=================================================================="
