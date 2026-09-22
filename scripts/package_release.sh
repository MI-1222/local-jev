#!/usr/bin/env bash
# ==============================================================================
# Local-Jev オフライン配布パッケージ生成スクリプト
#
# エアギャップ環境やオンプレミス環境へ持ち込むための配布アーカイブを自動生成する。
# コンテナイメージ (tar.gz), モデル成果物, Compose 設定, デプロイスクリプトを同梱する。
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_VERSION="v$(sed -n 's/^version = "\(.*\)"/\1/p' "${ROOT_DIR}/Cargo.toml" | head -n 1)"
VERSION="${1:-${DEFAULT_VERSION}}"
FLAVOR="${2:-cpu}"
OUTPUT_BASE_DIR="${3:-${ROOT_DIR}/dist}"

PACKAGE_NAME="local-jev-${VERSION}-${FLAVOR}"
PACKAGE_DIR="${OUTPUT_BASE_DIR}/${PACKAGE_NAME}"
IMAGE_TAG="local-jev:${FLAVOR}"

echo "=================================================================="
echo "Local-Jev オフライン配布パッケージの生成を開始します。"
echo "バージョン: ${VERSION}."
echo "フレーバー: ${FLAVOR}."
echo "出力先: ${PACKAGE_DIR}."
echo "=================================================================="

# 1. コンテナイメージの存在確認 (なければビルド)
if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "[INFO] イメージ '${IMAGE_TAG}' をビルドします..."
    if [[ "${FLAVOR}" == "allinone" ]]; then
        docker build -t "${IMAGE_TAG}" -f "${ROOT_DIR}/docker/Dockerfile.allinone" "${ROOT_DIR}"
    elif [[ "${FLAVOR}" == "cuda" || "${FLAVOR}" == "gpu" ]]; then
        docker build -t "${IMAGE_TAG}" -f "${ROOT_DIR}/docker/Dockerfile.cuda" "${ROOT_DIR}"
    else
        docker build -t "${IMAGE_TAG}" -f "${ROOT_DIR}/docker/Dockerfile.cpu" "${ROOT_DIR}"
    fi
fi

# 2. 配布ディレクトリの作成
mkdir -p "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}/models"

# 3. Docker イメージアーカイブのエクスポート (tar.gz)
ARCHIVE_PATH="${PACKAGE_DIR}/${PACKAGE_NAME}-image.tar.gz"
echo "[INFO] Docker イメージをエクスポート中: ${ARCHIVE_PATH}..."
docker save "${IMAGE_TAG}" | gzip > "${ARCHIVE_PATH}"

# 4. モデル成果物の同梱 (allinone 以外の場合)
if [[ "${FLAVOR}" != "allinone" ]]; then
    echo "[INFO] モデル成果物 (models/default) を同梱中..."
    cp -r "${ROOT_DIR}/models/default" "${PACKAGE_DIR}/models/"
fi

# 5. 配布用 docker-compose.yml の配置
echo "[INFO] 配布用 Compose テンプレートを配置中..."
cat << 'EOF' > "${PACKAGE_DIR}/docker-compose.yml"
services:
  local-jev:
    image: local-jev:cpu
    container_name: local-jev
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - LOCAL_JEV_HOST=0.0.0.0
      - LOCAL_JEV_PORT=3000
      - LOCAL_JEV_MODEL_DIR=/models/default
      - LOCAL_JEV_POOL_SIZE=2
      - LOCAL_JEV_INTRA_THREADS=2
      - LOCAL_JEV_INTER_THREADS=1
      - RUST_LOG=local_jev_server=info,local_jev_cli=info,tower_http=info
      - HF_HUB_OFFLINE=1
      - TRANSFORMERS_OFFLINE=1
    volumes:
      - ./models/default:/models/default:ro
    healthcheck:
      test: ["CMD", "/usr/local/bin/local-jev", "healthcheck", "--url", "http://127.0.0.1:3000/ready"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 5s
EOF

# 6. オフライン環境ワンクリックデプロイスクリプト (deploy.sh) の生成
cat << 'EOF' > "${PACKAGE_DIR}/deploy.sh"
#!/usr/bin/env bash
# ==============================================================================
# Local-Jev オフラインデプロイスクリプト
# ==============================================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Docker イメージをロード中..."
IMAGE_FILE=$(find "${DIR}" -name "*-image.tar.gz" | head -n 1)
if [[ -z "${IMAGE_FILE}" ]]; then
    echo "[ERROR] イメージアーカイブが見つかりません。" >&2
    exit 1
fi
docker load -i "${IMAGE_FILE}"

echo ">>> サービスを起動中..."
docker compose -f "${DIR}/docker-compose.yml" up -d

echo "[SUCCESS] Local-Jev サービスが正常に起動しました。"
echo "確認用エンドポイント: http://localhost:3000/ready"
EOF
chmod +x "${PACKAGE_DIR}/deploy.sh"

# 7. オフライン環境向けデプロイ手順書 (README.md) の生成
cat << 'EOF' > "${PACKAGE_DIR}/README.md"
# Local-Jev オフライン配布パッケージ利用手順書

本パッケージは、外部インターネットから遮断されたエアギャップ環境やオンプレミス環境において、
Local-Jev 推論サーバーを即座に稼働させるための自己完結型配布パッケージです。

## 同梱内容
- `*-image.tar.gz`: Docker コンテナイメージアーカイブ。
- `models/default/`: 推論用モデル成果物 (ONNX グラフ, トークナイザー, 較正設定)。
- `docker-compose.yml`: 本番運用向け Compose 定義。
- `deploy.sh`: ワンクリック導入スクリプト。

## デプロイ手順

### 1. イメージの読み込み
```bash
docker load -i local-jev-*-image.tar.gz
```

### 2. サーバーの起動
付属の `deploy.sh` を実行するか、以下のコマンドを実行します。
```bash
docker compose up -d
```

### 3. 動作確認
推論エンジンの準備完了状態 (Readiness) を確認します。
```bash
curl -i http://localhost:3000/ready
```

### 4. 停止方法
```bash
docker compose down
```
EOF

# 8. チェックサム (SHA256) の算出
echo "[INFO] チェックサムを生成中..."
(cd "${PACKAGE_DIR}" && find . -maxdepth 1 -type f -exec shasum -a 256 {} + > SHA256SUMS)

echo "=================================================================="
echo "[SUCCESS] 配布パッケージが正常に生成されました: ${PACKAGE_DIR}."
echo "=================================================================="
