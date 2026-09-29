# 運用監視と Prometheus メトリクス仕様

`sokuto` は、本番コンテナ環境やマイクロサービス基盤における安定運用のために、
ヘルスチェック(Liveness)、準備状態監視(Readiness)、および Prometheus 形式のパフォーマンスメトリクス出力を標準提供しています。

本ドキュメントでは、各運用監視エンドポイントの仕様、メトリクス項目の一覧、および最小限のインフラ設定例を解説します。

---

## 1. 運用監視エンドポイント一覧

| エンドポイント | メソッド | 用途                                            | 成功時ステータス |
| :------------- | :------: | :---------------------------------------------- | :--------------: |
| `/health`      |  `GET`   | プロセスのヘルスチェック(Liveness Probe)        |     `200 OK`     |
| `/ready`       |  `GET`   | 推論エンジンの準備完了状態監視(Readiness Probe) |     `200 OK`     |
| `/metrics`     |  `GET`   | Prometheus 形式のメトリクスエクスポート         |     `200 OK`     |

---

## 2. ヘルスチェック仕様

### 1. Liveness Probe (`GET /health`)

サーバープロセスが生存し、HTTP イベントループがリクエストを受け付け可能かを即座に返却します。

- **リクエスト**: `GET /health`
- **レスポンス**: `200 OK` (`Content-Type: application/json`)

```bash
curl -i http://localhost:3000/health
```

```json
{
  "status": "ok"
}
```

---

### 2. Readiness Probe (`GET /ready`)

ONNX モデルおよびトークナイザーがメモリ上に正常にロードされ、推論リクエストを安全に処理できる状態にあるかを検証します。

- **リクエスト**: `GET /ready`
- **ステータスコード**:
  - `200 OK`: モデル初期化完了。推論トラフィックの受付可能。
  - `503 Service Unavailable`: モデルロード中または初期化失敗。

#### 準備完了時 (`200 OK`)

```bash
curl -i http://localhost:3000/ready
```

```json
{
  "status": "ready"
}
```

#### 未準備時 (`503 Service Unavailable`)

```json
{
  "status": "not_ready"
}
```

---

### 3. CLI によるヘルスチェック確認

最小コンテナ環境など `curl` が存在しない環境向けに、スタンドアロン CLI サブコマンド `sokuto healthcheck` が提供されています。内部で軽量な HTTP/1.1 ソケット通信を行い、ステータスコード `200 OK` を検証します。

#### コマンド構文とオプション

| オプション                      | 短縮 | 環境変数                 | デフォルト値                  | 説明                                       |
| :------------------------------ | :--: | :----------------------- | :---------------------------- | :----------------------------------------- |
| `--url <URL>`                   | `-u` | `SOKUTO_HEALTHCHECK_URL` | `http://127.0.0.1:3000/ready` | 監視対象 URL (Readiness または Liveness)。 |
| `--timeout-secs <TIMEOUT_SECS>` | `-t` | なし                     | `3`                           | 接続および応答タイムアウト秒数。           |

#### 実行例

```bash
# デフォルト (http://127.0.0.1:3000/ready) を確認
sokuto healthcheck

# Liveness Probe を確認
sokuto healthcheck -u http://127.0.0.1:3000/health

# 別ホストおよびタイムアウトを指定
sokuto healthcheck -u http://10.0.0.5:8080/ready -t 5
```

---

## 3. Prometheus メトリクス仕様 (`GET /metrics`)

`sokuto` は、`metrics-exporter-prometheus` を通じて OpenMetrics / Prometheus 標準テキスト形式で統計情報を公開します。

- **リクエスト**: `GET /metrics`
- **レスポンス**: `200 OK` (`Content-Type: text/plain; version=0.0.4; charset=utf-8`)

### メトリクス名一覧

| メトリクス名                        |  種別   | ラベル               | 説明                                                                                |
| :---------------------------------- | :-----: | :------------------- | :---------------------------------------------------------------------------------- |
| `sokuto_requests_total`             | Counter | `endpoint`, `status` | HTTP リクエストの処理完了総数 (`endpoint="/v1/systemone"` で記録)。                 |
| `sokuto_http_duration_seconds`      | Summary | `endpoint`           | HTTP リクエスト全体の受付からレスポンス返却までの所要時間(秒)。                     |
| `sokuto_inference_duration_seconds` | Summary | なし                 | ONNX 推論エンジン単体の正味実行時間(秒)。                                           |
| `sokuto_batch_size`                 | Summary | なし                 | 1 リクエストあたりに処理された質問(Question)の件数。                                |
| `sokuto_question_type_total`        | Counter | `type`               | 質問プリミティブ別(`choice`, `score`, `noul`)の処理回数。                           |
| `sokuto_confidence_score`           | Summary | なし                 | 算出された判定確信度スコア(0.0 〜 1.0)の分布。                                      |
| `sokuto_gating_routes_total`        | Counter | `route`, `type`      | ゲーティング判定ルート(`auto_execute`, `confirm_or_escalate`, `fallback`)別の件数。 |

---

### Prometheus 出力サンプル

```text
# TYPE sokuto_question_type_total counter
sokuto_question_type_total{type="choice"} 1240
sokuto_question_type_total{type="score"} 680
sokuto_question_type_total{type="noul"} 420

# TYPE sokuto_requests_total counter
sokuto_requests_total{endpoint="/v1/systemone",status="200"} 1542

# TYPE sokuto_batch_size summary
sokuto_batch_size{quantile="0"} 1
sokuto_batch_size{quantile="0.5"} 1
sokuto_batch_size{quantile="0.9"} 4
sokuto_batch_size{quantile="0.95"} 8
sokuto_batch_size{quantile="0.99"} 16
sokuto_batch_size{quantile="0.999"} 32
sokuto_batch_size{quantile="1"} 32
sokuto_batch_size_sum 2310
sokuto_batch_size_count 1542

# TYPE sokuto_inference_duration_seconds summary
sokuto_inference_duration_seconds{quantile="0"} 0.012540
sokuto_inference_duration_seconds{quantile="0.5"} 0.018420
sokuto_inference_duration_seconds{quantile="0.9"} 0.024150
sokuto_inference_duration_seconds{quantile="0.95"} 0.028910
sokuto_inference_duration_seconds{quantile="0.99"} 0.035400
sokuto_inference_duration_seconds{quantile="0.999"} 0.048200
sokuto_inference_duration_seconds{quantile="1"} 0.052100
sokuto_inference_duration_seconds_sum 28.403
sokuto_inference_duration_seconds_count 1542

# TYPE sokuto_confidence_score summary
sokuto_confidence_score{quantile="0"} 0.1250
sokuto_confidence_score{quantile="0.5"} 0.8420
sokuto_confidence_score{quantile="0.9"} 0.9650
sokuto_confidence_score{quantile="0.95"} 0.9820
sokuto_confidence_score{quantile="0.99"} 0.9950
sokuto_confidence_score{quantile="0.999"} 0.9990
sokuto_confidence_score{quantile="1"} 1.0000
sokuto_confidence_score_sum 1945.312
sokuto_confidence_score_count 2310

# TYPE sokuto_http_duration_seconds summary
sokuto_http_duration_seconds{endpoint="/v1/systemone",quantile="0"} 0.015210
sokuto_http_duration_seconds{endpoint="/v1/systemone",quantile="0.5"} 0.021850
sokuto_http_duration_seconds{endpoint="/v1/systemone",quantile="0.9"} 0.028520
sokuto_http_duration_seconds{endpoint="/v1/systemone",quantile="0.95"} 0.033400
sokuto_http_duration_seconds{endpoint="/v1/systemone",quantile="0.99"} 0.041800
sokuto_http_duration_seconds{endpoint="/v1/systemone",quantile="0.999"} 0.055100
sokuto_http_duration_seconds{endpoint="/v1/systemone",quantile="1"} 0.061500
sokuto_http_duration_seconds_sum 33.701
sokuto_http_duration_seconds_count 1542

# TYPE sokuto_gating_routes_total counter
sokuto_gating_routes_total{route="auto_execute",type="choice"} 1180
sokuto_gating_routes_total{route="confirm_or_escalate",type="choice"} 45
sokuto_gating_routes_total{route="fallback",type="choice"} 15
```

---

## 4. 最小限のインフラ設定例

### Kubernetes Pod プローブ設定

Kubernetes で運用する際は、以下の通り Liveness と Readiness を設定します。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sokuto-inference
spec:
  containers:
    - name: sokuto
      image: sokuto:latest
      ports:
        - containerPort: 3000
      livenessProbe:
        httpGet:
          path: /health
          port: 3000
        initialDelaySeconds: 2
        periodSeconds: 10
      readinessProbe:
        httpGet:
          path: /ready
          port: 3000
        initialDelaySeconds: 5
        periodSeconds: 5
```

### Docker Compose ヘルスチェック設定

```yaml
services:
  sokuto:
    image: sokuto:latest
    ports:
      - '3000:3000'
    healthcheck:
      test: ['CMD', 'sokuto', 'healthcheck']
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 5s
```
