# sokuto API 共通仕様

本ドキュメントでは、HTTP API の基本仕様、通信プロトコル、エンドポイント一覧、エラーハンドリング規約、および OpenAPI 定義の取得方法について解説します。

---

## 1. 基本通信仕様

| 項目                     | 仕様                                                                |
| :----------------------- | :------------------------------------------------------------------ |
| **ベース URL**           | `http://<host>:<port>`(デフォルト: `http://localhost:3000`)         |
| **通信プロトコル**       | HTTP/1.1                                                            |
| **データフォーマット**   | JSON (`Content-Type: application/json; charset=utf-8`)              |
| **文字エンコーディング** | UTF-8                                                               |
| **認証方式**             | 認証なし(プライベート VPC またはリバースプロキシ配下での運用を前提) |
| **CORS**                 | 全オリジン許可(Permissive)                                          |
| **リクエスト上限**       | 最大 10MB(`DefaultBodyLimit`)                                       |

---

## 2. Jev 互換性とインターフェース特性

`sokuto` の API 設計は、TypeSafe AI の Jev (`v1.13.0`) クラウド API (`POST /v1/systemone`) と完全なスキーマ互換性を保持しています。

一般的なチャット補完 API とは異なり、文章生成(テキストトークンの逐次出力)を行わず、
単一のフォワードパスで型付き決定を導出するため、
すべての推論レスポンスにおいて **`completion_tokens` は常に 0** となります。

---

## 3. エンドポイント一覧

サーバーを起動している場合、
ブラウザから `http://localhost:3000/swagger-ui/` にアクセスすることで、
Web UI 上で全エンドポイントの仕様確認やテストリクエストの送信が可能です。

| メソッド | パス                     | 機能分類   | 概要                                                        |
| :------- | :----------------------- | :--------- | :---------------------------------------------------------- |
| `POST`   | `/v1/systemone`          | 推論       | Jev 互換の非自己回帰型判断推論を実行する。                  |
| `GET`    | `/health`                | 運用監視   | サーバープロセスの死活監視(Liveness Probe)。                |
| `GET`    | `/ready`                 | 運用監視   | モデルロード・推論準備状態の確認(Readiness Probe)。         |
| `GET`    | `/metrics`               | 運用監視   | Prometheus 形式のパフォーマンス・運用メトリクスを出力する。 |
| `GET`    | `/swagger-ui`            | 開発ツール | インタラクティブな Swagger UI ドキュメントを表示する。      |
| `GET`    | `/api-docs/openapi.json` | 開発ツール | OpenAPI 3.1 仕様書を JSON 形式で出力する。                  |

---

## 4. HTTP ステータスコード体系

`sokuto` は、RFC 準拠の標準的な HTTP ステータスコードを使用して結果を返却します。

| ステータスコード            | 意味                 | 主な発生要因                                                                            |
| :-------------------------- | :------------------- | :-------------------------------------------------------------------------------------- |
| `200 OK`                    | 成功                 | 推論または状態確認が正常に完了した。                                                    |
| `400 Bad Request`           | リクエスト構文エラー | JSON 構文不正、必須フィールドの欠落、Criteria の形式不一致、段階数の範囲外(2〜10)。     |
| `413 Payload Too Large`     | ペイロード超過       | 質問数が許容上限(デフォルト 128 件)を超過した、またはリクエストサイズが 10MB を超えた。 |
| `500 Internal Server Error` | サーバー内部エラー   | ONNX Runtime の推論実行エラー、トークナイズ処理の異常終了。                             |
| `503 Service Unavailable`   | サービス利用不可     | モデルロード未完了(`/ready`)、またはメトリクスレコーダー未初期化。                      |

---

## 5. エラーレスポンス構造

エラー発生時は、HTTP ステータスコードとともに、以下の Jev 互換 JSON ペイロードが返却されます。

### エラー JSON スキーマ

```json
{
  "error": {
    "message": "エラー内容の人間向け説明文",
    "type": "invalid_request_error または internal_server_error",
    "code": "固有のエラー識別コード"
  }
}
```

### エラーコード一覧

| エラーコード (`code`)       | `type`                  | 説明                                                                       |
| :-------------------------- | :---------------------- | :------------------------------------------------------------------------- |
| `bad_request`               | `invalid_request_error` | リクエスト JSON のパースに失敗した、または型が不正。                       |
| `empty_questions`           | `invalid_request_error` | `questions` オブジェクトが空(1 件も質問が含まれていない)。                 |
| `missing_criteria`          | `invalid_request_error` | Choice 型または Score 型で必須の `criteria` フィールドが指定されていない。 |
| `invalid_criteria_type`     | `invalid_request_error` | Choice 型に配列を指定した、または Score 型にオブジェクトを指定した。       |
| `invalid_choice_count`      | `invalid_request_error` | Choice 型の候補数が 0 または 255 を超過している。                          |
| `invalid_score_level_count` | `invalid_request_error` | Score 型の評価段階数が 2 未満または 10 を超過している。                    |
| `payload_too_large`         | `invalid_request_error` | 質問数または State 文字列長が安全制限を超過している。                      |
| `inference_engine_error`    | `internal_server_error` | ONNX Runtime の推論実行中にエラーが発生。                                  |
| `internal_server_error`     | `internal_server_error` | サーバー内部の不整合・スレッド異常が発生。                                 |

## 6. OpenAPI 仕様書と Swagger UI

`sokuto` は、`utoipa` クレートを通じて OpenAPI 3.1 仕様書をバイナリコンパイル時に自動生成します。

### ブラウザでの閲覧

サーバー起動後、ブラウザで以下の URL にアクセスすると Swagger UI を利用できます。

```text
http://localhost:3000/swagger-ui
```

### OpenAPI JSON の取得

直接 OpenAPI 3.1 仕様 JSON を取得する場合は以下のエンドポイントにアクセスします。

```bash
curl -s http://localhost:3000/api-docs/openapi.json | jq .
```

### CLI による静的ファイル出力

サーバーを起動せずに CI/CD パイプライン等でスキーマ JSON を生成する場合は、`sokuto export-openapi` サブコマンドを使用します。

```bash
# 標準出力へ出力
sokuto export-openapi

# ファイルへ保存
sokuto export-openapi -o openapi.json
```

---

## 7. クイック導通確認 (cURL)

サーバーが正常に起動しているか確認するための最小限の cURL コマンドです。

```bash
# ヘルスチェック
curl -i http://localhost:3000/health

# レディネスチェック
curl -i http://localhost:3000/ready
```
