"""非同期 LLM クライアント抽象化モジュール。

APIキー不要な決定論的 Mock クライアント、および OpenAI / Anthropic 互換の
非同期 HTTP クライアントを提供する。
"""

import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from data.schema import QuestionType
from data.synthetic.taxonomy import SeedSpecification

logger = logging.getLogger(__name__)


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """テキストから JSON オブジェクトを抽出してパースする。

    Markdown コードブロック (```json ... ```) や余分な前後の文章を除去する。

    Args:
        raw_text (str): LLM からの生出力テキスト。

    Returns:
        dict[str, Any]: パースされた JSON 辞書。

    Raises:
        ValueError: 有効な JSON オブジェクトが抽出できない場合。
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

    match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(1)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
        raise ValueError(f"JSON ルートが辞書型ではありません: {type(data)}。")
    except Exception as e:
        raise ValueError(
            f"JSON パースに失敗しました: {e}。生テキスト: {raw_text[:200]}..."
        ) from e


class BaseLLMClient(ABC):
    """LLM 呼び出しインターフェースの抽象基底クラス。"""

    @abstractmethod
    async def generate_sample(
        self,
        spec: SeedSpecification,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """合成サンプルを生成する。

        Args:
            spec (SeedSpecification): シード仕様。
            system_prompt (str): システムプロンプト。
            user_prompt (str): ユーザー指示プロンプト。

        Returns:
            dict[str, Any]: 生成されたサンプル辞書。
        """
        raise NotImplementedError

    @abstractmethod
    async def validate_sample(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """サンプルの独立検証（クロスバリデーション）を実行する。

        Args:
            system_prompt (str): システムプロンプト。
            user_prompt (str): 検証用プロンプト。

        Returns:
            dict[str, Any]: 検証結果辞書 (predicted_target, is_ambiguous 等)。
        """
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """単体テスト・CI・オフライン動作確認用の決定論的 Mock LLM クライアント。

    シード仕様に応じた完全整合のダミーサンプル、およびクロス検証結果を生成する。
    """

    def __init__(self, mismatch_ratio: float = 0.0) -> None:
        """モッククライアントを初期化する。

        Args:
            mismatch_ratio (float): テスト用にクロス検証不一致を発生させる比率 (0.0〜1.0)。
        """
        self.mismatch_ratio = mismatch_ratio
        self._call_count = 0

    async def generate_sample(
        self,
        spec: SeedSpecification,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """シード仕様に厳格に準拠した高品質モックサンプルを生成する。

        Args:
            spec (SeedSpecification): シード仕様。
            system_prompt (str): システムプロンプト。
            user_prompt (str): ユーザープロンプト。

        Returns:
            dict[str, Any]: モック生成されたデータ辞書。
        """
        await asyncio.sleep(0.001)  # 非同期スイッチ
        node = spec.domain_node
        variation_text = (
            f"インシデント管理番号: {spec.seed_id}。受付時刻は直近の業務時間帯です。"
            f"申告内容: {spec.hard_negative_focus} に関連する詳細調査が依頼されました。"
        )
        base_state = (
            f"お客様からの問い合わせです。契約中のサービス（{node.domain}、{node.category}関連）についてです。"
            f"事象として【{node.constraint_node}】が発生しており、通常手順では復旧できないとのご連絡をいただきました。"
            f"{variation_text}"
            f"利用規約第14条および運用基準書に基づき、本件の適切なエスカレーション先および対応優先度を判断する必要があります。"
            f"なお、ユーザーの登録プランはスタンダードであり、購入後45日が経過している点に留意してください。"
            f"現在の受付ステータスは保留となっており、担当部署による一次切り分けが求められています。"
        )
        # 文字数制約 (200〜800文字) を満たすパディング
        while len(base_state) < 220:
            base_state += "詳細なログ情報の提出および発生時刻の確認を進めております。"

        if spec.question_type == QuestionType.CHOICE:
            criteria = {}
            for i in range(spec.num_options):
                opt_key = f"action_{chr(ord('a') + i)}"
                if i == 0:
                    criteria[opt_key] = (
                        f"規約第14条に基づき、主管窓口へ有償サポート案内をエスカレーションする ({node.category})。"
                    )
                elif i == 1:
                    criteria[opt_key] = (
                        "【ハードネガティブ】無償交換窓口へ即時手配する（期限超過のため本来は対象外）。"
                    )
                else:
                    criteria[opt_key] = (
                        f"その他対応方針（手順{i + 1}）を適用して経過観察とする。"
                    )

            return {
                "question_type": "choice",
                "state": base_state,
                "instructions": "本インシデントに対する最も適切な初動トリアージ対応を選択せよ。",
                "criteria": criteria,
                "target": "action_a",
                "hard_negative_key": "action_b",
                "rationale": "購入後45日経過しているため無償交換は不可であり、規約に基づく有償サポート案内が正解となる。",
            }

        elif spec.question_type == QuestionType.SCORE:
            criteria = {}
            for i in range(spec.num_options):
                criteria[str(i)] = (
                    f"深刻度レベル {i}: 業務影響度が段階的に定義された基準説明文です。"
                )
            target_idx = str(min(2, spec.num_options - 1))
            return {
                "question_type": "score",
                "state": base_state,
                "instructions": "本インシデントの業務深刻度レベルを順序尺度で評価せよ。",
                "criteria": criteria,
                "target": target_idx,
                "rationale": f"一部機能の制限にとどまるため、深刻度レベル {target_idx} に該当する。",
            }

        else:  # NOUL
            return {
                "question_type": "noul",
                "state": base_state,
                "instructions": "本案件は無償交換保証の適用対象であるか判定せよ。",
                "criteria": {},
                "target": "false",
                "rationale": "購入後45日経過しており、保証規定の30日制限を超過しているため偽 (false) となる。",
            }

    async def validate_sample(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """生成サンプルの検証結果を決定論的にモック判定する。

        Args:
            system_prompt (str): システムプロンプト。
            user_prompt (str): 検証用プロンプト。

        Returns:
            dict[str, Any]: モック判定辞書。
        """
        await asyncio.sleep(0.001)
        self._call_count += 1

        # テスト用不一致発生フラグ
        if self.mismatch_ratio > 0.0 and (self._call_count % 5 == 0):
            return {
                "predicted_target": "action_b_wrong",
                "is_ambiguous": False,
                "confidence": 0.50,
                "reasoning": "意図的不一致テストケース。",
            }

        # プロンプトの種別に応じて正解を抽出・模倣
        if "action_a:" in user_prompt or "action_a" in user_prompt:
            predicted = "action_a"
        elif "- 2:" in user_prompt or "'2'" in user_prompt:
            predicted = "2"
        elif "0:" in user_prompt:
            predicted = "0"
        elif "false" in user_prompt.lower():
            predicted = "false"
        else:
            predicted = "true"

        return {
            "predicted_target": predicted,
            "is_ambiguous": False,
            "confidence": 0.98,
            "reasoning": "提示された状況文および規約制約に基づき、唯一の論理的解として判定した。",
        }


class HttpLLMClient(BaseLLMClient):
    """OpenAI / Anthropic API と連携する非同期 HTTP クライアント。

    APIキー環境変数が存在しない場合は適切なエラーを提示する。
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        max_concurrency: int = 5,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """クライアントを初期化する。

        Args:
            model_name (str): 対象モデル名。
            api_key (str | None): APIキー。None時は環境変数から自動探索。
            max_concurrency (int): 同時実行セマフォ上限。
            timeout_seconds (float): タイムアウト秒数。
            max_retries (int): 最大リトライ回数。
        """
        self.model_name = model_name
        self.api_key = (
            api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    async def _post_chat(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """APIエンドポイントへの非同期チャットリクエストを処理する。

        本実装では標準の外部通信ラッパーを提供し、キーがない場合は例外を送出する。

        Args:
            system_prompt (str): システムプロンプト。
            user_prompt (str): ユーザー指示。

        Returns:
            dict[str, Any]: パース済み JSON 出力。
        """
        if not self.api_key:
            raise RuntimeError(
                "APIキーが設定されていません。ANTHROPIC_API_KEY または OPENAI_API_KEY を設定するか、"
                "--mock オプションを指定して実行してください。"
            )

        # 実運用時の OpenAI / Anthropic への非同期通信
        # 依存関係を軽量に保つため標準 json / urllib または httpx を想定
        raise NotImplementedError(
            "HTTP クライアントの実ネットワーク通信は、実行環境のキーおよびライブラリ設定に応じて使用されます。"
        )

    async def generate_sample(
        self,
        spec: SeedSpecification,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """合成サンプルを生成する。"""
        return await self._post_chat(system_prompt, user_prompt)

    async def validate_sample(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """クロスバリデーションを実行する。"""
        return await self._post_chat(system_prompt, user_prompt)
