"""反事実・前提可変対照テスト (Contrast Sets) 生成プロンプトレシピおよびデータ定義モジュール。

小型エンコーダにおける前提可変・複合ルールの推論粗さ (Jaggedness) や、
表層キーワード (「プラチナ」「無料」等) への過剰反応を検証するための
最小編集対照ペア (Minimal Edit Pairs) の生成レシピ、スキーマ、および検証器を提供する。
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# doc07.md に基づく Frontier LLM 向け Contrast Sets 生成プロンプトレシピ
CONTRAST_SET_PROMPT_TEMPLATE = """あなたはエンタープライズ業務システムの QA アーキテクトです。

以下の業務規約に基づき、マイクロ決定 DAG の耐性を検証するための「対照テストセット (Contrast Sets)」を生成してください。

【対象規約】
{policy_description}

【生成要件】
1. 各組は「基本文 (base_state)」と「反事実文 (cf_state)」のペアで構成すること。
2. 反事実文は、基本文から単一の条件・エンティティ (例: 会員ランク、送金先、経過日数、セール適用等) のみを最小限編集 (Minimal Edit) して判定結果を反転させること。
3. 表層のキーワード (「無料」「プラチナ」「未開封」等) に過剰反応するモデルを落とすハードディストラクターを含めること。
4. 単なる語尾の否定 (〜ではない) に頼らず、肯定文のまま数値境界や固有名詞・属性を変更したペアを過半数含めること。
5. 出力は以下の JSON 形式とすること。
[
  {{
    "id": "pair_01",
    "domain": "{domain}",
    "base_state": "...",
    "expected_base": "...",
    "cf_state": "...",
    "expected_cf": "...",
    "flipped_factor": "...",
    "monolithic_prompt": "...",
    "notes": "..."
  }}
]
"""


@dataclass
class ContrastPair:
    """最小編集対照ペア (Contrast Pair) データモデル。

    Attributes:
        id (str): テストケースの一意識別子。
        domain (str): ドメイン識別子 (banking_fee, ec_return, security_triage 等)。
        base_state (str): 基本となる入力発話文。
        expected_base (str): 基本文に対する期待される決定値 (例: 'eligible', 'free', 'approved')。
        cf_state (str): 条件を 1 要素のみ反転させた反事実入力文。
        expected_cf (str): 反事実文に対する期待される決定値 (例: 'ineligible', 'fee', 'denied')。
        flipped_factor (str): 反転させた条件の識別子 (例: 'domestic_vs_overseas', 'days_exceeded')。
        monolithic_prompt (str): 単一プロンプト評価時に使用する規約文を含むプロンプト。
        notes (str): テストケースの意図や検証ポイントに関する補足説明。
    """

    id: str
    domain: str
    base_state: str
    expected_base: str
    cf_state: str
    expected_cf: str
    flipped_factor: str
    monolithic_prompt: str = ""
    notes: str = ""

    def validate(self) -> None:
        """対照ペアの健全性を検証する。

        Raises:
            ValueError: 必須項目が欠落しているか、Base と CF が同一、または決定が同一の場合。
        """
        if not self.id.strip():
            raise ValueError("対照ペアの id が空です。")
        if not self.base_state.strip() or not self.cf_state.strip():
            raise ValueError(f"対照ペア `{self.id}` の state が空です。")
        if self.base_state.strip() == self.cf_state.strip():
            raise ValueError(
                f"対照ペア `{self.id}` の base_state と cf_state が完全一致しています (反転なし)。"
            )
        if self.expected_base == self.expected_cf:
            raise ValueError(
                f"対照ペア `{self.id}` の expected_base と expected_cf が同一です (反転なし)。"
            )
        if not self.flipped_factor.strip():
            raise ValueError(f"対照ペア `{self.id}` の flipped_factor が空です。")

    def to_dict(self) -> dict[str, Any]:
        """辞書表現に変換する。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContrastPair":
        """辞書からインスタンスを生成する。"""
        return cls(
            id=data["id"],
            domain=data.get("domain", "general"),
            base_state=data["base_state"],
            expected_base=str(data["expected_base"]),
            cf_state=data["cf_state"],
            expected_cf=str(data["expected_cf"]),
            flipped_factor=data["flipped_factor"],
            monolithic_prompt=data.get("monolithic_prompt", ""),
            notes=data.get("notes", ""),
        )


@dataclass
class ContrastSetSuite:
    """対照ペアデータセットスイート。

    Attributes:
        domain (str): ドメイン名。
        policy_description (str): 業務規約の全文説明。
        pairs (list[ContrastPair]): 対照ペアのリスト。
        dag_definition (dict[str, Any]): 当該ドメインを解くためのマイクロ決定 DAG 定義。
    """

    domain: str
    policy_description: str
    pairs: list[ContrastPair] = field(default_factory=list)
    dag_definition: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """スイート全体の整合性を検証する。"""
        if not self.pairs:
            raise ValueError(
                f"ドメイン `{self.domain}` の対照ペアが 1 件もありません。"
            )
        seen_ids: set[str] = set()
        for pair in self.pairs:
            pair.validate()
            if pair.id in seen_ids:
                raise ValueError(f"対照ペア ID `{pair.id}` が重複しています。")
            seen_ids.add(pair.id)

    def save_json(self, path: str | Path) -> None:
        """JSON ファイルとして保存する。"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "domain": self.domain,
            "policy_description": self.policy_description,
            "dag_definition": self.dag_definition,
            "pairs": [p.to_dict() for p in self.pairs],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "ContrastSetSuite":
        """JSON ファイルからロードする。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        suite = cls(
            domain=data["domain"],
            policy_description=data.get("policy_description", ""),
            dag_definition=data.get("dag_definition", {}),
            pairs=[ContrastPair.from_dict(p) for p in data.get("pairs", [])],
        )
        suite.validate()
        return suite
