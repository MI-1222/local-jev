"""Jev 学習・評価パイプライン用統一中間スキーマモジュール。

各 NLP タスクを統一表現形式 (State + Instructions + Criteria) へ標準化し、
後続の候補シャッフル、負例混入、トークナイズ処理との受け渡しを担うデータ構造を定義する。
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class QuestionType(str, Enum):
    """決定プリミティブ種別列挙型。

    Choice: 離散選択肢から最適な1つを選択。
    Score: 順序尺度評価。
    Noul: 真偽判定。
    """

    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass(frozen=True)
class UnifiedSample:
    """Jev モデル学習・評価用の統一中間データ構造。

    Attributes:
        dataset_name (str): データセット識別名 (例: 'banking77', 'mnli')。
        sample_id (str): サンプル固有識別子。
        question_type (QuestionType): 決定プリミティブ種別。
        state (str): 入力非構造化文脈テキスト。
        instructions (str): タスク指示文。
        criteria (dict[str, str]): 候補識別子をキー、自然言語説明文を値とする辞書。
        target (str): 正解識別子 (Choice 時は criteria のキー、Noul 時は 'true' または 'false')。
        metadata (dict[str, Any]): 元スプリットや追加情報を含むメタデータ辞書。
    """

    dataset_name: str
    sample_id: str
    question_type: QuestionType
    state: str
    instructions: str
    criteria: dict[str, str]
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """データ整合性を検証する。

        Raises:
            ValueError: Choice 型で target が criteria のキーに含まれない場合、
                または Noul 型で target が 'true' / 'false' 以外の場合。
        """
        if self.question_type == QuestionType.CHOICE:
            if self.criteria and self.target not in self.criteria:
                raise ValueError(
                    f"Choice 型の target '{self.target}' が criteria のキーに含まれていません: {list(self.criteria.keys())}。"
                )
        elif self.question_type == QuestionType.NOUL:
            valid_targets = {"true", "false"}
            if self.target.lower() not in valid_targets:
                raise ValueError(
                    f"Noul 型の target は 'true' または 'false' である必要があります: '{self.target}'。"
                )

    def to_dict(self) -> dict[str, Any]:
        """辞書オブジェクトへ変換する。

        Returns:
            dict[str, Any]: シリアライズ可能な辞書表現。
        """
        data = asdict(self)
        data["question_type"] = self.question_type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedSample":
        """辞書オブジェクトから UnifiedSample を復元する。

        Args:
            data (dict[str, Any]): 辞書データ。

        Returns:
            UnifiedSample: 復元された統一サンプルインスタンス。
        """
        raw_type = data["question_type"]
        question_type = (
            raw_type if isinstance(raw_type, QuestionType) else QuestionType(raw_type)
        )
        return cls(
            dataset_name=data["dataset_name"],
            sample_id=data["sample_id"],
            question_type=question_type,
            state=data["state"],
            instructions=data["instructions"],
            criteria=dict(data.get("criteria", {})),
            target=str(data["target"]),
            metadata=dict(data.get("metadata", {})),
        )
