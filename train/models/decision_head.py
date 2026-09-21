"""動的スキーマ対応デシジョンヘッドおよび Jev 統合モデルモジュール。

バックボーンエンコーダの最終隠れ状態から、各候補マーカー `[OP]` の隠れベクトルを
Gather 抽出層で抽出し、2層 MLP 射影ヘッドによって候補ごとのスカラーロジットを算出する。
"""

import logging
from typing import cast

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from contract import (
    TENSOR_ATTENTION_MASK,
    TENSOR_INPUT_IDS,
    TENSOR_LOGITS,
    TENSOR_OP_INDICES,
)

logger = logging.getLogger(__name__)


class OptionGatherLayer(nn.Module):
    """`[OP]` トークン位置の隠れベクトルを抽出する Gather 層。

    バックボーンが出力する全トークンの隠れ状態系列から、
    候補位置インデックスに対応するスライスを並列に収集する。
    """

    def forward(self, hidden_states: Tensor, op_indices: Tensor) -> Tensor:
        """隠れ状態テンソルから op_indices で指定された位置のベクトルを抽出する。

        内部ロジック:
        1. 隠れ層の次元数 `hidden_size` を取得する。
        2. `op_indices` を `[batch_size, num_options, hidden_size]` に拡張する。
        3. `torch.gather` を系列長次元 (`dim=1`) に適用してマーカーベクトルを抽出する。

        Args:
            hidden_states (Tensor): バックボーン最終層の隠れ状態 `[batch_size, seq_len, hidden_size]`。
            op_indices (Tensor): 各候補の `[OP]` トークン位置インデックス `[batch_size, num_options]`。

        Returns:
            Tensor: 候補マーカー隠れベクトル `[batch_size, num_options, hidden_size]`。
        """
        hidden_size = hidden_states.shape[-1]
        expanded_indices = op_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
        return torch.gather(hidden_states, dim=1, index=expanded_indices)


class DecisionHead(nn.Module):
    """2層 MLP による候補ロジット射影ヘッド。

    数理仕様:
    $$z_i = W_2 \\cdot \\text{GeLU}(W_1 h_i + b_1) + b_2$$
    """

    def __init__(self, hidden_size: int, mlp_hidden_size: int | None = None) -> None:
        """デシジョンヘッドを初期化する。

        Args:
            hidden_size (int): バックボーンの隠れ層次元数。
            mlp_hidden_size (int | None): MLP 中間層の次元数(未指定時は `hidden_size` と同一)。
        """
        super().__init__()
        mid_size = mlp_hidden_size if mlp_hidden_size is not None else hidden_size
        self.dense = nn.Linear(hidden_size, mid_size)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(mid_size, 1)

    def forward(self, option_vectors: Tensor) -> Tensor:
        """候補マーカーベクトルから候補ごとのスカラーロジットを算出する。

        内部ロジック:
        1. 全結合層と GeLU 活性化関数により `[batch_size, num_options, mid_size]` に射影する。
        2. 出力射影層により `[batch_size, num_options, 1]` に変換し、最終次元を削除する。

        Args:
            option_vectors (Tensor): 候補マーカーベクトル `[batch_size, num_options, hidden_size]`。

        Returns:
            Tensor: 候補ロジットテンソル `[batch_size, num_options]`。
        """
        hidden = self.act(self.dense(option_vectors))
        return self.out_proj(hidden).squeeze(-1)


class JevDecisionModel(nn.Module):
    """Jev アーキテクチャに準拠した非自己回帰型決定モデル。

    入力トークン列を受け取り、単一フォワードパスで各候補の決定ロジットを出力する。
    ONNX エクスポートおよび推論ランタイムとの契約インターフェースに準拠する。
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        mlp_hidden_size: int | None = None,
    ) -> None:
        """Jev 決定モデルを初期化する。

        Args:
            backbone (PreTrainedModel): 事前学習済みエンコーダバックボーン(ModernBERT / mmBERT 等)。
            mlp_hidden_size (int | None): デシジョンヘッドの MLP 中間次元数。
        """
        super().__init__()
        self.backbone = backbone
        hidden_size = cast(int, backbone.config.hidden_size)

        self.gather_layer = OptionGatherLayer()
        self.decision_head = DecisionHead(
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        op_indices: Tensor,
    ) -> Tensor:
        """単一フォワードパスで決定ロジットを算出する。

        内部ロジック:
        1. バックボーンエンコーダで全系列の隠れ状態 `last_hidden_state` を取得する。
        2. `OptionGatherLayer` で `op_indices` に対応する各候補マーカーの隠れベクトルを抽出する。
        3. `DecisionHead` で候補ごとのスカラーロジット `[batch_size, num_options]` を算出する。

        Args:
            input_ids (Tensor): トークンID列 `[batch_size, seq_len]`。
            attention_mask (Tensor): アテンションマスク `[batch_size, seq_len]`。
            op_indices (Tensor): 候補マーカー位置インデックス `[batch_size, num_options]`。

        Returns:
            Tensor: 各候補のロジット `[batch_size, num_options]`。
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state: Tensor = outputs.last_hidden_state
        option_vectors = self.gather_layer(last_hidden_state, op_indices)
        return self.decision_head(option_vectors)

    @classmethod
    def get_input_names(cls) -> list[str]:
        """ONNX エクスポート時の入力テンソル名リストを取得する。

        Returns:
            list[str]: 入力テンソル名のリスト。
        """
        return [TENSOR_INPUT_IDS, TENSOR_ATTENTION_MASK, TENSOR_OP_INDICES]

    @classmethod
    def get_output_names(cls) -> list[str]:
        """ONNX エクスポート時の出力テンソル名リストを取得する。

        Returns:
            list[str]: 出力テンソル名のリスト。
        """
        return [TENSOR_LOGITS]

    @classmethod
    def get_dynamic_axes(cls) -> dict[str, dict[int, str]]:
        """ONNX エクスポート時の動的軸定義辞書を取得する。

        Returns:
            dict[str, dict[int, str]]: 動的軸の辞書。
        """
        return {
            TENSOR_INPUT_IDS: {0: "batch_size", 1: "sequence_length"},
            TENSOR_ATTENTION_MASK: {0: "batch_size", 1: "sequence_length"},
            TENSOR_OP_INDICES: {0: "batch_size", 1: "num_options"},
            TENSOR_LOGITS: {0: "batch_size", 1: "num_options"},
        }
