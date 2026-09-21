"""位置バイアス排除対応の PyTorch Dataset および Collate モジュール。

エポック連動型オンザフライシャッフルによって、Choice 型候補の物理的位置バイアス
(Primacy / Recency Bias) を根絶し、Score 型の順序尺度および Noul 型の契約を保護する。
"""

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from contract import MAX_SEQUENCE_LENGTH
from data.formatter import tokenize_sample
from data.schema import QuestionType, UnifiedSample


class JevDataset(Dataset[dict[str, Tensor]]):
    """エポックごとに候補順序を動的シャッフルする Jev 学習・評価用 Dataset。

    Attributes:
        samples (list[UnifiedSample]): 統一サンプルのリスト。
        tokenizer (PreTrainedTokenizerFast): トークナイザーインスタンス。
        op_token_id (int): 特殊トークン [OP] の ID。
        max_length (int): 最大系列長。
        is_train (bool): 学習モードフラグ。False の場合はシャッフルを無効化する。
        base_seed (int): 決定論的シード計算用のベースシード値。
        current_epoch (int): 現在のエポック番号。
    """

    def __init__(
        self,
        samples: list[UnifiedSample],
        tokenizer: PreTrainedTokenizerFast,
        op_token_id: int,
        max_length: int = MAX_SEQUENCE_LENGTH,
        is_train: bool = True,
        base_seed: int = 42,
    ) -> None:
        """データセットを初期化する。

        Args:
            samples (list[UnifiedSample]): 統一中間サンプルのリスト。
            tokenizer (PreTrainedTokenizerFast): トークナイザー。
            op_token_id (int): [OP] 特殊トークンの ID。
            max_length (int): トークナイズ時の最大許容系列長。
            is_train (bool): 学習モードフラグ(デフォルト: True)。
            base_seed (int): シード生成用の基準シード値(デフォルト: 42)。
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.op_token_id = op_token_id
        self.max_length = max_length
        self.is_train = is_train
        self.base_seed = base_seed
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """エポック番号を更新し、次エポックの動的シャッフルシードを切り替える。

        Args:
            epoch (int): 新しいエポック番号。
        """
        self.current_epoch = epoch

    def __len__(self) -> int:
        """データセット内のサンプル総数を取得する。

        Returns:
            int: サンプル件数。
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """指定インデックスのサンプルをオンザフライでトークナイズして返却する。

        Choice 型かつ学習時のみ候補をシャッフルし、Score 型は順序尺度を保護するため
        絶対にシャッフルしない。マルチプロセスワーカー間でも決定論的な再現性を保つため、
        エポック番号・サンプルインデックス・ベースシードからシードを動的に算出する。

        Args:
            index (int): サンプルインデックス。

        Returns:
            dict[str, Tensor]: トークナイズ済みテンソル辞書。
        """
        sample = self.samples[index]

        should_shuffle = self.is_train and (sample.question_type == QuestionType.CHOICE)

        seed = (
            (self.base_seed + self.current_epoch * 1_000_003 + index) & 0xFFFFFFFF
            if should_shuffle
            else None
        )

        return tokenize_sample(
            sample=sample,
            tokenizer=self.tokenizer,
            op_token_id=self.op_token_id,
            max_length=self.max_length,
            shuffle_options=should_shuffle,
            seed=seed,
        )


JevDynamicDataset = JevDataset


def jev_collate_fn(
    batch: list[dict[str, Tensor]],
    pad_token_id: int,
) -> dict[str, Tensor]:
    """可変長系列および可変長候補数をパディングして単一バッチテンソルに結合する。

    系列長はバッチ内の最大長 `max_seq_len`、候補数はバッチ内の最大数 `max_options` に
    揃えてパディングする。未定義候補位置の誤評価を防ぐため、`op_mask` も生成する。

    Args:
        batch (list[dict[str, Tensor]]): 単一サンプルのテンソル辞書リスト。
        pad_token_id (int): パディングトークン ID。

    Returns:
        dict[str, Tensor]:
            - input_ids: [batch_size, max_seq_len] のパディング済みトークン列。
            - attention_mask: [batch_size, max_seq_len] のパディングマスク。
            - op_indices: [batch_size, max_options] の [OP] マーカーインデックス。
            - op_mask: [batch_size, max_options] の有効候補マスク(有効: True, パディング: False)。
            - labels: [batch_size] の正解ラベルテンソル。
    """
    batch_size = len(batch)
    max_seq_len = max(item["input_ids"].size(0) for item in batch)
    max_options = max(item["op_indices"].size(0) for item in batch)

    padded_input_ids = torch.full(
        (batch_size, max_seq_len),
        pad_token_id,
        dtype=torch.long,
    )
    padded_attention_mask = torch.zeros(
        (batch_size, max_seq_len),
        dtype=torch.long,
    )
    padded_op_indices = torch.zeros(
        (batch_size, max_options),
        dtype=torch.long,
    )
    op_mask = torch.zeros(
        (batch_size, max_options),
        dtype=torch.bool,
    )
    labels = torch.tensor([item["label"].item() for item in batch], dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        num_ops = item["op_indices"].size(0)

        padded_input_ids[i, :seq_len] = item["input_ids"]
        padded_attention_mask[i, :seq_len] = item["attention_mask"]
        padded_op_indices[i, :num_ops] = item["op_indices"]
        op_mask[i, :num_ops] = True

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "op_indices": padded_op_indices,
        "op_mask": op_mask,
        "labels": labels,
    }


pad_jev_collate_fn = jev_collate_fn
