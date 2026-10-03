"""粗密二段階階層データセットおよびトークナイズモジュール。

単一の細分類サンプルと階層オントロジー (HierarchicalMapping) から、
大分類判定 (Coarse) と細分類判定 (Fine / Oracle または Soft-Beam) の
ペアサンプルを動的に生成し、位置バイアスを排除しつつ親子クラスタの
インデックス整合性を保持したパディングバッチを構築する。
"""

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

from contract import MAX_SEQUENCE_LENGTH
from data.formatter import _build_criteria_text, _get_special_tokens
from data.hierarchical import HierarchicalMapping
from data.schema import QuestionType, UnifiedSample

DEFAULT_COARSE_INSTRUCTIONS = (
    "問い合わせ内容に最も合致する大分類カテゴリを選択してください。"
)
DEFAULT_FINE_INSTRUCTIONS = "問い合わせ内容に最も合致する詳細意図を選択してください。"


@dataclass(frozen=True)
class HierarchicalSamplePair:
    """大分類と細分類のペアサンプル構造体。

    Attributes:
        coarse_sample (UnifiedSample): 大分類用統一サンプル。
        fine_sample (UnifiedSample): 細分類用統一サンプル (Oracle または Soft-Beam)。
        is_soft_beam (bool): Soft-Beam 模倣サンプルかどうか。
        fine_to_coarse_key (dict[str, str]): 細分類候補キーから親大分類キーへの対応辞書。
    """

    coarse_sample: UnifiedSample
    fine_sample: UnifiedSample
    is_soft_beam: bool
    fine_to_coarse_key: dict[str, str]


class HierarchicalDatasetGenerator:
    """細分類サンプル列とオントロジーから階層ペアサンプルを生成するクラス。

    Attributes:
        mapping (HierarchicalMapping): 階層マッピングオントロジー。
        soft_beam_ratio (float): Soft-Beam 模倣サンプルの生成比率 (0.0〜1.0)。
        negative_ratio (float): 受け皿 (その他 / 該当なし) 候補を正解とする負例の混入比率。
        max_fine_candidates (int): Fine プロンプトに含める最大候補数 (アテンション希釈防止)。
        coarse_instructions (str): 大分類用指示文。
        fine_instructions (str): 細分類用指示文。
    """

    def __init__(
        self,
        mapping: HierarchicalMapping,
        soft_beam_ratio: float = 0.30,
        negative_ratio: float = 0.15,
        max_fine_candidates: int = 24,
        coarse_instructions: str = DEFAULT_COARSE_INSTRUCTIONS,
        fine_instructions: str = DEFAULT_FINE_INSTRUCTIONS,
    ) -> None:
        """生成器を初期化する。

        Args:
            mapping (HierarchicalMapping): 階層オントロジー。
            soft_beam_ratio (float): Soft-Beam 模倣比率 (デフォルト: 0.30)。
            negative_ratio (float): 負例混入比率 (デフォルト: 0.15)。
            max_fine_candidates (int): Fine 候補の上限数 (デフォルト: 24)。
            coarse_instructions (str): 大分類プロンプト用指示文。
            fine_instructions (str): 細分類プロンプト用指示文。
        """
        mapping.validate()
        self.mapping = mapping
        self.soft_beam_ratio = soft_beam_ratio
        self.negative_ratio = negative_ratio
        self.max_fine_candidates = max_fine_candidates
        self.coarse_instructions = coarse_instructions
        self.fine_instructions = fine_instructions

        # クラスタ巡回順序キャッシュ
        self.coarse_keys = list(self.mapping.all_coarse_keys)

    def generate_pair(
        self,
        fine_sample: UnifiedSample,
        rng: random.Random | None = None,
    ) -> HierarchicalSamplePair:
        """単一の細分類サンプルから Coarse と Fine のペアを生成する。

        Args:
            fine_sample (UnifiedSample): 細分類正解を持つ統一サンプル。
            rng (random.Random | None): 乱数生成器。未指定時はグローバル random を使用。

        Returns:
            HierarchicalSamplePair: 生成された大分類・細分類ペア。

        Raises:
            ValueError: サンプルの target がオントロジーに存在しない場合。
        """
        local_rng = rng if rng is not None else random.Random()
        target_fine = fine_sample.target

        if target_fine not in self.mapping.fine_to_coarse:
            raise ValueError(
                f"細分類ターゲット `{target_fine}` がオントロジーに存在しません。"
            )

        target_coarse = self.mapping.fine_to_coarse[target_fine]

        # 1. 大分類サンプルの構築 (7 クラスタ)
        coarse_criteria = dict(self.mapping.coarse_categories)
        coarse_sample = UnifiedSample(
            dataset_name=fine_sample.dataset_name,
            sample_id=f"{fine_sample.sample_id}_coarse",
            question_type=QuestionType.CHOICE,
            state=fine_sample.state,
            instructions=self.coarse_instructions,
            criteria=coarse_criteria,
            target=target_coarse,
            metadata={
                **fine_sample.metadata,
                "hierarchical_level": "coarse",
                "original_target": target_fine,
            },
        )

        # 2. 細分類サンプルの候補選定 (Oracle vs Soft-Beam)
        use_soft_beam = local_rng.random() < self.soft_beam_ratio
        fine_candidates: dict[str, str] = {}
        fine_to_coarse_map: dict[str, str] = {}

        # 正解クラスタ配下の全候補を追加
        oracle_keys = self.mapping.coarse_to_fine[target_coarse]
        for k in oracle_keys:
            fine_candidates[k] = self.mapping.fine_criteria[k]
            fine_to_coarse_map[k] = target_coarse

        if use_soft_beam:
            # 近傍クラスタを 1 つ選択して候補をマージ
            other_coarse_keys = [k for k in self.coarse_keys if k != target_coarse]
            if other_coarse_keys:
                neighbor_coarse = local_rng.choice(other_coarse_keys)
                neighbor_keys = self.mapping.coarse_to_fine[neighbor_coarse]
                for k in neighbor_keys:
                    if len(fine_candidates) >= self.max_fine_candidates:
                        break
                    fine_candidates[k] = self.mapping.fine_criteria[k]
                    fine_to_coarse_map[k] = neighbor_coarse

        fine_sample_out = UnifiedSample(
            dataset_name=fine_sample.dataset_name,
            sample_id=f"{fine_sample.sample_id}_fine",
            question_type=QuestionType.CHOICE,
            state=fine_sample.state,
            instructions=self.fine_instructions,
            criteria=fine_candidates,
            target=target_fine,
            metadata={
                **fine_sample.metadata,
                "hierarchical_level": "fine",
                "is_soft_beam": use_soft_beam,
                "parent_coarse": target_coarse,
            },
        )

        return HierarchicalSamplePair(
            coarse_sample=coarse_sample,
            fine_sample=fine_sample_out,
            is_soft_beam=use_soft_beam,
            fine_to_coarse_key=fine_to_coarse_map,
        )

    def generate_pairs(
        self,
        fine_samples: list[UnifiedSample],
        base_seed: int = 42,
    ) -> list[HierarchicalSamplePair]:
        """サンプルリストからペアリストを一括生成する。

        Args:
            fine_samples (list[UnifiedSample]): 入力サンプルリスト。
            base_seed (int): 乱数シード。

        Returns:
            list[HierarchicalSamplePair]: 生成されたペアサンプルのリスト。
        """
        rng = random.Random(base_seed)
        pairs: list[HierarchicalSamplePair] = []
        for sample in fine_samples:
            pairs.append(self.generate_pair(sample, rng=rng))
        return pairs


def _tokenize_hierarchical_sample(
    sample: UnifiedSample,
    tokenizer: PreTrainedTokenizerFast,
    op_token_id: int,
    max_length: int,
    shuffle_options: bool,
    seed: int | None,
    fine_to_coarse_key: dict[str, str] | None = None,
    coarse_key_to_idx: dict[str, int] | None = None,
    negative_keys: list[str] | None = None,
) -> tuple[dict[str, Tensor], list[str], Tensor | None]:
    """階層用サンプルをトークナイズし、候補キー順序および親クラスタインデックスを追跡する。

    受け皿候補 (None / その他) が存在する場合、スロット 0 への偏位を防ぐため
    中央スロット付近へ再配置する。

    Args:
        sample (UnifiedSample): 対象サンプル。
        tokenizer (PreTrainedTokenizerFast): トークナイザー。
        op_token_id (int): [OP] トークン ID。
        max_length (int): 最大系列長。
        shuffle_options (bool): 候補をシャッフルするかどうか。
        seed (int | None): シャッフル用シード。
        fine_to_coarse_key (dict[str, str] | None): 細分類候補の親大分類キー辞書。
        coarse_key_to_idx (dict[str, int] | None): 大分類キーから整数インデックスへの辞書。
        negative_keys (list[str] | None): 受け皿候補判定用キーリスト。

    Returns:
        tuple[dict[str, Tensor], list[str], Tensor | None]:
            - トークナイズ辞書 (input_ids, attention_mask, op_indices, label)。
            - シャッフル後の候補キー並び順。
            - 各候補スロットの親大分類インデックス列 (Fine 時のみ、Coarse 時は None)。
    """
    option_keys = list(sample.criteria.keys())

    if shuffle_options:
        rng = random.Random(seed)
        rng.shuffle(option_keys)

        # 受け皿候補 (その他 / None) の中央スロット配置
        if negative_keys:
            neg_set = set(negative_keys)
            catchall_in_keys = [k for k in option_keys if k in neg_set]
            if catchall_in_keys and len(option_keys) >= 3:
                # 該当する受け皿候補を一旦除外
                for ck in catchall_in_keys:
                    option_keys.remove(ck)
                # 中央スロットへ挿入
                mid_idx = len(option_keys) // 2
                for ck in catchall_in_keys:
                    option_keys.insert(mid_idx, ck)

    if sample.target not in option_keys:
        raise ValueError(
            f"正解ラベル '{sample.target}' が候補リストに含まれていません: {option_keys}。"
        )
    target_index = option_keys.index(sample.target)

    # Criteria 文字列の構築
    criteria_text = _build_criteria_text(sample, option_keys)

    prefix_text = "State: "
    suffix_text = f"\nInstructions: {sample.instructions}\nCriteria: {criteria_text}"

    prefix_ids: list[int] = tokenizer.encode(prefix_text, add_special_tokens=False)
    suffix_ids: list[int] = tokenizer.encode(suffix_text, add_special_tokens=False)
    state_ids: list[int] = tokenizer.encode(sample.state, add_special_tokens=False)

    leading_special, trailing_special = _get_special_tokens(tokenizer)
    fixed_len = (
        len(leading_special) + len(prefix_ids) + len(suffix_ids) + len(trailing_special)
    )

    if fixed_len >= max_length:
        raise ValueError(
            f"Instructions と Criteria のトークン長 ({fixed_len}) が "
            f"max_length ({max_length}) を超過しているため、State を収容できません。"
        )

    allowed_state_len = max_length - fixed_len
    truncated_state_ids = state_ids[:allowed_state_len]

    final_input_ids = (
        leading_special
        + prefix_ids
        + truncated_state_ids
        + suffix_ids
        + trailing_special
    )

    input_ids_tensor = torch.tensor(final_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.ones_like(input_ids_tensor)

    op_indices = (input_ids_tensor == op_token_id).nonzero(as_tuple=True)[0]
    op_indices_list = op_indices.tolist()

    if len(op_indices_list) != len(option_keys):
        raise ValueError(
            f"[OP] マーカーの検出数 ({len(op_indices_list)}) が "
            f"候補数 ({len(option_keys)}) と一致しません。"
        )

    # Fine サンプルの場合、各スロットに対応する親大分類のインデックスを追跡
    parent_indices_tensor: Tensor | None = None
    if fine_to_coarse_key is not None and coarse_key_to_idx is not None:
        parent_idx_list: list[int] = []
        for k in option_keys:
            parent_key = fine_to_coarse_key.get(k, "")
            parent_idx = coarse_key_to_idx.get(parent_key, -1)
            parent_idx_list.append(parent_idx)
        parent_indices_tensor = torch.tensor(parent_idx_list, dtype=torch.long)

    tokenized = {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "op_indices": op_indices,
        "label": torch.tensor(target_index, dtype=torch.long),
    }

    return tokenized, option_keys, parent_indices_tensor


class HierarchicalJevDataset(Dataset[dict[str, Any]]):
    """大分類と細分類のペアをオンザフライで提供する Dataset。

    Attributes:
        pairs (list[HierarchicalSamplePair]): 階層ペアサンプルのリスト。
        mapping (HierarchicalMapping): 階層オントロジー。
        tokenizer (PreTrainedTokenizerFast): トークナイザー。
        op_token_id (int): [OP] 特殊トークン ID。
        max_length (int): 最大許容系列長。
        is_train (bool): 学習モードフラグ (シャッフルの有効/無効)。
        base_seed (int): 決定論的シード計算用基準値。
        current_epoch (int): 現在のエポック番号。
    """

    def __init__(
        self,
        pairs: list[HierarchicalSamplePair],
        mapping: HierarchicalMapping,
        tokenizer: PreTrainedTokenizerFast,
        op_token_id: int,
        max_length: int = MAX_SEQUENCE_LENGTH,
        is_train: bool = True,
        base_seed: int = 42,
    ) -> None:
        """データセットを初期化する。

        Args:
            pairs (list[HierarchicalSamplePair]): 階層ペアサンプルリスト。
            mapping (HierarchicalMapping): 階層オントロジー。
            tokenizer (PreTrainedTokenizerFast): トークナイザー。
            op_token_id (int): [OP] 特殊トークン ID。
            max_length (int): 最大許容系列長。
            is_train (bool): 学習モードフラグ。
            base_seed (int): 基準シード。
        """
        self.pairs = pairs
        self.mapping = mapping
        self.tokenizer = tokenizer
        self.op_token_id = op_token_id
        self.max_length = max_length
        self.is_train = is_train
        self.base_seed = base_seed
        self.current_epoch = 0

        # 大分類キーの固定順序インデックスマップ
        self.coarse_key_to_idx = {
            k: i for i, k in enumerate(self.mapping.all_coarse_keys)
        }

    def set_epoch(self, epoch: int) -> None:
        """エポック番号を更新する。

        Args:
            epoch (int): 新しいエポック番号。
        """
        self.current_epoch = epoch

    def __len__(self) -> int:
        """サンプル総数を取得する。

        Returns:
            int: サンプル件数。
        """
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """インデックスに対応する Coarse と Fine のトークナイズ結果を返却する。

        Args:
            index (int): サンプルインデックス。

        Returns:
            dict[str, Any]:
                - coarse: Coarse トークナイズ辞書。
                - fine: Fine トークナイズ辞書。
                - fine_parent_indices: 各 Fine 候補スロットの親大分類 ID テンソル。
                - is_soft_beam: Soft-Beam サンプルフラグ。
        """
        pair = self.pairs[index]

        should_shuffle = self.is_train
        seed_coarse = (
            (self.base_seed + self.current_epoch * 1_000_003 + index * 2) & 0xFFFFFFFF
            if should_shuffle
            else None
        )
        seed_fine = (
            (self.base_seed + self.current_epoch * 1_000_003 + index * 2 + 1)
            & 0xFFFFFFFF
            if should_shuffle
            else None
        )

        tokenized_coarse, _, _ = _tokenize_hierarchical_sample(
            sample=pair.coarse_sample,
            tokenizer=self.tokenizer,
            op_token_id=self.op_token_id,
            max_length=self.max_length,
            shuffle_options=should_shuffle,
            seed=seed_coarse,
            negative_keys=self.mapping.negative_keys,
        )

        tokenized_fine, _, parent_indices = _tokenize_hierarchical_sample(
            sample=pair.fine_sample,
            tokenizer=self.tokenizer,
            op_token_id=self.op_token_id,
            max_length=self.max_length,
            shuffle_options=should_shuffle,
            seed=seed_fine,
            fine_to_coarse_key=pair.fine_to_coarse_key,
            coarse_key_to_idx=self.coarse_key_to_idx,
            negative_keys=self.mapping.negative_keys,
        )

        return {
            "coarse": tokenized_coarse,
            "fine": tokenized_fine,
            "fine_parent_indices": parent_indices,
            "is_soft_beam": pair.is_soft_beam,
        }


def hierarchical_collate_fn(
    batch: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Tensor]:
    """階層ペアサンプルのミニバッチを整列・パディングして結合テンソルを構築する。

    Coarse と Fine のそれぞれに対して独立に系列長と候補数をバッチ内最大値でパディングし、
    細分類候補と大分類クラスタを紐付ける `fine_parent_indices` も適切にパディング (-1) する。

    Args:
        batch (list[dict[str, Any]]): サンプル辞書リスト。
        pad_token_id (int): パディングトークン ID。

    Returns:
        dict[str, Tensor]:
            - coarse_input_ids: [B, max_seq_coarse]
            - coarse_attention_mask: [B, max_seq_coarse]
            - coarse_op_indices: [B, max_opt_coarse]
            - coarse_op_mask: [B, max_opt_coarse]
            - coarse_labels: [B]
            - fine_input_ids: [B, max_seq_fine]
            - fine_attention_mask: [B, max_seq_fine]
            - fine_op_indices: [B, max_opt_fine]
            - fine_op_mask: [B, max_opt_fine]
            - fine_labels: [B]
            - fine_parent_indices: [B, max_opt_fine]
            - is_soft_beam: [B] (bool テンソル)
    """
    batch_size = len(batch)

    # 1. Coarse 系列のサイズ計算
    max_seq_coarse = max(item["coarse"]["input_ids"].size(0) for item in batch)
    max_opt_coarse = max(item["coarse"]["op_indices"].size(0) for item in batch)

    # 2. Fine 系列のサイズ計算
    max_seq_fine = max(item["fine"]["input_ids"].size(0) for item in batch)
    max_opt_fine = max(item["fine"]["op_indices"].size(0) for item in batch)

    # Coarse パディングテンソル初期化
    coarse_input_ids = torch.full(
        (batch_size, max_seq_coarse), pad_token_id, dtype=torch.long
    )
    coarse_attention_mask = torch.zeros((batch_size, max_seq_coarse), dtype=torch.long)
    coarse_op_indices = torch.zeros((batch_size, max_opt_coarse), dtype=torch.long)
    coarse_op_mask = torch.zeros((batch_size, max_opt_coarse), dtype=torch.bool)
    coarse_labels = torch.tensor(
        [item["coarse"]["label"].item() for item in batch], dtype=torch.long
    )

    # Fine パディングテンソル初期化
    fine_input_ids = torch.full(
        (batch_size, max_seq_fine), pad_token_id, dtype=torch.long
    )
    fine_attention_mask = torch.zeros((batch_size, max_seq_fine), dtype=torch.long)
    fine_op_indices = torch.zeros((batch_size, max_opt_fine), dtype=torch.long)
    fine_op_mask = torch.zeros((batch_size, max_opt_fine), dtype=torch.bool)
    fine_labels = torch.tensor(
        [item["fine"]["label"].item() for item in batch], dtype=torch.long
    )
    fine_parent_indices = torch.full((batch_size, max_opt_fine), -1, dtype=torch.long)

    is_soft_beam = torch.tensor(
        [bool(item["is_soft_beam"]) for item in batch], dtype=torch.bool
    )

    for i, item in enumerate(batch):
        c = item["coarse"]
        c_seq = c["input_ids"].size(0)
        c_opt = c["op_indices"].size(0)
        coarse_input_ids[i, :c_seq] = c["input_ids"]
        coarse_attention_mask[i, :c_seq] = c["attention_mask"]
        coarse_op_indices[i, :c_opt] = c["op_indices"]
        coarse_op_mask[i, :c_opt] = True

        f = item["fine"]
        f_seq = f["input_ids"].size(0)
        f_opt = f["op_indices"].size(0)
        fine_input_ids[i, :f_seq] = f["input_ids"]
        fine_attention_mask[i, :f_seq] = f["attention_mask"]
        fine_op_indices[i, :f_opt] = f["op_indices"]
        fine_op_mask[i, :f_opt] = True

        p_idx = item.get("fine_parent_indices")
        if p_idx is not None:
            fine_parent_indices[i, : len(p_idx)] = p_idx

    return {
        "coarse_input_ids": coarse_input_ids,
        "coarse_attention_mask": coarse_attention_mask,
        "coarse_op_indices": coarse_op_indices,
        "coarse_op_mask": coarse_op_mask,
        "coarse_labels": coarse_labels,
        "fine_input_ids": fine_input_ids,
        "fine_attention_mask": fine_attention_mask,
        "fine_op_indices": fine_op_indices,
        "fine_op_mask": fine_op_mask,
        "fine_labels": fine_labels,
        "fine_parent_indices": fine_parent_indices,
        "is_soft_beam": is_soft_beam,
    }
