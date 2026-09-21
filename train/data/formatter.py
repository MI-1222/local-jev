"""プロンプト整形およびモデル入力トークナイズモジュール。

UnifiedSample を [OP] マーカー付きプロンプトへ変換し、
State 優先トランケーションを適用して Criteria の欠落を完全に防ぎつつ
Jev 決定ヘッド用の入力テンソルを構築する。
"""

import random

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from contract import MAX_SEQUENCE_LENGTH, TOKEN_OPTION_MARKER
from data.schema import QuestionType, UnifiedSample


def format_prompt(
    sample: UnifiedSample,
    shuffle_options: bool = False,
    seed: int | None = None,
) -> tuple[str, list[str]]:
    """統一サンプルからプロンプト文字列と候補キー順序リストを生成する。

    Choice 型では Criteria の各候補の先頭に [OP] マーカーを付与して結合する。
    Noul 型では暗黙の真偽2候補 '[OP] 真 (True) [OP] 偽 (False)' を展開する。

    Args:
        sample (UnifiedSample): 統一中間サンプル。
        shuffle_options (bool): 候補順序をランダムシャッフルするかどうか。
        seed (int | None): シャッフル用の乱数シード。

    Returns:
        tuple[str, list[str]]:
            - 生成されたプロンプト文字列。
            - プロンプト内での候補キーの並び順リスト。

    Raises:
        NotImplementedError: 未対応の QuestionType が指定された場合。
    """
    if sample.question_type == QuestionType.CHOICE:
        option_keys = list(sample.criteria.keys())
        if shuffle_options:
            rng = random.Random(seed)
            rng.shuffle(option_keys)

        criteria_parts = [
            f"{TOKEN_OPTION_MARKER} {sample.criteria[k]}" for k in option_keys
        ]
        criteria_text = " ".join(criteria_parts)

        prompt = (
            f"State: {sample.state}\n"
            f"Instructions: {sample.instructions}\n"
            f"Criteria: {criteria_text}"
        )
        return prompt, option_keys

    elif sample.question_type == QuestionType.NOUL:
        # Noul 型は真偽二値候補として展開
        option_keys = ["true", "false"]
        if shuffle_options:
            rng = random.Random(seed)
            rng.shuffle(option_keys)

        label_map = {
            "true": "真 (True)",
            "false": "偽 (False)",
        }
        criteria_parts = [f"{TOKEN_OPTION_MARKER} {label_map[k]}" for k in option_keys]
        criteria_text = " ".join(criteria_parts)

        prompt = (
            f"State: {sample.state}\n"
            f"Instructions: {sample.instructions}\n"
            f"Criteria: {criteria_text}"
        )
        return prompt, option_keys

    elif sample.question_type == QuestionType.SCORE:
        # Score 型: 評価段階を順序通りに展開
        option_keys = list(sample.criteria.keys())
        criteria_parts = [
            f"{TOKEN_OPTION_MARKER} {sample.criteria[k]}" for k in option_keys
        ]
        criteria_text = " ".join(criteria_parts)

        prompt = (
            f"State: {sample.state}\n"
            f"Instructions: {sample.instructions}\n"
            f"Criteria: {criteria_text}"
        )
        return prompt, option_keys

    else:
        raise NotImplementedError(f"未対応の質問タイプです: {sample.question_type}。")


def _get_special_tokens(
    tokenizer: PreTrainedTokenizerFast,
) -> tuple[list[int], list[int]]:
    """トークナイザーに応じた先頭および末尾の特殊トークンID列を取得する。

    Args:
        tokenizer (PreTrainedTokenizerFast): 対象トークナイザー。

    Returns:
        tuple[list[int], list[int]]: (先頭特殊トークンID列, 末尾特殊トークンID列)。
    """
    leading: list[int] = []
    trailing: list[int] = []

    if tokenizer.cls_token_id is not None:
        leading.append(tokenizer.cls_token_id)
    elif tokenizer.bos_token_id is not None:
        leading.append(tokenizer.bos_token_id)

    if tokenizer.sep_token_id is not None:
        trailing.append(tokenizer.sep_token_id)
    elif tokenizer.eos_token_id is not None:
        trailing.append(tokenizer.eos_token_id)

    return leading, trailing


def tokenize_sample(
    sample: UnifiedSample,
    tokenizer: PreTrainedTokenizerFast,
    op_token_id: int,
    max_length: int = MAX_SEQUENCE_LENGTH,
    shuffle_options: bool = False,
    seed: int | None = None,
) -> dict[str, Tensor]:
    """統一サンプルをモデル入力用テンソル辞書へ安全に変換する。

    【State優先トランケーション保証】:
    系列長が max_length を超過する場合、文末にある Criteria ([OP] マーカー群) や
    Instructions を絶対に切り落とさず、State 側のトークンを優先的に切り詰める。
    これにより [OP] マーカーの総数維持と正解ラベルインデックスの整合性を保証する。

    Args:
        sample (UnifiedSample): 変換対象の統一サンプル。
        tokenizer (PreTrainedTokenizerFast): 特殊トークン登録済みトークナイザー。
        op_token_id (int): [OP] 特殊トークンの ID。
        max_length (int): 許容最大トークン系列長。
        shuffle_options (bool): 候補の順序をシャッフルするかどうか。
        seed (int | None): シャッフル用シード。

    Returns:
        dict[str, Tensor]:
            - input_ids (Tensor): 形状 (seq_len,) の入力トークンID列。
            - attention_mask (Tensor): 形状 (seq_len,) のアテンションマスク。
            - op_indices (Tensor): 形状 (num_options,) の [OP] 出現位置インデックス列。
            - label (Tensor): 正解候補インデックススカラー (dtype=torch.long)。

    Raises:
        ValueError: Criteria と Instructions だけで max_length を超過した場合、
            または [OP] 検出数と候補数が不一致の場合。
    """
    _, option_keys = format_prompt(sample, shuffle_options=shuffle_options, seed=seed)

    if sample.target not in option_keys:
        raise ValueError(
            f"正解ラベル '{sample.target}' が候補リストに含まれていません: {option_keys}。"
        )
    target_index = option_keys.index(sample.target)

    # 1. Criteria 文字列の構築 ([OP] の前後に半角スペースを確保)
    if sample.question_type == QuestionType.NOUL:
        label_map = {"true": "真 (True)", "false": "偽 (False)"}
        criteria_parts = [f"{TOKEN_OPTION_MARKER} {label_map[k]}" for k in option_keys]
    else:
        criteria_parts = [
            f"{TOKEN_OPTION_MARKER} {sample.criteria[k]}" for k in option_keys
        ]
    criteria_text = " ".join(criteria_parts)

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

    # State の長さを許容内に切り詰める (末尾トランケーション)
    allowed_state_len = max_length - fixed_len
    truncated_state_ids = state_ids[:allowed_state_len]

    # トークンID列を結合
    final_input_ids = (
        leading_special
        + prefix_ids
        + truncated_state_ids
        + suffix_ids
        + trailing_special
    )

    input_ids_tensor = torch.tensor(final_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.ones_like(input_ids_tensor)

    # [OP] トークン位置の抽出
    op_indices = (input_ids_tensor == op_token_id).nonzero(as_tuple=True)[0]
    op_indices_list = op_indices.tolist()

    if len(op_indices_list) != len(option_keys):
        raise ValueError(
            f"[OP] マーカーの検出数 ({len(op_indices_list)}) が "
            f"候補数 ({len(option_keys)}) と一致しません。"
        )

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "op_indices": op_indices,
        "label": torch.tensor(target_index, dtype=torch.long),
    }
