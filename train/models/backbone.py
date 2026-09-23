"""バックボーンモデルおよびトークナイザー構築モジュール。

ModernBERT / mmBERT などの双方向エンコーダをロードし、
決定アンカーとなる特殊マーカートークン `[OP]` を登録・初期化する。
"""

import logging
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from contract import TOKEN_OPTION_MARKER

logger = logging.getLogger(__name__)

DEFAULT_MODERNBERT_JA_MODEL_ID = "sbintuitions/modernbert-ja-130m"
"""第一推奨の日本語特化エンコーダモデル識別子。"""

DEFAULT_BACKBONE_MODEL_ID = DEFAULT_MODERNBERT_JA_MODEL_ID
"""標準のバックボーンエンコーダモデル識別子。"""

DEFAULT_MMBERT_MODEL_ID = "jhu-clsp/mmBERT-base"
"""代替の多言語エンコーダモデル識別子。"""

DEFAULT_MODERNBERT_MODEL_ID = "answerdotai/ModernBERT-base"
"""旧バージョンの英語特化エンコーダモデル識別子(後方互換用)。"""


def initialize_token_embedding_with_normalized_centroid(
    embedding_weight: Tensor,
    target_token_id: int,
    existing_vocab_size: int,
    epsilon: float = 1e-8,
) -> float:
    """既存語彙埋め込みのセントロイドを正規化し、指定トークンIDの埋め込みベクトルを初期化する。

    数理的背景:
    既存語彙埋め込みの単純な平均ベクトル(セントロイド)は、各ベクトルの方向が分散しているために
    高次元空間で相殺され、L2 ノルムが極端に縮退する。
    そこで、セントロイドベクトルを L2 正規化して方向を取り出し、
    既存トークン群の平均 L2 ノルムでスケーリングすることで、
    既存埋め込み空間の中心的な指向性と健全な振幅(ノルム)を両立させた初期化を行う。

    Args:
        embedding_weight (Tensor): 埋め込み層の重みテンソル `[V, D]`。
        target_token_id (int): 初期化対象のトークンID。
        existing_vocab_size (int): 既存語彙数(セントロイド計算に用いるスライスの終端インデックス)。
        epsilon (float): ゼロ除算防止用の微小値。

    Returns:
        float: スケーリングに用いた既存トークンの平均 L2 ノルム値。
    """
    with torch.no_grad():
        existing_weights = embedding_weight[:existing_vocab_size]
        centroid = existing_weights.mean(dim=0)
        centroid_norm = torch.norm(centroid, p=2)

        avg_token_norm = torch.norm(existing_weights, p=2, dim=1).mean()

        normalized_centroid = (centroid / (centroid_norm + epsilon)) * avg_token_norm
        embedding_weight[target_token_id].copy_(normalized_centroid)
        return float(avg_token_norm.item())


def prepare_backbone_and_tokenizer(
    model_name_or_path: str = DEFAULT_BACKBONE_MODEL_ID,
    dtype: torch.dtype = torch.float32,
) -> tuple[PreTrainedModel, PreTrainedTokenizerFast, int]:
    """バックボーンエンコーダとトークナイザーを準備し、`[OP]` 特殊トークンを登録する。

    内部処理の流れ:
    1. トークナイザーをロードし、語彙に `[OP]` が存在しない場合は追加登録する。
    2. バックボーンモデルをロードする。
    3. 語彙数の拡張に応じてモデルの入力埋め込み層 (`input_embeddings`) をリサイズする。
    4. 新規追加された `[OP]` の埋め込みベクトルを、既存トークンのセントロイド正規化ベクトルで初期化し学習の安定化を図る。

    Args:
        model_name_or_path (str): Hugging Face モデルID またはローカルディレクトリパス。
        dtype (torch.dtype): モデルのテンソル精度。

    Returns:
        tuple[PreTrainedModel, PreTrainedTokenizerFast, int]:
            - 初期化済みバックボーンエンコーダ。
            - 特殊トークン登録済みトークナイザー。
            - `[OP]` マーカーのトークンID。

    Raises:
        ValueError: トークナイザーのロードに失敗した場合。
    """
    logger.info("トークナイザーをロード中: %s...", model_name_or_path)
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if loaded_tokenizer is None:
        raise ValueError(
            f"トークナイザーのロードに失敗しました: {model_name_or_path}。"
        )
    tokenizer = cast(PreTrainedTokenizerFast, loaded_tokenizer)

    vocab = tokenizer.get_vocab()
    if TOKEN_OPTION_MARKER not in vocab:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [TOKEN_OPTION_MARKER]}
        )
        logger.info(
            "特殊トークン %s を追加しました(追加件数: %d)。",
            TOKEN_OPTION_MARKER,
            num_added,
        )
    else:
        num_added = 0
        logger.info("特殊トークン %s は既に語彙に存在します。", TOKEN_OPTION_MARKER)

    token_id_raw = tokenizer.convert_tokens_to_ids(TOKEN_OPTION_MARKER)
    op_token_id: int = (
        token_id_raw[0] if isinstance(token_id_raw, list) else token_id_raw
    )
    logger.info("特殊トークン %s のトークンID: %d", TOKEN_OPTION_MARKER, op_token_id)

    logger.info("バックボーンモデルをロード中: %s...", model_name_or_path)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        dtype=dtype,
    )

    current_vocab_size = len(tokenizer)
    embeddings_module = cast(nn.Embedding, model.get_input_embeddings())
    previous_vocab_size = embeddings_module.weight.shape[0]

    if current_vocab_size != previous_vocab_size:
        logger.info(
            "埋め込み層をリサイズ中: %d -> %d...",
            previous_vocab_size,
            current_vocab_size,
        )
        model.resize_token_embeddings(current_vocab_size)

        new_embeddings = cast(nn.Embedding, model.get_input_embeddings())
        target_norm = initialize_token_embedding_with_normalized_centroid(
            embedding_weight=new_embeddings.weight,
            target_token_id=op_token_id,
            existing_vocab_size=previous_vocab_size,
        )
        logger.info(
            "[OP] トークン(ID: %d)の埋め込みを既存トークンのセントロイド正規化ベクトルで初期化しました(目標ノルム: %.4f)。",
            op_token_id,
            target_norm,
        )

    return model, tokenizer, op_token_id


def save_tokenizer_for_runtime(
    tokenizer: PreTrainedTokenizerFast, output_dir: Path | str
) -> Path:
    """Rust 推論ランタイムで読み込み可能な `tokenizer.json` を含む設定一式を出力する。

    Args:
        tokenizer (PreTrainedTokenizerFast): `[OP]` トークン登録済みのトークナイザー。
        output_dir (Path | str): 出力先ディレクトリ。

    Returns:
        Path: 出力先ディレクトリパス。
    """
    export_path = Path(output_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(export_path)
    logger.info(
        "トークナイザー設定を %s にエクスポートしました(`tokenizer.json` を含む)。",
        export_path,
    )
    return export_path


def verify_option_marker_tokenization(
    tokenizer: PreTrainedTokenizerFast, op_token_id: int
) -> bool:
    """`[OP]` トークンが正しく単一IDとしてエンコード・デコードされるか検証する。

    Args:
        tokenizer (PreTrainedTokenizerFast): 検証対象のトークナイザー。
        op_token_id (int): 登録された `[OP]` のトークンID。

    Returns:
        bool: 検証に合格した場合は True、失敗した場合は False。
    """
    sample_text = (
        f"Criteria: {TOKEN_OPTION_MARKER} 返金要望 {TOKEN_OPTION_MARKER} 配送確認"
    )
    token_ids = cast(Tensor, tokenizer.encode(sample_text, add_special_tokens=False))
    token_ids_list: list[int] = (
        token_ids if isinstance(token_ids, list) else token_ids.tolist()
    )

    found_op_count = token_ids_list.count(op_token_id)
    if found_op_count != 2:
        logger.error(
            "トークナイズ検証失敗: 期待される [OP] 出現数 2 回に対し、%d 回検出されました。",
            found_op_count,
        )
        return False

    decoded = tokenizer.decode(token_ids)
    if TOKEN_OPTION_MARKER not in decoded:
        logger.error("デコード検証失敗: [OP] が復元されませんでした: %s", decoded)
        return False

    logger.info("トークナイズ検証成功: [OP] が単一IDとして正常に保持されています。")
    return True
