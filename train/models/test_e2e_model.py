"""実際のバックボーンを用いた Jev 決定モデルのエンドツーエンド結合テストモジュール。"""

import pytest
import torch

from contract import TOKEN_OPTION_MARKER
from models.backbone import DEFAULT_BACKBONE_MODEL_ID, prepare_backbone_and_tokenizer
from models.decision_head import JevDecisionModel


@pytest.mark.slow
def test_e2e_modernbert_decision_inference() -> None:
    """ModernBERT と結合した JevDecisionModel で実際のテキストからロジットが出力されるか検証する。

    検証項目:
    - 実際の日本語バックボーン (`modernbert-ja-130m`) とデシジョンヘッドを接続し、
      3 候補を含む自然言語プロンプトから `[1, 3]` のロジットが出力されること。
    - 出力ロジットに NaN や Inf が含まれないこと。
    """
    backbone, tokenizer, op_token_id = prepare_backbone_and_tokenizer(
        DEFAULT_BACKBONE_MODEL_ID
    )
    model = JevDecisionModel(backbone=backbone)
    model.eval()

    prompt = (
        f"State: 顧客から届いた商品が破損していたため交換を要求された。"
        f"Instructions: 意図を分類せよ。"
        f"Criteria: {TOKEN_OPTION_MARKER} 返金要望 {TOKEN_OPTION_MARKER} 商品交換 {TOKEN_OPTION_MARKER} その他"
    )

    encoding = tokenizer(prompt, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    op_indices_list = (input_ids[0] == op_token_id).nonzero(as_tuple=True)[0].tolist()
    assert len(op_indices_list) == 3

    op_indices = torch.tensor([op_indices_list], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, op_indices)

    assert logits.shape == (1, 3)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


if __name__ == "__main__":
    pytest.main([__file__])
