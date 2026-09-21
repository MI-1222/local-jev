"""デシジョンヘッドおよび Jev 統合モデルの単体テストモジュール。"""

import pytest
import torch
from transformers import AutoConfig, AutoModel

from models.decision_head import DecisionHead, JevDecisionModel, OptionGatherLayer


def test_option_gather_layer() -> None:
    """OptionGatherLayer が指定インデックスのベクトルを正確に抽出するか検証する。

    検証項目:
    - 隠れ状態テンソル `[batch_size, seq_len, hidden_size]` から、
      `op_indices` で指定された位置のベクトルが漏れなく抽出されること。
    """
    gather_layer = OptionGatherLayer()

    batch_size = 2
    seq_len = 5
    hidden_size = 4

    hidden_states = torch.arange(
        batch_size * seq_len * hidden_size, dtype=torch.float32
    ).view(batch_size, seq_len, hidden_size)

    op_indices = torch.tensor([[1, 4], [0, 3]], dtype=torch.long)
    extracted = gather_layer(hidden_states, op_indices)

    assert extracted.shape == (2, 2, 4)
    assert torch.equal(extracted[0, 0], hidden_states[0, 1])
    assert torch.equal(extracted[0, 1], hidden_states[0, 4])
    assert torch.equal(extracted[1, 0], hidden_states[1, 0])
    assert torch.equal(extracted[1, 1], hidden_states[1, 3])


def test_decision_head_shape() -> None:
    """DecisionHead が `[batch_size, num_options]` のロジットを出力するか検証する。

    検証項目:
    - 任意の隠れベクトル `[batch_size, num_options, hidden_size]` に対して、
      正しい 2 次元の候補ロジットテンソルが返却されること。
    """
    hidden_size = 768
    head = DecisionHead(hidden_size=hidden_size, mlp_hidden_size=256)

    batch_size = 3
    num_options = 4
    option_vectors = torch.randn(batch_size, num_options, hidden_size)

    logits = head(option_vectors)
    assert logits.shape == (batch_size, num_options)


def test_jev_decision_model_forward_and_backward() -> None:
    """軽量バックボーンを用いた JevDecisionModel のフォワード・バックワード検証。

    検証項目:
    - 出力テンソル形状が `[batch_size, num_options]` となること。
    - 損失のスカラーサムからバックワードを実行した際に、
      デシジョンヘッドおよびバックボーン埋め込み層に勾配が正常に逆伝播すること。
    """
    config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4

    backbone = AutoModel.from_config(config)
    model = JevDecisionModel(backbone=backbone, mlp_hidden_size=64)

    batch_size = 2
    seq_len = 16
    num_options = 3

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    op_indices = torch.tensor([[2, 6, 10], [1, 5, 9]], dtype=torch.long)

    logits = model(input_ids, attention_mask, op_indices)
    assert logits.shape == (batch_size, num_options)

    loss = logits.sum()
    loss.backward()

    assert model.decision_head.dense.weight.grad is not None
    assert backbone.embeddings.tok_embeddings.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])
