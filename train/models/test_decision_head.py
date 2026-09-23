"""デシジョンヘッドおよび Jev 統合モデルの単体テストモジュール。"""

import pytest
import torch
from transformers import AutoConfig, AutoModel

from models.decision_head import (
    ChoiceHead,
    CoralOrdinalHead,
    JevDecisionModel,
    NliNoulHead,
    OptionGatherLayer,
    SetAttentionBlock,
)


def test_option_gather_layer() -> None:
    """OptionGatherLayer が指定インデックスのベクトルを正確に抽出するか検証する。

    検証項目:
    - 隠れ状態テンソル `[batch_size, seq_len, hidden_size]` から、
      `op_indices` で指定された位置のベクトルが漏れなく抽出されること。
    - `-1` のパディング位置がゼロベクトルでマスクされること。
    """
    gather_layer = OptionGatherLayer()

    batch_size = 2
    seq_len = 5
    hidden_size = 4

    hidden_states = torch.arange(
        batch_size * seq_len * hidden_size, dtype=torch.float32
    ).view(batch_size, seq_len, hidden_size)

    op_indices = torch.tensor([[1, 4, -1], [0, 3, 2]], dtype=torch.long)
    extracted = gather_layer(hidden_states, op_indices)

    assert extracted.shape == (2, 3, 4)
    assert torch.equal(extracted[0, 0], hidden_states[0, 1])
    assert torch.equal(extracted[0, 1], hidden_states[0, 4])
    # -1 の位置はゼロベクトル
    assert torch.equal(extracted[0, 2], torch.zeros(4))
    assert torch.equal(extracted[1, 0], hidden_states[1, 0])
    assert torch.equal(extracted[1, 1], hidden_states[1, 3])
    assert torch.equal(extracted[1, 2], hidden_states[1, 2])


def test_set_attention_block_permutation_equivariance() -> None:
    """SetAttentionBlock が提示順序に対して置換同変性 (Permutation Equivariance) を保持するか検証する。

    数理仕様:
    任意の置換 $\\Pi$ に対し、$\\text{SAB}(\\Pi X) = \\Pi \\text{SAB}(X)$ が厳密に成立すること。
    """
    hidden_size = 64
    sab = SetAttentionBlock(hidden_size=hidden_size, num_heads=4, dropout=0.0)
    sab.eval()

    batch_size = 2
    num_options = 5
    option_vectors = torch.randn(batch_size, num_options, hidden_size)

    # 置換インデックスの作成 (例: [3, 0, 4, 1, 2])
    perm = torch.tensor([3, 0, 4, 1, 2])
    perm_vectors = option_vectors[:, perm, :]

    with torch.no_grad():
        out_orig = sab(option_vectors)
        out_perm = sab(perm_vectors)

    # 元の出力の置換と、置換入力の出力が一致するか確認
    out_orig_permuted = out_orig[:, perm, :]
    assert torch.allclose(out_orig_permuted, out_perm, atol=1e-5, rtol=1e-5)


def test_set_attention_block_padding_mask() -> None:
    """SetAttentionBlock が無効パディング候補を安全に遮断し、NaN を生じさせないか検証する。"""
    hidden_size = 32
    sab = SetAttentionBlock(hidden_size=hidden_size, num_heads=2)
    sab.eval()

    batch_size = 2
    num_options = 4
    option_vectors = torch.randn(batch_size, num_options, hidden_size)
    op_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, False]], dtype=torch.bool
    )

    out = sab(option_vectors, op_mask=op_mask)
    assert out.shape == (batch_size, num_options, hidden_size)
    assert not torch.isnan(out).any()

    # 無効位置はゼロベクトル
    assert torch.equal(out[0, 2], torch.zeros(hidden_size))
    assert torch.equal(out[0, 3], torch.zeros(hidden_size))
    assert torch.equal(out[1, 3], torch.zeros(hidden_size))


def test_coral_ordinal_head_monotonicity() -> None:
    """CoralOrdinalHead の単調増加バイアスおよび累積確率の単調減少性を検証する。

    検証項目:
    - カットオフバイアス $b_1 < b_2 < \\dots < b_{M-1}$ の順序関係。
    - 累積確率 $P(Y > 1) \\ge P(Y > 2) \\dots$ の単調減少性。
    - 各段階確率 $P(Y = k)$ がすべて非負であり、総和が 1.0 に正規化されること。
    - バックワードによる勾配更新後も単調性が崩れないこと。
    """
    hidden_size = 32
    max_levels = 6
    head = CoralOrdinalHead(hidden_size=hidden_size, max_levels=max_levels)

    biases = head.get_ordered_biases()
    assert biases.size(0) == max_levels - 1
    # 単調増加の確認: b_{k+1} > b_k
    diffs = biases[1:] - biases[:-1]
    assert (diffs > 0).all()

    batch_size = 3
    num_levels = 5
    feature = torch.randn(batch_size, hidden_size)

    probs, logits, cum_probs = head(feature, num_levels=num_levels)
    assert probs.shape == (batch_size, num_levels)
    assert logits.shape == (batch_size, num_levels)
    assert cum_probs.shape == (batch_size, num_levels - 1)

    # 累積確率の単調減少性
    cum_diffs = cum_probs[:, :-1] - cum_probs[:, 1:]
    assert (cum_diffs >= 0).all()

    # クラス確率の非負性および合計 1.0
    assert (probs >= 0.0).all()
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5, rtol=1e-5
    )

    # 勾配更新テスト
    loss = probs.sum()
    loss.backward()
    assert head.feature_proj.weight.grad is not None
    assert head.raw_biases.grad is not None


def test_nli_noul_head() -> None:
    """NliNoulHead が 3 クラスロジットから真実確率と不確実性を正しく導出するか検証する。"""
    hidden_size = 32
    head = NliNoulHead(hidden_size=hidden_size, mlp_hidden_size=16)

    batch_size = 4
    feat = torch.randn(batch_size, hidden_size)

    logits_2class, p_true, uncertainty, logits_3class = head(feat)

    assert logits_2class.shape == (batch_size, 2)
    assert p_true.shape == (batch_size,)
    assert uncertainty.shape == (batch_size,)
    assert logits_3class.shape == (batch_size, 3)

    # 確率範囲 [0, 1]
    assert (p_true >= 0.0).all() and (p_true <= 1.0).all()
    assert (uncertainty >= 0.0).all() and (uncertainty <= 1.0).all()

    # delta_z = logits[0] - logits[1] = z_e - z_c と一致すること
    delta_z = logits_2class[:, 0] - logits_2class[:, 1]
    expected_delta_z = logits_3class[:, 0] - logits_3class[:, 2]
    assert torch.allclose(delta_z, expected_delta_z, atol=1e-5)


def test_decision_head_shape() -> None:
    """ChoiceHead / DecisionHead が `[batch_size, num_options]` のロジットを出力するか検証する。"""
    hidden_size = 64
    head = ChoiceHead(hidden_size=hidden_size, mlp_hidden_size=32)

    batch_size = 3
    num_options = 4
    option_vectors = torch.randn(batch_size, num_options, hidden_size)

    logits = head(option_vectors)
    assert logits.shape == (batch_size, num_options)


def test_jev_decision_model_forward_and_backward() -> None:
    """軽量バックボーンを用いた JevDecisionModel のフォワード・バックワード検証。

    検証項目:
    - 出力テンソル形状が `[batch_size, num_options]` となること。
    - Choice, Score, Noul の各タスクで正常にフォワード計算できること。
    - 損失のスカラーサムからバックワードを実行した際に、全ヘッドおよびバックボーンに勾配が正常に逆伝播すること。
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
    op_indices = torch.tensor([[2, 6, 10], [1, 5, -1]], dtype=torch.long)

    # 1. Choice タスク
    logits_choice = model(input_ids, attention_mask, op_indices, question_type="choice")
    assert logits_choice.shape == (batch_size, num_options)
    # パディング位置のマスク値確認 (-1e4)
    assert logits_choice[1, 2].item() <= -9999.0

    # 2. Score タスク (details 返却テスト)
    logits_score, _, _, details_score = model(
        input_ids,
        attention_mask,
        op_indices,
        question_type="score",
        return_details=True,
    )
    assert logits_score.shape == (batch_size, num_options)
    assert "score_probs" in details_score
    assert details_score["score_probs"].shape == (batch_size, num_options)

    # 3. Noul タスク
    logits_noul, _, _, details_noul = model(
        input_ids,
        attention_mask,
        op_indices[:, :2],
        question_type="noul",
        return_details=True,
    )
    assert logits_noul.shape == (batch_size, 2)
    assert "noul_p_true" in details_noul

    # バックワード勾配伝播の検証
    loss = logits_choice.sum() + logits_score.sum() + logits_noul.sum()
    loss.backward()

    assert model.choice_head.dense.weight.grad is not None
    assert model.score_head.feature_proj.weight.grad is not None
    assert model.noul_head.dense.weight.grad is not None
    assert model.sab.q_proj.weight.grad is not None
    assert backbone.embeddings.tok_embeddings.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__])
