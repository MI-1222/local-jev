"""事後温度較正 (Post-hoc Calibration) 単体テストスイート。

設定の双方向シリアライズ、バケット判定整合性、パディングマスク完全性、
精度不変性 (Accuracy Invariance)、ECE/NLL 計算、スカラー最適化エンジン、
および Rust 契約互換性を包括的に検証する。
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from calibration.config import (
    CalibrationRunConfig,
    matches_bucket_expr,
)
from calibration.evaluator import (
    compute_accuracy,
    compute_brier_score,
    compute_ece,
    compute_reliability_diagram_data,
    get_masked_probabilities,
)
from calibration.optimizer import (
    LogitCache,
    TemperatureOptimizer,
)
from contract import CalibrationConfig
from data.schema import QuestionType


def test_calibration_config_serialization() -> None:
    """CalibrationRunConfig の YAML/JSON シリアライズおよび復元が完全であるかを検証する。"""
    config = CalibrationRunConfig(
        checkpoint_path="runs/sft/best_checkpoint",
        tau_min=0.1,
        tau_max=4.0,
        min_samples_per_bucket=20,
        default_temperature=1.05,
        num_bins=15,
        max_val_samples=500,
        batch_size=32,
        seed=123,
        output_dir="runs/calibration/custom",
    )

    # 辞書変換
    d = config.to_dict()
    restored_dict = CalibrationRunConfig.from_dict(d)
    assert restored_dict.tau_min == pytest.approx(0.1)
    assert restored_dict.min_samples_per_bucket == 20

    # YAML 変換
    yaml_str = config.to_yaml()
    restored_yaml = CalibrationRunConfig.from_yaml(yaml_str)
    assert restored_yaml.checkpoint_path == "runs/sft/best_checkpoint"
    assert restored_yaml.num_bins == 15

    # JSON 変換
    json_str = config.to_json()
    restored_json = CalibrationRunConfig.from_json(json_str)
    assert restored_json.seed == 123
    assert restored_json.output_dir == "runs/calibration/custom"

    # ファイル永続化と読み込み
    with TemporaryDirectory() as tmp_dir:
        yaml_file = Path(tmp_dir) / "config.yaml"
        config.save(yaml_file)
        loaded_yaml = CalibrationRunConfig.load(yaml_file)
        assert loaded_yaml.tau_max == pytest.approx(4.0)

        json_file = Path(tmp_dir) / "config.json"
        config.save(json_file)
        loaded_json = CalibrationRunConfig.load(json_file)
        assert loaded_json.batch_size == 32


def test_matches_bucket_expr() -> None:
    """バケットマッチングルールが Rust 実装と厳密に同一であることを検証する。"""
    # 完全一致
    assert matches_bucket_expr("2", 2)
    assert not matches_bucket_expr("2", 3)

    # 範囲 (3-5)
    assert matches_bucket_expr("3-5", 3)
    assert matches_bucket_expr("3-5", 4)
    assert matches_bucket_expr("3-5", 5)
    assert not matches_bucket_expr("3-5", 2)
    assert not matches_bucket_expr("3-5", 6)

    # 範囲 (6-10)
    assert matches_bucket_expr("6-10", 6)
    assert matches_bucket_expr("6-10", 10)
    assert not matches_bucket_expr("6-10", 5)
    assert not matches_bucket_expr("6-10", 11)

    # 上限なし (11+)
    assert matches_bucket_expr("11+", 11)
    assert matches_bucket_expr("11+", 50)
    assert not matches_bucket_expr("11+", 10)


def test_get_masked_probabilities_integrity() -> None:
    """温度適用後に無効候補領域の確率が厳密にゼロとなることを検証する。"""
    # 3候補中、最後の1候補が無効
    logits = torch.tensor([[10.0, 5.0, 2.0]], dtype=torch.float32)
    op_mask = torch.tensor([[True, True, False]], dtype=torch.bool)

    # 高温 tau = 5.0 を適用
    probs = get_masked_probabilities(logits, op_mask, temperature=5.0)

    # 無効候補の確率は 0.0 でなければならない
    assert probs[0, 2].item() == pytest.approx(0.0)
    # 有効候補の確率の和は 1.0 でなければならない
    assert (probs[0, 0] + probs[0, 1]).item() == pytest.approx(1.0, abs=1e-5)


def test_accuracy_invariance() -> None:
    """温度スケーリングの前後で Top-1 正解率が一切変化しないことを検証する。"""
    torch.manual_seed(42)
    batch_size = 50
    num_options = 5

    logits = torch.randn(batch_size, num_options)
    labels = torch.randint(0, num_options, (batch_size,))
    op_mask = torch.ones(batch_size, num_options, dtype=torch.bool)

    acc_base = compute_accuracy(logits, labels, op_mask)

    # 極端な低温 (0.2) と高温 (4.0) での精度比較
    for tau in [0.2, 0.5, 1.5, 3.0, 5.0]:
        acc_scaled = compute_accuracy(logits / tau, labels, op_mask)
        assert acc_base == pytest.approx(acc_scaled, abs=1e-6)


def test_ece_and_nll_metrics() -> None:
    """ECE、NLL、Brier スコアの数理計算が正しく機能することを検証する。"""
    # 完全な正解確信度 (過信なし、完全較正)
    probs_perfect = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels_perfect = torch.tensor([0, 1], dtype=torch.long)
    op_mask = torch.ones_like(probs_perfect, dtype=torch.bool)

    ece_perfect = compute_ece(probs_perfect, labels_perfect, op_mask, num_bins=10)
    assert ece_perfect == pytest.approx(0.0, abs=1e-5)

    brier_perfect = compute_brier_score(probs_perfect, labels_perfect, op_mask)
    assert brier_perfect == pytest.approx(0.0, abs=1e-5)

    # 完全な誤信 (確信度 1.0 で誤答)
    probs_wrong = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    ece_wrong = compute_ece(probs_wrong, labels_perfect, op_mask, num_bins=10)
    assert ece_wrong == pytest.approx(1.0, abs=1e-5)

    diagram = compute_reliability_diagram_data(
        probs_perfect, labels_perfect, op_mask, num_bins=10
    )
    assert len(diagram) == 10
    assert diagram[-1]["count"] == 2  # 確信度 1.0 のサンプルが最上位ビンに収まる


def test_temperature_optimizer_synthetic() -> None:
    """合成過信データに対して温度最適化を実行し、ECE が改善されることを検証する。"""
    torch.manual_seed(42)
    n_samples = 200
    n_options = 4

    # 意図的に過信したロジット分布 (極端にシャープなロジット)
    # 正解ラベル
    labels = torch.randint(0, n_options, (n_samples,))
    # ノイズを含む正解ロジット
    base_logits = torch.randn(n_samples, n_options) * 0.5
    for i in range(n_samples):
        # 正解率 約 70% になるように設定
        if i % 10 < 7:
            base_logits[i, labels[i]] += 4.0  # 過剰に大きなロジット (過信)
        else:
            wrong_idx = (labels[i] + 1) % n_options
            base_logits[i, wrong_idx] += 4.0

    op_mask = torch.ones(n_samples, n_options, dtype=torch.bool)

    cache = LogitCache(
        logits=base_logits,
        op_mask=op_mask,
        labels=labels,
        question_types=[QuestionType.CHOICE.value] * n_samples,
        candidate_counts=[n_options] * n_samples,
    )

    config = CalibrationRunConfig(
        tau_min=0.1,
        tau_max=5.0,
        min_samples_per_bucket=30,
        default_temperature=1.0,
        num_bins=10,
    )
    optimizer = TemperatureOptimizer(config=config)

    # 3-5 バケットの最適温度を探索
    calib_config, summary = optimizer.calibrate(cache)

    choice_3_5_tau = calib_config.temperature_map.choice["3-5"]
    # 過信を緩和するため tau* > 1.0 になるはずである
    assert choice_3_5_tau > 1.0

    # 3-5 バケットの事後 ECE が事前 ECE より大幅に改善していることを確認
    b_metrics = summary["buckets"]["choice_3-5"]
    assert b_metrics["post_calibration"]["ece"] < b_metrics["pre_calibration"]["ece"]
    assert (
        b_metrics["post_calibration"]["accuracy"]
        == b_metrics["pre_calibration"]["accuracy"]
    )


def test_sparse_bucket_fallback() -> None:
    """サンプル数が統計的有意性閾値に満たないバケットがフォールバックされることを検証する。"""
    # わずか 3 サンプルしかないキャッシュ
    logits = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 2])
    op_mask = torch.ones(3, 4, dtype=torch.bool)

    cache = LogitCache(
        logits=logits,
        op_mask=op_mask,
        labels=labels,
        question_types=[QuestionType.CHOICE.value] * 3,
        candidate_counts=[4] * 3,
    )

    # min_samples_per_bucket = 15 に設定
    config = CalibrationRunConfig(
        min_samples_per_bucket=15,
        default_temperature=1.0,
    )
    optimizer = TemperatureOptimizer(config=config)

    _calib_config, summary = optimizer.calibrate(cache)
    m = summary["buckets"]["choice_3-5"]
    assert m["is_fallback"] is True
    assert m["temperature"] == 1.0


def test_contract_roundtrip_compatibility() -> None:
    """生成された成果物一式が contract.py および Rust 側仕様と完全に合致することを検証する。"""
    config = CalibrationRunConfig()
    optimizer = TemperatureOptimizer(config=config)

    # 仮想キャッシュ
    cache = LogitCache(
        logits=torch.randn(20, 2),
        op_mask=torch.ones(20, 2, dtype=torch.bool),
        labels=torch.randint(0, 2, (20,)),
        question_types=[QuestionType.NOUL.value] * 20,
        candidate_counts=[2] * 20,
    )

    calib_config, summary = optimizer.calibrate(cache)

    with TemporaryDirectory() as tmp_dir:
        run_dir = optimizer.save_run_artifacts(
            output_dir=tmp_dir,
            calib_config=calib_config,
            metrics_summary=summary,
        )

        calib_file = run_dir / "calibration.json"
        assert calib_file.exists()

        # contract.py の CalibrationConfig でロード可能であることを検証
        loaded_contract = CalibrationConfig.load(calib_file)
        assert loaded_contract.version == "1.0"
        assert isinstance(loaded_contract.temperature_map.choice, dict)
        assert isinstance(loaded_contract.temperature_map.score, dict)
        assert isinstance(loaded_contract.temperature_map.noul, float)

        # 必須成果物ファイルの存在確認
        assert (run_dir / "calibration_run_config.yaml").exists()
        assert (run_dir / "calibration_metrics.json").exists()
        assert (run_dir / "run_metadata.json").exists()
        assert (run_dir / "summary.md").exists()


def test_collect_logits_signature() -> None:
    """collect_logits が op_mask なしで model を呼び出し、正しく LogitCache を構築することを検証する。"""
    from typing import Any

    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class DummyModel(nn.Module):
        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            op_indices: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, num_options = op_indices.shape
            return torch.zeros(batch_size, num_options)

    class DummyDataset(Dataset[dict[str, Any]]):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "input_ids": torch.tensor([[1, 2, 3], [1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
                "op_indices": torch.tensor([[0, 1], [0, 1]]),
                "op_mask": torch.tensor([[True, True], [True, True]]),
                "labels": torch.tensor([0, 1]),
                "question_types": [
                    QuestionType.CHOICE.value,
                    QuestionType.CHOICE.value,
                ],
            }

    dataloader: DataLoader[dict[str, Any]] = DataLoader(
        DummyDataset(),
        batch_size=None,
    )

    optimizer = TemperatureOptimizer()
    cache = optimizer.collect_logits(
        model=DummyModel(),
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    assert cache.logits.shape == (2, 2)
    assert cache.op_mask.shape == (2, 2)
    assert cache.labels.shape == (2,)
    assert len(cache.question_types) == 2
    assert len(cache.candidate_counts) == 2


def test_collect_logits_heterogeneous_options() -> None:
    """異なる選択肢数を持つバッチ群 (例: Banking77 の 77 択と MNLI の 3 択) がパディング結合されることを検証する。"""
    from typing import Any

    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    class DynamicOptionModel(nn.Module):
        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            op_indices: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, num_options = op_indices.shape
            return torch.full((batch_size, num_options), 2.0)

    class MultiBatchDataset(Dataset[dict[str, Any]]):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, Any]:
            if index == 0:
                # 77 択バッチ
                return {
                    "input_ids": torch.ones(1, 10, dtype=torch.long),
                    "attention_mask": torch.ones(1, 10, dtype=torch.long),
                    "op_indices": torch.zeros(1, 77, dtype=torch.long),
                    "op_mask": torch.ones(1, 77, dtype=torch.bool),
                    "labels": torch.tensor([0]),
                    "question_types": [QuestionType.CHOICE.value],
                }
            # 3 択バッチ
            return {
                "input_ids": torch.ones(1, 10, dtype=torch.long),
                "attention_mask": torch.ones(1, 10, dtype=torch.long),
                "op_indices": torch.zeros(1, 3, dtype=torch.long),
                "op_mask": torch.ones(1, 3, dtype=torch.bool),
                "labels": torch.tensor([1]),
                "question_types": [QuestionType.CHOICE.value],
            }

    dataloader: DataLoader[dict[str, Any]] = DataLoader(
        MultiBatchDataset(),
        batch_size=None,
    )

    optimizer = TemperatureOptimizer()
    cache = optimizer.collect_logits(
        model=DynamicOptionModel(),
        dataloader=dataloader,
        device=torch.device("cpu"),
    )

    # 全体サイズは最大選択肢数 77 にパディングされていることを検証
    assert cache.logits.shape == (2, 77)
    assert cache.op_mask.shape == (2, 77)
    assert cache.labels.shape == (2,)

    # バッチ0 (77 択) は全 77 列が有効
    assert cache.op_mask[0].sum().item() == 77
    # バッチ1 (3 択) は最初の 3 列のみ有効、残りは False
    assert cache.op_mask[1].sum().item() == 3
    assert cache.op_mask[1, :3].all().item() is True
    assert (~cache.op_mask[1, 3:]).all().item() is True

    # filter_by_bucket で 3 択バケット ("3-5") を抽出した際、有効最大数に合わせて 3 列にトリムされることを検証
    sub_logits, sub_labels, sub_mask = cache.filter_by_bucket(
        target_type=QuestionType.CHOICE,
        bucket_expr="3-5",
    )
    assert sub_logits.shape == (1, 3)
    assert sub_mask.shape == (1, 3)
    assert sub_labels.tolist() == [1]
