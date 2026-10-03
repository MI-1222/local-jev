"""HierarchicalMapping およびオントロジー定義の単体テスト。"""

from pathlib import Path

import pytest

from data.hierarchical import HierarchicalMapping


def test_banking77_preset_integrity() -> None:
    """Banking77 プリセットが 7 大分類・77 細分類として厳密に MECE であることを検証する。"""
    mapping = HierarchicalMapping.banking77()
    assert mapping.name == "Banking77-7Clusters"
    assert len(mapping.coarse_categories) == 7
    assert len(mapping.fine_criteria) == 77
    assert len(mapping.all_coarse_keys) == 7
    assert len(mapping.all_fine_keys) == 77

    # 全細分類が重複なく親大分類に割り当てられていること
    assert len(mapping.fine_to_coarse) == 77
    total_in_coarse = sum(len(f_list) for f_list in mapping.coarse_to_fine.values())
    assert total_in_coarse == 77

    # 受け皿候補の存在確認
    assert "other_or_unsupported" in mapping.fine_criteria
    assert mapping.fine_to_coarse["other_or_unsupported"] == "exchange_fee"


def test_validation_mece_violation() -> None:
    """細分類の重複割り当て (MECE 違反) を検知して例外を送出することを検証する。"""
    mapping = HierarchicalMapping(
        name="InvalidMapping",
        coarse_categories={"c1": "大分類1", "c2": "大分類2"},
        fine_criteria={"f1": "細分類1", "f2": "細分類2"},
        coarse_to_fine={"c1": ["f1", "f2"], "c2": ["f1"]},  # f1 が重複
        fine_to_coarse={"f1": "c1", "f2": "c1"},
    )
    with pytest.raises(ValueError, match="MECE 違反"):
        mapping.validate()


def test_validation_missing_fine_in_criteria() -> None:
    """定義されていない細分類キーの参照を検知することを検証する。"""
    mapping = HierarchicalMapping(
        name="MissingFine",
        coarse_categories={"c1": "大分類1"},
        fine_criteria={"f1": "細分類1"},
        coarse_to_fine={"c1": ["f1", "f_unknown"]},
        fine_to_coarse={"f1": "c1", "f_unknown": "c1"},
    )
    with pytest.raises(ValueError, match="fine_criteria に存在しません"):
        mapping.validate()


def test_serialization_json(tmp_path: Path) -> None:
    """JSON への保存と復元で完全な同一性が保たれることを検証する。"""
    mapping = HierarchicalMapping.banking77()
    json_path = tmp_path / "banking77.json"
    mapping.to_json(json_path)

    restored = HierarchicalMapping.from_json(json_path)
    assert restored.name == mapping.name
    assert restored.coarse_categories == mapping.coarse_categories
    assert restored.fine_criteria == mapping.fine_criteria
    assert restored.coarse_to_fine == mapping.coarse_to_fine
    assert restored.fine_to_coarse == mapping.fine_to_coarse
    assert restored.all_coarse_keys == mapping.all_coarse_keys
    assert restored.all_fine_keys == mapping.all_fine_keys
