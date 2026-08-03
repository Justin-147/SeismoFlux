from __future__ import annotations

from pathlib import Path

from seismoflux.d1_replay.protocol import (
    CONFIG_TO_RUNTIME_FOLD_ID,
    EXPECTED_MODEL_IDS,
    RUNTIME_FOLD_SEED_CODE,
    load_d1_protocol,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_committed_d1_protocol_and_water_level_are_bound() -> None:
    protocol = load_d1_protocol(REPOSITORY_ROOT)

    assert protocol.config_sha256 == (
        "a37112d60798fc267dc9419869a7011ae4e05cb335f4d6b6874fd69f102cf06c"
    )
    assert protocol.water_level_content_sha256 == (
        "2445d940693f6350ef805724ec52e80a8dabfb83c2cd5562ca0bae5281de020a"
    )
    assert tuple(item["id"] for item in protocol.config["models"]) == EXPECTED_MODEL_IDS
    assert protocol.water_level["model_effect_fields_read"] == []
    assert protocol.water_level["global_cluster_catalog"]["cluster_count"] == 23
    assert CONFIG_TO_RUNTIME_FOLD_ID == {
        "F1": "fold_1",
        "F2": "fold_2",
        "F3": "fold_3",
    }
    assert RUNTIME_FOLD_SEED_CODE == {"fold_1": 1, "fold_2": 2, "fold_3": 3}
