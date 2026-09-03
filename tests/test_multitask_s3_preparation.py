from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from seismoflux.multitask_s3.preparation import read_issue_cache, sha256, write_issue_cache


def test_prepared_cache_roundtrip_has_no_targets_or_fitted_parameters(tmp_path):
    issue = datetime(2023, 8, 1, tzinfo=UTC)
    identity = {"grid_cells": 2, "grid_id": "synthetic", "source": "fixed"}
    features = np.zeros((2, 20))
    features[0, 0] = np.nan
    log_mass = np.log([0.4, 0.6])
    path = tmp_path / "issue.npz"
    record = write_issue_cache(
        path,
        issue_time=issue,
        identity=identity,
        features=features,
        kernel_log_masses={25.0: log_mass, 75.0: log_mass, 150.0: log_mass},
        r30_log_mass=log_mass,
        expected_counts_per_day={"Ms5_6": 0.05, "Ms6_plus": 0.01, "Ms5_plus": 0.06},
    )
    loaded = read_issue_cache(path, issue_time=issue, identity=identity)
    assert record["sha256"] == sha256(path)
    assert record["bytes"] == path.stat().st_size
    np.testing.assert_equal(loaded["features"], features)
    assert set(loaded) == {
        "features",
        "kernel_25",
        "kernel_75",
        "kernel_150",
        "r30_log_mass",
        "metadata",
    }
    assert loaded["metadata"]["expected_counts_per_day"]["Ms5_plus"] == 0.06
    with pytest.raises(ValueError, match="another actual report"):
        read_issue_cache(path, issue_time=issue + timedelta(days=7), identity=identity)
    with pytest.raises(ValueError, match="different frozen inputs"):
        read_issue_cache(path, issue_time=issue, identity={**identity, "source": "changed"})


def test_wrong_grid_shape_is_not_silently_accepted(tmp_path):
    issue = datetime(2023, 8, 1, tzinfo=UTC)
    path = tmp_path / "wrong.npz"
    log_mass = np.log([0.4, 0.6])
    identity = {"grid_cells": 2}
    write_issue_cache(
        path,
        issue_time=issue,
        identity=identity,
        features=np.zeros((2, 19)),
        kernel_log_masses={25.0: log_mass, 75.0: log_mass, 150.0: log_mass},
        r30_log_mass=log_mass,
        expected_counts_per_day={"Ms5_plus": 0.1},
    )
    with pytest.raises(ValueError, match="20-column"):
        read_issue_cache(path, issue_time=issue, identity=identity)
