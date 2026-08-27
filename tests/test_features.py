from __future__ import annotations

import numpy as np
import pandas as pd

from build_features_windows import build_daily_features


def test_log1p_volume_keeps_zero_volume_row() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
            "ticker": ["AMD", "AMD", "AMD"],
            "Open": [10.0, 10.0, 11.0],
            "High": [11.0, 11.0, 12.0],
            "Low": [9.0, 9.0, 10.0],
            "Close": [10.0, 11.0, 12.0],
            "Volume": [100, 0, 9],
        }
    )
    features = build_daily_features(frame, "donor_train")
    assert len(features) == 2
    assert features["date"].min() == pd.Timestamp("2022-01-04")
    assert features.loc[features["date"].eq(pd.Timestamp("2022-01-04")), "log1p_volume"].iloc[0] == 0.0
    assert np.isclose(
        features.loc[features["date"].eq(pd.Timestamp("2022-01-05")), "log1p_volume"].iloc[0],
        np.log(10.0),
    )

