import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline

tage = np.arange(100)

daten = pd.DataFrame({
    "id": ["beispiel"] * 100,
    "timestamp": pd.date_range("2026-01-01", periods=100, freq="D"),
    "target": 50 + 0.2 * tage + 10 * np.sin(2 * np.pi * tage / 7),
})

print("Lade Chronos-2 ...")

pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
)


prognose = pipeline.predict_df(
    daten,
    prediction_length=7,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)

print("\nPrognose für die nächsten sieben Tage:")
print(prognose.to_string(index=False))