"""Kronos price forecast, run as a subprocess inside .venv-kronos (needs PyTorch).

stdin:  {"candles": [{time, open, high, low, close, volume}, ...], "future": ["YYYY-MM-DD", ...], "paths": 8}
stdout: {"dates": [...], "mid": [...], "low": [...], "high": [...]}   (closes: median, 10th, 90th percentile)

With --serve it stays running and loads the model once: one JSON request per stdin line,
one JSON reply per stdout line ({"error": ...} on failure). Saves the slow torch/model start-up per forecast.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "Kronos"))
# on Azure, torch lives in a separate --target dir; append so the app's own packages still win
if os.environ.get("KRONOS_PKGS"):
    sys.path.append(os.environ["KRONOS_PKGS"])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

MODEL = "NeoQuasar/Kronos-small"
TOKENIZER = "NeoQuasar/Kronos-Tokenizer-base"


_predictor = None


def predictor():
    global _predictor
    if _predictor is None:
        _predictor = KronosPredictor(Kronos.from_pretrained(MODEL), KronosTokenizer.from_pretrained(TOKENIZER),
                                     device="cpu", max_context=512)
    return _predictor


def forecast(req: dict) -> dict:
    df = pd.DataFrame(req["candles"]).tail(400)
    x_ts = pd.Series(pd.to_datetime(df["time"]).values)
    y_ts = pd.Series(pd.to_datetime(req["future"]))
    paths = int(req.get("paths", 8))
    torch.manual_seed(0)
    hist = df[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    preds = predictor().predict_batch([hist] * paths, [x_ts] * paths, [y_ts] * paths, pred_len=len(y_ts),
                                    T=1.0, top_p=0.9, sample_count=1, verbose=False)
    closes = np.array([p["close"].values for p in preds])
    return {"dates": req["future"], "model": MODEL, "paths": paths,
            "mid": np.median(closes, 0).round(2).tolist(),
            "low": np.percentile(closes, 10, 0).round(2).tolist(),
            "high": np.percentile(closes, 90, 0).round(2).tolist()}


def main():
    if "--serve" not in sys.argv:
        json.dump(forecast(json.load(sys.stdin)), sys.stdout)
        return
    predictor()
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            out = forecast(json.loads(line))
        except Exception as e:  # noqa: BLE001 - report and keep serving
            out = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
