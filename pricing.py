"""Pricing logic for the Streamlit app.

Reads the files saved by the notebook and scores one driver. No model training and no
PyMC here: a prediction is a few dictionary lookups and two exponentials, so it is instant.
"""
import json
import math
from pathlib import Path

import pandas as pd

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


def load_artifacts(folder=ARTIFACT_DIR):
    """Load the model JSON and the result tables saved by the notebook."""
    folder = Path(folder)
    with open(folder / "model_artifacts.json", encoding="utf-8") as fh:
        model = json.load(fh)
    tables = {
        "regions": pd.read_csv(folder / "regional_estimates.csv"),
        "relativity": pd.read_csv(folder / "relativity_table.csv"),
        "comparison": pd.read_csv(folder / "model_comparison.csv"),
        "lorenz": pd.read_csv(folder / "lorenz_curves.csv"),
        "lift": pd.read_csv(folder / "lift_chart.csv"),
    }
    return model, tables


def band(value, rules):
    """Return the label of the first band whose upper bound the value does not exceed."""
    for label, upper in rules:
        if upper is None or value <= upper:
            return label
    return rules[-1][0]


def clean_inputs(model, raw):
    """Clip numeric inputs to the range seen in training and check categorical inputs."""
    out = dict(raw)
    for col, (lo, hi) in model["input_ranges"].items():
        value = float(raw.get(col, lo))
        if math.isnan(value):
            value = lo
        out[col] = min(max(value, lo), hi)
    for col in ("VehBrand", "VehGas"):
        if out.get(col) not in model["levels"][col]:
            out[col] = model["base_levels"][col]
    if out.get("Region") not in model["regions"]:
        out["Region"] = None  # unknown region: no regional adjustment
    return out


def rating_levels(model, inputs):
    """Map a driver's inputs to the level used for each rating factor."""
    levels = {}
    for band_col, spec in model["band_rules"].items():
        levels[band_col] = band(inputs[spec["source"]], spec["rules"])
    for col in ("VehBrand", "VehGas"):
        levels[col] = inputs[col]
    return levels


def region_row(tables, region):
    regions = tables["regions"]
    match = regions.loc[regions["Region"] == region]
    return None if match.empty else match.iloc[0]


def price(model, tables, raw_inputs):
    """Score one driver. Returns frequency, severity, pure premium, and per-factor effects."""
    inputs = clean_inputs(model, raw_inputs)
    levels = rating_levels(model, inputs)
    fq, sv = model["frequency_glm"], model["severity_glm"]

    effects = []  # pure premium relativity of each factor vs the baseline driver
    lin_f, lin_s = fq["intercept"], sv["intercept"]
    for factor in model["factors"]:
        level = levels[factor]
        bf = fq["coefficients"][factor].get(level, 0.0)
        bs = sv["coefficients"][factor].get(level, 0.0)
        lin_f += bf
        lin_s += bs
        effects.append({
            "Factor": model["factor_labels"][factor],
            "Level": level,
            "Baseline level": model["base_levels"][factor],
            "Relativity": math.exp(bf + bs),
        })

    reg = region_row(tables, inputs["Region"]) if inputs["Region"] else None
    region_mult = float(reg["PostMean"]) if reg is not None else 1.0
    effects.append({
        "Factor": "Region",
        "Level": reg["RegionName"] if reg is not None else "Unknown",
        "Baseline level": "National average",
        "Relativity": region_mult,
    })

    frequency = math.exp(lin_f) * region_mult
    severity = math.exp(lin_s)
    pure_premium = frequency * severity
    portfolio = model["portfolio"]
    result = {
        "inputs": inputs,
        "levels": levels,
        "frequency": frequency,
        "severity": severity,
        "pure_premium": pure_premium,
        "risk_ratio": pure_premium / portfolio["pure_premium"],
        "baseline_pure_premium": math.exp(fq["intercept"] + sv["intercept"]),
        "effects": pd.DataFrame(effects),
        "region": None,
    }
    if reg is not None:
        scale = frequency / region_mult  # frequency before the regional multiplier
        result["region"] = {
            "name": reg["RegionName"],
            "multiplier": region_mult,
            "ci_low": float(reg["CI_low"]),
            "ci_high": float(reg["CI_high"]),
            "pp_low": scale * float(reg["CI_low"]) * severity,
            "pp_high": scale * float(reg["CI_high"]) * severity,
            "exposure": float(reg["Exposure"]),
        }
    return result
