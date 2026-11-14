# Auto Insurance Pure Premium Model

Predicting the expected yearly claim cost of a motor insurance policy with frequency and severity GLMs, a Bayesian hierarchical model for regional effects, and a Streamlit pricing app.

**Live app:** https://janvi-insurance-pricing.streamlit.app


## Overview

Pricing teams split a policy's expected cost into two questions: how often will this driver claim, and how much will a claim cost? This project answers both on the public French motor third-party liability dataset (freMTPL2, about 678,000 policies), combines them into a pure premium, and checks how well the model ranks risk on held-out data. A Streamlit app turns the model into a calculator that shows which factors raise or lower a driver's premium.

## Data

- **freMTPL2freq:** one row per policy with exposure (fraction of a year insured), claim count, and rating factors (driver age, vehicle age, power, brand, fuel, bonus-malus, population density, region).
- **freMTPL2sev:** one row per claim with its cost, summed per policy and joined on `IDpol`.
- Loaded from OpenML (IDs 41214 and 41215).

**Cleaning rules:** claims with no recorded cost were set to zero claims, costs with no recorded claim were dropped, exposure was capped at 1 year, claim counts at 4, and claim cost per policy at €200,000 (large losses are normally priced with a separate load).

## Methods

1. **Rating factors:** driver age, vehicle age, vehicle power, bonus-malus, and population density were binned into rating bands; brand and fuel type stay as categories. The baseline level of each factor is its most common level.
2. **Frequency GLM:** Poisson regression with a log link, fitted on claims per year with exposure as the weight (equivalent to a log-exposure offset).
3. **Severity GLM:** Gamma regression with a log link on cost per claim, weighted by claim count, fitted on policies with claims.
4. **Bayesian regional adjustment:** a hierarchical Poisson model in PyMC on region-level totals: `Claims_r ~ Poisson(Expected_r × m_r)`, `log m_r ~ Normal(0, σ)`. Expected claims come from the frequency GLM, so the regional multiplier captures only what the other factors miss, and regions with little data are shrunk toward the national average.
5. **Benchmarks:** a single Tweedie GLM on pure premium (power 1.9) and a Poisson gradient boosting model for frequency.
6. **Validation:** 80/20 train/test split (seed 42), Gini coefficient from the exposure-weighted Lorenz curve, Poisson deviance, predicted vs actual total cost, and a decile lift chart.

## Results

[FILL IN: paste the table from `artifacts/results_summary.md`]

| Model | Gini | Freq Poisson deviance | Predicted ÷ actual cost |
|---|---|---|---|
| Freq GLM × Sev GLM + Bayesian region (final) | | | |
| Freq GLM × Sev GLM (no region) | | | |
| Gradient boosting freq × Sev GLM | | | |
| Tweedie GLM (single model) | | n/a | |

![Lorenz curve](images/lorenz_curve.png)
![Lift chart](images/lift_chart.png)
![Regional estimates](images/regional_estimates.png)

## Business summary

- **Top risk drivers:** [FILL IN: the three drivers and their relativities from the notebook]
- **Recommended model:** [FILL IN after seeing the results. If gradient boosting beats the GLM by only a small Gini margin, recommend the GLM: its relativities can be filed with a regulator, explained to agents, and checked for fairness. If the gap is large, use the GBM to find missing interactions and add them to the GLM.]
- **Limitations:** French data from one insurer, so relativities won't transfer directly to the US market; no inflation or claim trend adjustment; no external data (credit, telematics, weather); large claims capped rather than modeled; regional effects estimated on frequency only.
- **How I'd test it in a pilot:** score a sample of new business with the new rates alongside current rates (shadow pricing), compare loss ratios by decile after 6–12 months, and run a champion–challenger split on a small share of quotes before a full rollout.

## Repository structure

```
├── notebooks/auto_insurance_pure_premium.ipynb   # full analysis (Google Colab)
├── app.py                                         # Streamlit app
├── pricing.py                                     # scoring logic used by the app
├── artifacts/                                     # model files saved by the notebook
├── images/                                        # plots and app screenshot for this README
├── .streamlit/config.toml                         # app theme
└── requirements.txt
```

## How to run

**Notebook:** open `notebooks/auto_insurance_pure_premium.ipynb` in Google Colab and choose Runtime → Run all (about 5–10 minutes on a CPU). It downloads the data, trains all models, and downloads `artifacts.zip`.

**App locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads only the small files in `artifacts/`: no training or sampling happens at startup.

_Disclaimer: this project uses public French motor insurance data and is for demonstration only. It is not a real price quote._
