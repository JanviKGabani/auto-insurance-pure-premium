"""Auto insurance pure premium calculator.

Run locally:   streamlit run app.py
Needs the artifacts/ folder produced by the Colab notebook, next to this file.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pricing import load_artifacts, price

TEAL, AMBER, RED, GREEN, GREY = "#2B6F77", "#E0A458", "#B5473A", "#3C8D5A", "#8A8F98"

st.set_page_config(page_title="Pure premium calculator", page_icon="🚗", layout="wide")


@st.cache_data
def get_artifacts():
    return load_artifacts()


try:
    MODEL, T = get_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Put the artifacts/ folder from the notebook next to app.py, then reload.")
    st.stop()

RANGES = MODEL["input_ranges"]
REGION_LABELS = {r: f"{MODEL['region_names'].get(r, r)} ({r})" for r in MODEL["regions"]}


def euro(x, decimals=0):
    return f"€{x:,.{decimals}f}"


# ---------------------------------------------------------------- sidebar form
with st.sidebar.form("driver"):
    st.header("Driver and vehicle")
    drv_age = st.number_input("Driver age (years)", int(max(18, RANGES["DrivAge"][0])),
                              int(RANGES["DrivAge"][1]), 35, step=1)
    bonus_malus = st.number_input(
        "Bonus-malus", int(RANGES["BonusMalus"][0]), int(RANGES["BonusMalus"][1]), 50, step=1,
        help="French no-claims scale. 50 is the best possible score and 100 is a new driver's "
             "starting point; claims push it above 100.")
    veh_age = st.number_input("Vehicle age (years)", int(RANGES["VehAge"][0]), int(RANGES["VehAge"][1]), 5, step=1)
    veh_power = st.number_input("Vehicle power (French fiscal class)", int(RANGES["VehPower"][0]),
                                int(RANGES["VehPower"][1]), 6, step=1)
    brands = MODEL["levels"]["VehBrand"]
    veh_brand = st.selectbox("Vehicle brand (anonymised code)", brands,
                             index=brands.index(MODEL["base_levels"]["VehBrand"]))
    fuels = MODEL["levels"]["VehGas"]
    veh_gas = st.selectbox("Fuel type", fuels, index=fuels.index(MODEL["base_levels"]["VehGas"]))
    region = st.selectbox("Region", MODEL["regions"], format_func=REGION_LABELS.get,
                          index=MODEL["regions"].index("R11") if "R11" in MODEL["regions"] else 0)
    density = st.number_input("Population density (people per km²)", int(RANGES["Density"][0]),
                              int(RANGES["Density"][1]), 1000, step=100)
    st.form_submit_button("Calculate premium", type="primary", use_container_width=True)

inputs = {"DrivAge": drv_age, "BonusMalus": bonus_malus, "VehAge": veh_age, "VehPower": veh_power,
          "VehBrand": veh_brand, "VehGas": veh_gas, "Region": region, "Density": density}
res = price(MODEL, T, inputs)

st.title("Auto insurance pure premium calculator")
st.caption("Expected claim cost per policy-year for a French motor liability policy, "
           "before expenses, commissions and profit.")

tab_calc, tab_insights = st.tabs(["Premium Calculator", "Model Insights"])

# ---------------------------------------------------------------- tab 1
with tab_calc:
    c1, c2, c3 = st.columns(3)
    c1.metric("Claim frequency", f"{res['frequency']:.3f} per year",
              help="Expected number of claims in one year of cover.")
    c2.metric("Average claim severity", euro(res["severity"]), help="Expected cost of one claim.")
    c3.metric("Pure premium", f"{euro(res['pure_premium'])} per year",
              help="Claim frequency × claim severity.")

    ratio = res["risk_ratio"]
    word = "above" if ratio >= 1 else "below"
    st.markdown(
        f"This driver is **{ratio:.2f}× the average risk** "
        f"(portfolio average pure premium {euro(MODEL['portfolio']['pure_premium'])} per year), "
        f"{abs(ratio - 1):.0%} {word} average.")

    st.subheader("What moves this driver's premium")
    eff = res["effects"].copy()
    eff["Change"] = (eff["Relativity"] - 1) * 100
    eff = eff.iloc[eff["Change"].abs().argsort()]  # biggest effects at the top of the chart
    fig = go.Figure(go.Bar(
        x=eff["Change"], y=eff["Factor"], orientation="h",
        marker_color=[RED if v > 0 else GREEN for v in eff["Change"]],
        text=[f"{v:+.0f}%" for v in eff["Change"]], textposition="outside",
        customdata=eff[["Level", "Baseline level"]],
        hovertemplate="%{y}: %{customdata[0]} (baseline %{customdata[1]})<br>%{x:+.1f}%<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis_title="Change in pure premium vs the baseline driver (%)")
    fig.add_vline(x=0, line_color=GREY)
    st.plotly_chart(fig, use_container_width=True)
    base_desc = ", ".join(f"{MODEL['factor_labels'][f].lower()} {MODEL['base_levels'][f]}"
                          for f in MODEL["factors"])
    st.caption(f"Baseline driver ({euro(res['baseline_pure_premium'])} per year before the regional "
               f"adjustment): the most common level of each factor, {base_desc}.")

    if res["region"]:
        r = res["region"]
        st.subheader(f"Regional estimate: {r['name']}")
        st.markdown(
            f"Claims in this region run at **{r['multiplier']:.2f}×** what the other factors predict, "
            f"with a 95% credible interval of **{r['ci_low']:.2f} to {r['ci_high']:.2f}**. "
            f"For this driver, that puts the pure premium between **{euro(r['pp_low'])} and "
            f"{euro(r['pp_high'])}** per year. Regions with less data have wider intervals and are pulled "
            f"closer to 1.00.")

    with st.expander("How this works"):
        st.markdown(f"""
- **Two questions, two models.** One model predicts *how often* a driver claims (a Poisson model of
  claim counts per year of cover). A second predicts *how much* a claim costs (a Gamma model of claim
  amounts). The pure premium is the two multiplied together.
- **Rating factors multiply.** Each factor, such as driver age band or bonus-malus band, scales the
  baseline driver's premium up or down. The bar chart shows those multipliers as percentages.
- **Regions use a Bayesian model.** A small region's own claim record is noisy, so its estimate is
  blended with the national picture. The credible interval shows how sure the model is.
  Method: {MODEL['region_method']}.
- **Checked on unseen data.** The model was trained on 80% of
  {MODEL['portfolio']['n_policies_total']:,} policies and tested on the other 20% (see Model Insights).
""")
    st.info("Demonstration only. This model uses public French motor insurance data (freMTPL2) and is not a "
            "real price quote. It doesn't include expenses, profit, inflation or trend, large-loss loads, "
            "or any data beyond the inputs shown.")

# ---------------------------------------------------------------- tab 2
with tab_insights:
    st.subheader("Model comparison (test set)")
    comp = T["comparison"].copy()
    st.dataframe(comp.style.format({"Gini": "{:.3f}", "Freq Poisson deviance": "{:.4f}",
                                    "Predicted ÷ actual cost": "{:.3f}"}, na_rep="n/a"),
                 hide_index=True, use_container_width=True)
    st.caption("Gini: how well the model ranks cheap vs expensive risks (higher is better). "
               "Poisson deviance: frequency error (lower is better). Predicted ÷ actual: overall price "
               "level (1.00 is perfect).")

    left, right = st.columns(2)
    with left:
        st.subheader("Lorenz curve")
        lz = T["lorenz"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lz["CumExposure"], y=lz["Random"], name="Random (Gini 0)",
                                 line=dict(color=GREY, dash="dash")))
        palette = [TEAL, AMBER, RED, GREEN]
        for i, model_name in enumerate(c for c in lz.columns if c not in ("CumExposure", "Random")):
            g = comp.loc[comp["Model"] == model_name, "Gini"]
            label = f"{model_name} (Gini {g.iloc[0]:.3f})" if len(g) else model_name
            fig.add_trace(go.Scatter(x=lz["CumExposure"], y=lz[model_name], name=label,
                                     line=dict(color=palette[i % len(palette)], width=2.5 if i == 0 else 1.5)))
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Share of exposure (lowest predicted risk first)",
                          yaxis_title="Share of claim cost",
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Decile lift chart (final model)")
        lift = T["lift"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lift["Decile"], y=lift["PredictedPurePremium"], name="Predicted", marker_color=TEAL))
        fig.add_trace(go.Bar(x=lift["Decile"], y=lift["ActualPurePremium"], name="Actual", marker_color=AMBER))
        fig.update_layout(barmode="group", height=430, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis=dict(title="Decile of predicted pure premium (1 = lowest risk)", dtick=1),
                          yaxis_title="€ per policy-year", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Relativity table")
    rel = T["relativity"]
    factor = st.selectbox("Factor", rel["Factor"].unique())
    st.dataframe(
        rel.loc[rel["Factor"] == factor, ["Level", "Baseline", "ExposureShare", "FrequencyRelativity",
                                          "SeverityRelativity", "PurePremiumRelativity"]]
        .style.format({"ExposureShare": "{:.1%}", "FrequencyRelativity": "{:.3f}",
                       "SeverityRelativity": "{:.3f}", "PurePremiumRelativity": "{:.3f}"}),
        hide_index=True, use_container_width=True)
    st.caption("1.000 is the baseline level. 1.250 means 25% more than the baseline. "
               "Regions show the Bayesian frequency multiplier (severity isn't adjusted by region).")
