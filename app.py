"""
app.py
------
Streamlit dashboard for predicting (and explaining) international football
match outcomes. It uses the exact same pipeline as the Jupyter notebook,
imported from model.py.

Run locally:   streamlit run app.py
Run in Docker: see Dockerfile / DASHBOARD_GUIDE.md
"""
import streamlit as st
import pandas as pd
import model as M

st.set_page_config(page_title="World Cup Match Predictor", page_icon="⚽", layout="centered")


# Train once, then cache so the app is instant on every interaction.
@st.cache_resource(show_spinner="Building features and training the model (first load only)...")
def load_model():
    mdl, elo, latest_form, metrics = M.train_pipeline("international_matches1.csv")
    teams = sorted(elo.keys())
    return mdl, elo, latest_form, metrics, teams


mdl, elo, latest_form, metrics, teams = load_model()

st.title("⚽ World Cup Match Outcome Predictor")
st.caption("Predicts Win / Draw / Loss for international matches — and explains *why*. "
           "Same Elo + form model as the analysis notebook.")

# --- model quality banner ---
c1, c2, c3 = st.columns(3)
c1.metric("Test accuracy", f"{metrics['accuracy']*100:.1f}%")
c2.metric("Baseline (guess Win)", f"{metrics['baseline']*100:.1f}%")
c3.metric("Matches trained on", f"{metrics['n_train']:,}")

st.divider()

# --- match picker ---
st.subheader("Pick a matchup")
col_a, col_b = st.columns(2)
default_h = teams.index("Brazil") if "Brazil" in teams else 0
default_a = teams.index("Argentina") if "Argentina" in teams else 1
home = col_a.selectbox("Home team", teams, index=default_h)
away = col_b.selectbox("Away team", teams, index=default_a)
home_adv = st.checkbox("Home team is playing at its own stadium (home advantage)", value=True)

if home == away:
    st.warning("Please pick two different teams.")
    st.stop()

readable, label, proba, feat = M.predict_match(mdl, elo, latest_form, home, away, home_adv)

# --- prediction ---
st.divider()
st.subheader("Prediction")
st.success(f"**Most likely outcome: {readable}**")

prob_df = pd.DataFrame({
    "Outcome": [f"{home} win", "Draw", f"{away} win"],
    "Probability": [proba.get("Win", 0), proba.get("Draw", 0), proba.get("Loss", 0)],
}).set_index("Outcome")
st.bar_chart(prob_df, horizontal=True)

# --- WHY: the explanation panel (the lecturer's tip) ---
st.divider()
st.subheader("Why this prediction?")
elo_gap = feat["Elo_Diff"]
stronger = home if elo_gap > 0 else away
st.markdown(
    f"- **Team strength (Elo):** {home} = `{feat['Home_Elo']:.0f}`, "
    f"{away} = `{feat['Away_Elo']:.0f}`  →  **{stronger} is stronger by {abs(elo_gap):.0f} points.** "
    "This is the single biggest driver of the result.\n"
    f"- **Recent form (last 5 games):** scoring difference `{feat['Form_GF_Diff']:+.2f}` goals, "
    f"conceding difference `{feat['Form_GA_Diff']:+.2f}` goals.\n"
    f"- **Home advantage:** {'applied (+)' if feat['Home_Adv'] else 'not applied (neutral venue)'}."
)

imp = M.feature_importance(mdl).sort_values()
st.markdown("**Across all matches, these factors decide outcomes:**")
st.bar_chart(imp, horizontal=True)
st.caption("Feature importance from the Random Forest. Team-strength gap (Elo) dominates; "
           "defence (goals conceded) matters more than attack; home advantage is real but small.")

with st.expander("Show the raw feature values fed to the model"):
    st.dataframe(pd.DataFrame([feat]).T.rename(columns={0: "value"}))
