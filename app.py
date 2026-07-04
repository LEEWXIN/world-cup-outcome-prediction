"""
app.py
------
Streamlit dashboard for predicting (and explaining) international football
match outcomes. Visual design implements the "Match Outcome Predictor"
handoff spec (see design_handoff_match_predictor/README.md) as closely as
Streamlit's native widgets allow. Uses the exact same pipeline as the
Jupyter notebook, imported from model.py.

Run locally:   streamlit run app.py
Run in Docker: see Dockerfile / DASHBOARD_GUIDE.md
"""
import streamlit as st
import pandas as pd
import model as M

st.set_page_config(page_title="Match Outcome Predictor", page_icon="⚽", layout="centered")

# ---------------------------------------------------------------------------
# Design tokens — copied verbatim from the design handoff spec (OKLCH)
# ---------------------------------------------------------------------------
PAGE_BG      = "oklch(0.97 0.012 80)"
CARD_BG      = "oklch(0.995 0.006 80)"
BORDER       = "oklch(0.87 0.012 80)"
DIVIDER      = "oklch(0.85 0.012 80)"
TEXT_PRIMARY = "oklch(0.22 0.02 80)"
TEXT_MUTED   = "oklch(0.5 0.02 80)"
TEXT_SECOND  = "oklch(0.3 0.02 80)"
GREEN        = "oklch(0.32 0.09 155)"   # pitch green — primary / home accent
GREEN_LIGHT  = "oklch(0.82 0.05 155)"   # away side of strength bar
AMBER_FILL   = "oklch(0.72 0.14 80)"
AMBER_TEXT   = "oklch(0.48 0.13 80)"
CLAY_FILL    = "oklch(0.58 0.13 35)"
CLAY_TEXT    = "oklch(0.48 0.13 35)"
GRAY_FILL    = "oklch(0.65 0.01 80)"
GRAY_TEXT    = "oklch(0.45 0.01 80)"
TRACK_BG     = "oklch(0.93 0.01 80)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {{ background: {PAGE_BG}; }}
html, body, [class*="css"] {{ color: {TEXT_PRIMARY} !important; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; color: {TEXT_PRIMARY} !important; }}

.block-container {{ max-width: 1080px; padding-top: 40px; padding-bottom: 80px; }}

/* restyle native Streamlit selects + checkbox to match the mono/compact spec */
div[data-baseweb="select"] > div {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 15px !important; font-weight: 500 !important;
    border: 1px solid {BORDER} !important; border-radius: 6px !important;
    background: white !important; color: {TEXT_PRIMARY} !important;
}}
div[data-baseweb="select"] * {{ color: {TEXT_PRIMARY} !important; }}
[data-testid="stCheckbox"] label p {{
    font-family: 'IBM Plex Mono', monospace; font-size: 14px; color: {TEXT_MUTED} !important;
}}

.card {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 26px 32px; margin-bottom: 16px;
}}
.eyebrow {{
    font-family: 'IBM Plex Mono', monospace; font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.14em; color: {TEXT_MUTED}; margin-bottom: 8px;
}}
.grid-2 {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 36px; }}
.divider-left {{ border-left: 1px solid {BORDER}; padding-left: 32px; }}
</style>
""", unsafe_allow_html=True)


# Train once, then cache so the app is instant on every interaction.
@st.cache_resource(show_spinner="Building features and training the model (first load only)...")
def load_model():
    mdl, elo, latest_form, metrics, h2h, history, h2h_record = M.train_pipeline("international_matches1.csv")
    teams = sorted(elo.keys())
    return mdl, elo, latest_form, metrics, h2h, history, h2h_record, teams


mdl, elo, latest_form, metrics, h2h, history, h2h_record, teams = load_model()

# ---------------------------------------------------------------------------
# Top bar — eyebrow + title (left), three stat readouts (right)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap;
            border-bottom:1px solid {DIVIDER}; padding-bottom:18px; margin-bottom:22px;">
  <div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:13px; text-transform:uppercase;
                letter-spacing:0.16em; color:{GREEN}; font-weight:500; margin-bottom:6px;">
      Data Analytics Project
    </div>
    <h1 style="font-size:34px; font-weight:700; margin:0; color:{TEXT_PRIMARY};">Match Outcome Predictor</h1>
  </div>
  <div style="display:flex; gap:20px;">
    <div style="border-left:1px solid {DIVIDER}; padding-left:20px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; text-transform:uppercase; color:{TEXT_MUTED};">Accuracy</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:600; color:{GREEN};">{metrics['accuracy']*100:.1f}%</div>
    </div>
    <div style="border-left:1px solid {DIVIDER}; padding-left:20px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; text-transform:uppercase; color:{TEXT_MUTED};">Baseline</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:600; color:{TEXT_PRIMARY};">{metrics['baseline']*100:.1f}%</div>
    </div>
    <div style="border-left:1px solid {DIVIDER}; padding-left:20px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; text-transform:uppercase; color:{TEXT_MUTED};">N matches</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:600; color:{TEXT_PRIMARY};">{metrics['n_train']:,}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Control row — team pickers + home-venue checkbox
# ---------------------------------------------------------------------------
col_a, col_vs, col_b, col_chk = st.columns([4, 1, 4, 4])
default_h = teams.index("Argentina") if "Argentina" in teams else 0
default_a = teams.index("Brazil") if "Brazil" in teams else 1
home = col_a.selectbox("Home team", teams, index=default_h, label_visibility="collapsed")
col_vs.markdown(f"<div style='font-family:\"IBM Plex Mono\",monospace; font-size:14px; color:{TEXT_MUTED}; text-align:center; padding-top:8px;'>vs</div>", unsafe_allow_html=True)
away = col_b.selectbox("Away team", teams, index=default_a, label_visibility="collapsed")
home_adv = col_chk.checkbox("Home venue", value=True)

if home == away:
    st.warning("Please pick two different teams.")
    st.stop()

readable, label, proba, feat = M.predict_match(mdl, elo, latest_form, home, away, home_adv, h2h=h2h)

# ---------------------------------------------------------------------------
# Main terminal panel — team comparison + prediction (left) | probabilities (right)
# ---------------------------------------------------------------------------
h_elo, a_elo = feat["Home_Elo"], feat["Away_Elo"]
h_share = h_elo / (h_elo + a_elo) * 100
a_share = 100 - h_share

last5_h = M.last5_string(history, home) or "-----"
last5_a = M.last5_string(history, away) or "-----"
tick_color = {"W": GREEN, "D": GRAY_FILL, "L": CLAY_FILL}


def ticks_html(s, align):
    boxes = "".join(
        f"<div style='width:18px;height:18px;border-radius:4px;background:{tick_color.get(c, TRACK_BG)};'></div>"
        for c in s
    )
    justify = "flex-start" if align == "left" else "flex-end"
    return f"<div style='display:flex; gap:6px; justify-content:{justify};'>{boxes}</div>"


if label == "Win":
    outcome_color = AMBER_TEXT
elif label == "Loss":
    outcome_color = CLAY_TEXT
else:
    outcome_color = GRAY_TEXT

p_home, p_draw, p_away = proba.get("Win", 0), proba.get("Draw", 0), proba.get("Loss", 0)

hw, hd, aw, htot = M.h2h_record_for(h2h_record, home, away)
if htot == 0:
    h2h_html = (f"<div style='font-family:\"IBM Plex Mono\",monospace; font-size:13px; "
                f"color:{TEXT_MUTED}; margin-bottom:18px;'>Head-to-head: these teams have not played before.</div>")
else:
    h2h_html = (f"<div style='font-family:\"IBM Plex Mono\",monospace; font-size:13px; "
                f"color:{TEXT_SECOND}; margin-bottom:18px;'>Head-to-head ({htot} matches): "
                f"<span style='color:{TEXT_PRIMARY}; font-weight:600;'>{home} {hw}W</span> — "
                f"{hd}D — <span style='color:{TEXT_PRIMARY}; font-weight:600;'>{away} {aw}W</span></div>")

st.markdown(f"""
<div class="card">
  <div class="grid-2">
    <div>
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <div>
          <div style="font-size:17px; font-weight:600; color:{TEXT_PRIMARY};">{home}</div>
          <div style="font-family:'IBM Plex Mono',monospace; font-size:14px; color:{TEXT_MUTED};">ELO {h_elo:.0f}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:17px; font-weight:600; color:{TEXT_PRIMARY};">{away}</div>
          <div style="font-family:'IBM Plex Mono',monospace; font-size:14px; color:{TEXT_MUTED};">ELO {a_elo:.0f}</div>
        </div>
      </div>
      <div style="display:flex; height:9px; border-radius:4px; overflow:hidden; margin-bottom:16px;">
        <div style="width:{h_share:.1f}%; background:{GREEN};"></div>
        <div style="width:{a_share:.1f}%; background:{GREEN_LIGHT};"></div>
      </div>
      <div style="display:flex; justify-content:space-between; margin-bottom:16px;">
        {ticks_html(last5_h, "left")}
        {ticks_html(last5_a, "right")}
      </div>
      {h2h_html}
      <div style="border-left:3px solid {outcome_color}; padding-left:14px;">
        <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; text-transform:uppercase; color:{TEXT_MUTED}; margin-bottom:4px;">Most likely outcome</div>
        <div style="font-size:26px; font-weight:700; color:{outcome_color};">{readable}</div>
      </div>
    </div>
    <div class="divider-left">
      <div class="eyebrow">Win probability</div>
      <div style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:5px; color:{TEXT_PRIMARY};">
          <span>{home} win</span><span style="font-family:'IBM Plex Mono',monospace; font-weight:600;">{p_home*100:.0f}%</span>
        </div>
        <div style="height:11px; border-radius:5px; background:{TRACK_BG};">
          <div style="width:{p_home*100:.1f}%; height:100%; border-radius:4px; background:{AMBER_FILL};"></div>
        </div>
      </div>
      <div style="margin-bottom:14px;">
        <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:5px; color:{TEXT_PRIMARY};">
          <span>Draw</span><span style="font-family:'IBM Plex Mono',monospace; font-weight:600;">{p_draw*100:.0f}%</span>
        </div>
        <div style="height:11px; border-radius:5px; background:{TRACK_BG};">
          <div style="width:{p_draw*100:.1f}%; height:100%; border-radius:4px; background:{GRAY_FILL};"></div>
        </div>
      </div>
      <div>
        <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:5px; color:{TEXT_PRIMARY};">
          <span>{away} win</span><span style="font-family:'IBM Plex Mono',monospace; font-weight:600;">{p_away*100:.0f}%</span>
        </div>
        <div style="height:11px; border-radius:5px; background:{TRACK_BG};">
          <div style="width:{p_away*100:.1f}%; height:100%; border-radius:4px; background:{CLAY_FILL};"></div>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Why panel — explanation bullets (left) | global feature importance (right)
# ---------------------------------------------------------------------------
elo_gap = feat["Elo_Diff"]
stronger, weaker = (home, away) if elo_gap > 0 else (away, home)
bullet1 = f"{stronger} holds a {abs(elo_gap):.0f}-point Elo edge over {weaker}."
bullet2 = f"Last 5: {home} {last5_h} vs {away} {last5_a}."
bullet3 = (f"{home} gets a home boost, equivalent to roughly +100 Elo points."
           if home_adv else "Neutral venue — no home advantage applied.")

# Real global feature importance from the trained model, grouped into the
# four categories the design calls for (not static placeholder numbers).
imp = M.feature_importance(mdl)
groups = {
    "Elo rating gap": imp[["Elo_Diff", "Home_Elo", "Away_Elo"]].sum(),
    "Recent form (last 5 matches)": imp[["Form_Pts_Diff", "Form_GF_Diff", "Form_GA_Diff"]].sum(),
    "Home advantage": imp[["Home_Adv"]].sum(),
    "Other factors (head-to-head, experience)": imp[["H2H_Diff", "Exp_Diff"]].sum(),
}
total = sum(groups.values())
groups = {k: v / total * 100 for k, v in groups.items()}

bullets_html = "".join(
    f"<div style='display:flex; gap:8px; margin-bottom:10px;'>"
    f"<span style='color:{GREEN}; font-family:\"IBM Plex Mono\",monospace;'>—</span>"
    f"<span style='font-family:\"IBM Plex Mono\",monospace; font-size:15px; line-height:1.6; color:{TEXT_SECOND};'>{b}</span></div>"
    for b in [bullet1, bullet2, bullet3]
)

importance_html = "".join(
    f"<div style='margin-bottom:12px;'>"
    f"<div style='display:flex; justify-content:space-between; font-size:13px; color:{TEXT_SECOND}; margin-bottom:5px;'>"
    f"<span>{name}</span><span style='font-family:\"IBM Plex Mono\",monospace; font-weight:600;'>{pct:.0f}%</span></div>"
    f"<div style='height:8px; border-radius:4px; background:{TRACK_BG};'>"
    f"<div style='width:{pct:.1f}%; height:100%; border-radius:3px; background:{GREEN};'></div></div></div>"
    for name, pct in groups.items()
)

st.markdown(f"""
<div class="card">
  <div class="grid-2">
    <div>
      <div class="eyebrow">Why this prediction</div>
      {bullets_html}
    </div>
    <div class="divider-left">
      <div class="eyebrow">Global feature importance</div>
      {importance_html}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("Show the raw feature values fed to the model"):
    st.dataframe(pd.DataFrame([feat]).T.rename(columns={0: "value"}))

st.markdown(f"""
<div style="text-align:center; margin-top:28px; padding-top:18px; border-top:1px solid {DIVIDER};
            font-family:'IBM Plex Mono',monospace; font-size:12px; color:{TEXT_MUTED};">
  Trained on {metrics['n_train'] + metrics['n_test']:,} international matches, 1872–2022 ·
  Source: Kaggle — International Football Results (martj42)
</div>
""", unsafe_allow_html=True)