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

def html(s: str) -> str:
    """Strip leading whitespace from every line of a multi-line HTML/CSS
    block before handing it to st.markdown. Streamlit's Markdown parser
    treats any line starting with 4+ spaces (including a wrapped style=
    attribute lined up for readability) as a code block and prints it as
    literal text / breaks the tag instead of rendering it. HTML/CSS don't
    care about whitespace, so stripping it per line is always safe."""
    return "\n".join(line.lstrip() for line in s.strip("\n").split("\n"))

# ---------------------------------------------------------------------------
# Design tokens — restyled to match the "Match Outcome Predictor" reference
# mockup (blue accent, IBM Plex Sans + Mono, tighter/compact spacing).
# Functionality below is unchanged — this is a CSS-only pass.
# ---------------------------------------------------------------------------
PAGE_BG      = "oklch(98% 0.004 95)"
CARD_BG      = "#ffffff"
BORDER       = "oklch(90% 0.005 95)"
BORDER_INPUT = "oklch(87% 0.005 95)"
DIVIDER      = "oklch(91% 0.005 95)"
TEXT_PRIMARY = "oklch(22% 0.01 95)"
TEXT_MUTED   = "oklch(55% 0.02 95)"
TEXT_SECOND  = "oklch(30% 0.01 95)"
BLUE         = "oklch(50% 0.1 255)"     # primary accent (replaces the old pitch green)
BLUE_LIGHT   = "oklch(80% 0.05 255)"    # away side of the Elo strength bar
AMBER_FILL   = "oklch(60% 0.13 70)"     # home-win colour
AMBER_TEXT   = "oklch(55% 0.14 70)"
CLAY_FILL    = "oklch(55% 0.15 25)"     # away-win colour
CLAY_TEXT    = "oklch(52% 0.15 25)"
GRAY_FILL    = "oklch(65% 0.01 95)"     # draw colour
GRAY_TEXT    = "oklch(45% 0.01 95)"
TRACK_BG     = "oklch(93% 0.005 95)"
SUCCESS_BG   = "oklch(94% 0.03 155)"
SUCCESS_TEXT = "oklch(32% 0.06 155)"
FORM_WIN     = "oklch(58% 0.13 150)"    # distinct green for "W" form tiles — NOT the theme accent
GREEN        = FORM_WIN                 # kept as an alias so downstream "GREEN" references (form
GREEN_LIGHT  = BLUE_LIGHT                # tiles, bullet markers) stay green, matching the reference mockup

st.markdown(html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {{ background: {PAGE_BG}; }}
html, body, [class*="css"] {{ color: {TEXT_PRIMARY} !important; font-family: 'IBM Plex Sans', sans-serif !important; }}
h1, h2, h3 {{ font-family: 'IBM Plex Sans', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.01em; color: {TEXT_PRIMARY} !important; }}

.block-container {{ max-width: 1080px; padding-top: 32px; padding-bottom: 60px; }}

/* restyle native Streamlit selects + checkbox to match the reference mockup */
div[data-baseweb="select"] > div {{
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13.5px !important; font-weight: 600 !important;
    border: 1px solid {BORDER_INPUT} !important; border-radius: 6px !important;
    background: {PAGE_BG} !important; color: {TEXT_PRIMARY} !important;
}}
div[data-baseweb="select"] * {{ color: {TEXT_PRIMARY} !important; }}
[data-testid="stCheckbox"] label p {{
    font-family: 'IBM Plex Sans', sans-serif; font-size: 12.5px; font-weight: 500; color: {TEXT_SECOND} !important;
}}
input[type="number"], input[type="text"], input[type="date"] {{
    border: 1px solid {BORDER_INPUT} !important; border-radius: 6px !important;
    background: {PAGE_BG} !important; font-family: 'IBM Plex Mono', monospace !important;
}}
.stButton > button, .stFormSubmitButton > button {{
    background: {BLUE} !important; color: #fff !important; border: none !important;
    border-radius: 6px !important; font-weight: 600 !important; font-size: 12.5px !important;
}}
[data-testid="stExpander"] {{
    background: {CARD_BG}; border: 1px solid {BORDER} !important; border-radius: 10px !important;
}}
[data-testid="stExpander"] summary {{
    font-family: 'IBM Plex Sans', sans-serif !important; font-size: 12.5px !important; font-weight: 600 !important;
}}

.card {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 18px 20px; margin-bottom: 14px;
}}
.eyebrow {{
    font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; text-transform: uppercase;
    letter-spacing: 0.05em; color: {TEXT_MUTED}; margin-bottom: 10px;
}}
.grid-2 {{ display: grid; grid-template-columns: 1.6fr 1fr; gap: 14px; }}
.divider-left {{ border-left: none; padding-left: 0; }}
</style>
"""), unsafe_allow_html=True)


# Train once, then cache so the app is instant on every interaction. This
# returns the "base" state trained purely from the CSV.
@st.cache_resource(show_spinner="Building features and training the model (first load only)...")
def load_model():
    mdl, elo, latest_form, metrics, h2h, history, h2h_record, h2h_matches = M.train_pipeline("international_matches1.csv")
    teams = sorted(elo.keys())
    return mdl, elo, latest_form, metrics, h2h, history, h2h_record, h2h_matches, teams


mdl, base_elo, base_latest_form, metrics, base_h2h, base_history, base_h2h_record, base_h2h_matches, teams = load_model()

# The feature state (Elo / form / H2H) needs to be mutable across interactions
# whenever the user logs a new match result, but the base state from
# load_model() is cached and shared across ALL users/sessions — so we must
# not mutate it directly. Instead we keep a per-session working copy in
# st.session_state, seeded once from the base state, and every "Add match
# result" submission updates this copy only. The classifier (`mdl`) is
# intentionally NOT part of this — it stays fixed until a real retrain.
if "elo" not in st.session_state:
    st.session_state.elo = dict(base_elo)
    st.session_state.latest_form = base_latest_form.copy()
    st.session_state.h2h = dict(base_h2h)
    st.session_state.h2h_record = {k: dict(v) for k, v in base_h2h_record.items()}
    st.session_state.h2h_matches = {k: list(v) for k, v in base_h2h_matches.items()}
    st.session_state.history = base_history.copy()
    st.session_state.added_matches = []  # log of matches added this session, for display

elo = st.session_state.elo
latest_form = st.session_state.latest_form
h2h = st.session_state.h2h
h2h_record = st.session_state.h2h_record
h2h_matches = st.session_state.h2h_matches
history = st.session_state.history

# ---------------------------------------------------------------------------
# Top bar — eyebrow + title (left), three stat readouts (right)
# ---------------------------------------------------------------------------
st.markdown(html(f"""
<div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap;
            border-bottom:1px solid {DIVIDER}; padding-bottom:18px; margin-bottom:22px;">
  <div>
    <div style="font-family:'IBM Plex Mono',monospace; font-size:13px; text-transform:uppercase;
                letter-spacing:0.16em; color:{BLUE}; font-weight:500; margin-bottom:6px;">
      Data Analytics Project
    </div>
    <h1 style="font-size:34px; font-weight:700; margin:0; color:{TEXT_PRIMARY};">Match Outcome Predictor</h1>
  </div>
  <div style="display:flex; gap:20px;">
    <div style="border-left:1px solid {DIVIDER}; padding-left:20px;">
      <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; text-transform:uppercase; color:{TEXT_MUTED};">Accuracy</div>
      <div style="font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:600; color:{BLUE};">{metrics['accuracy']*100:.1f}%</div>
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
"""), unsafe_allow_html=True)

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

h_rank, n_teams = M.elo_rank(elo, home)
a_rank, _ = M.elo_rank(elo, away)
h_rank_str = f"ELO {h_elo:.0f} · #{h_rank} of {n_teams}" if h_rank else f"ELO {h_elo:.0f}"
a_rank_str = f"ELO {a_elo:.0f} · #{a_rank} of {n_teams}" if a_rank else f"ELO {a_elo:.0f}"

last5_h = M.last5_string(history, home) or "-----"
last5_a = M.last5_string(history, away) or "-----"
tick_color = {"W": GREEN, "D": GRAY_FILL, "L": CLAY_FILL}


RESULT_LABEL = {"W": "Win", "D": "Draw", "L": "Loss"}


def ticks_html(s, align):
    # Letter + title attribute so the result isn't color-only (accessibility).
    boxes = "".join(
        f"<div title='{RESULT_LABEL.get(c, c)}' style='width:18px;height:18px;border-radius:4px;"
        f"background:{tick_color.get(c, TRACK_BG)}; display:flex; align-items:center; justify-content:center;"
        f"font-family:\"IBM Plex Mono\",monospace; font-size:10px; font-weight:600; color:white;'>{c}</div>"
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

st.markdown(html(f"""
<div class="card">
  <div class="grid-2">
    <div>
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <div>
          <div style="font-size:17px; font-weight:600; color:{TEXT_PRIMARY};">{home}</div>
          <div style="font-family:'IBM Plex Mono',monospace; font-size:14px; color:{TEXT_MUTED};">{h_rank_str}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:17px; font-weight:600; color:{TEXT_PRIMARY};">{away}</div>
          <div style="font-family:'IBM Plex Mono',monospace; font-size:14px; color:{TEXT_MUTED};">{a_rank_str}</div>
        </div>
      </div>
      <div style="display:flex; height:9px; border-radius:4px; overflow:hidden; margin-bottom:16px;">
        <div style="width:{h_share:.1f}%; background:{AMBER_FILL};"></div>
        <div style="width:{a_share:.1f}%; background:{CLAY_FILL};"></div>
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
"""), unsafe_allow_html=True)

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

narrative = M.narrative_sentence(feat, groups, home, away, label)
narrative_html = (
    f"<div style='font-size:16px; line-height:1.6; color:{TEXT_PRIMARY}; "
    f"margin-bottom:16px; padding-bottom:16px; border-bottom:1px solid {DIVIDER};'>{narrative}</div>"
)

bullets_html = "".join(
    f"<div style='display:flex; gap:8px; margin-bottom:10px;'>"
    f"<span style='color:{BLUE}; font-family:\"IBM Plex Mono\",monospace;'>—</span>"
    f"<span style='font-family:\"IBM Plex Mono\",monospace; font-size:15px; line-height:1.6; color:{TEXT_SECOND};'>{b}</span></div>"
    for b in [bullet1, bullet2, bullet3]
)

importance_html = "".join(
    f"<div style='margin-bottom:12px;'>"
    f"<div style='display:flex; justify-content:space-between; font-size:13px; color:{TEXT_SECOND}; margin-bottom:5px;'>"
    f"<span>{name}</span><span style='font-family:\"IBM Plex Mono\",monospace; font-weight:600;'>{pct:.0f}%</span></div>"
    f"<div style='height:8px; border-radius:4px; background:{TRACK_BG};'>"
    f"<div style='width:{pct:.1f}%; height:100%; border-radius:3px; background:{BLUE};'></div></div></div>"
    for name, pct in groups.items()
)

st.markdown(html(f"""
<div class="card">
  <div class="grid-2">
    <div>
      <div class="eyebrow">Why this prediction</div>
      {narrative_html}
      {bullets_html}
    </div>
    <div class="divider-left">
      <div class="eyebrow">Global feature importance</div>
      {importance_html}
    </div>
  </div>
</div>
"""), unsafe_allow_html=True)

with st.expander("Show the raw feature values fed to the model"):
    st.dataframe(pd.DataFrame([feat]).T.rename(columns={0: "value"}))

past_meetings = M.h2h_match_list(h2h_matches, home, away, n=10)
if past_meetings:
    with st.expander(f"Show the last {len(past_meetings)} meetings between {home} and {away}"):
        rows_html = ""
        for m in past_meetings:
            r_color = tick_color.get(m["result_for_home"], TEXT_MUTED)
            rows_html += (
                f"<div style='display:flex; justify-content:space-between; padding:6px 0; "
                f"border-bottom:1px solid {DIVIDER}; font-family:\"IBM Plex Mono\",monospace; font-size:13px;'>"
                f"<span style='color:{TEXT_MUTED};'>{m['date'].strftime('%Y-%m-%d')}</span>"
                f"<span style='color:{TEXT_SECOND};'>played at {m['played_at']}</span>"
                f"<span style='font-weight:600; color:{TEXT_PRIMARY};'>{home} {m['score']} "
                f"<span style='color:{r_color};'>({m['result_for_home']})</span></span>"
                f"</div>"
            )
        st.markdown(rows_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Add a completed match result — incremental Elo/form/H2H update, no full
# retrain. Answers the "can you feed in new match data?" question: the very
# next prediction for these two teams reflects the new result immediately,
# because Elo/form/H2H are the pre-match features the classifier reads, and
# those get refreshed here. The classifier itself keeps its existing
# weights until a real periodic retrain (see report future-work).
# ---------------------------------------------------------------------------
with st.expander("➕ Add a completed match result (update Elo & form live)"):
    st.caption(
        "This updates Elo, recent form, and head-to-head for the two teams "
        "immediately — the same formulas used to build the training data — "
        "without retraining the Random Forest itself. Try it, then re-pick "
        "the same two teams above and watch the numbers move."
    )
    with st.form("add_match_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        new_home = c1.selectbox("Home team", teams, index=teams.index(home), key="new_home")
        new_away = c2.selectbox("Away team", teams, index=teams.index(away), key="new_away")
        new_date = c3.date_input("Date", value=pd.Timestamp.today())
        c4, c5 = st.columns(2)
        new_hg = c4.number_input("Home goals", min_value=0, max_value=30, value=1, step=1)
        new_ag = c5.number_input("Away goals", min_value=0, max_value=30, value=1, step=1)
        submitted = st.form_submit_button("Add result & update model inputs")

    if submitted:
        if new_home == new_away:
            st.error("Home and away team must be different.")
        else:
            (st.session_state.elo, st.session_state.latest_form, st.session_state.h2h,
             st.session_state.h2h_record, st.session_state.h2h_matches, st.session_state.history) = (
                M.update_after_match(
                    st.session_state.elo, st.session_state.latest_form, st.session_state.h2h,
                    st.session_state.h2h_record, st.session_state.h2h_matches, st.session_state.history,
                    home=new_home, away=new_away, home_goals=int(new_hg), away_goals=int(new_ag),
                    date=new_date,
                )
            )
            st.session_state.added_matches.append(
                f"{new_date} — {new_home} {new_hg}-{new_ag} {new_away}"
            )
            st.success(
                f"Added: {new_home} {new_hg}-{new_ag} {new_away}. "
                f"Elo, form and head-to-head updated — pick these two teams above to see it."
            )
            st.rerun()

    if st.session_state.added_matches:
        st.markdown(f"<div style='font-family:\"IBM Plex Mono\",monospace; font-size:12px; "
                    f"color:{TEXT_MUTED}; margin-top:10px;'>Added this session:</div>", unsafe_allow_html=True)
        for m in st.session_state.added_matches:
            st.markdown(f"<div style='font-family:\"IBM Plex Mono\",monospace; font-size:13px; "
                        f"color:{TEXT_SECOND};'>• {m}</div>", unsafe_allow_html=True)
        st.caption(
            "Note: these updates live only in this browser session (Streamlit resets state "
            "on restart). For them to persist and to improve the classifier itself, the new "
            "matches would need to be appended to the CSV and the model periodically retrained."
        )

st.markdown(html(f"""
<div style="text-align:center; margin-top:28px; padding-top:18px; border-top:1px solid {DIVIDER};
            font-family:'IBM Plex Mono',monospace; font-size:12px; color:{TEXT_MUTED};">
  Trained on {metrics['n_train'] + metrics['n_test']:,} international matches, 1872–2022 ·
  Source: Kaggle — International Football Results (martj42)
</div>
"""), unsafe_allow_html=True)