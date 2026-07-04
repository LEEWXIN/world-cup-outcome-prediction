"""
model.py
--------
Core logic for the World Cup match-outcome dashboard.

This is the SAME pipeline as the Jupyter notebook, packaged as reusable
functions so the Streamlit app (app.py) can call it. Keeping the logic here
(separate from the UI) means each piece can be tested on its own.

Pipeline:  load CSV -> build Elo + form features -> train Random Forest
           -> predict any (home vs away) matchup and explain why.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# The nine pre-match features the model learns from (same as the notebook)
FEATURES = ["Elo_Diff", "Home_Elo", "Away_Elo", "Form_Pts_Diff", "Form_GF_Diff",
            "Form_GA_Diff", "H2H_Diff", "Exp_Diff", "Home_Adv"]

BASE_ELO, K = 1500, 30


def load_and_prepare(csv_path):
    """Read the matches CSV, sort by date, and create the Win/Draw/Loss target."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(
        subset=["Date", "Home Team", "Away Team", "Home Goals", "Away Goals"]
    ).reset_index(drop=True)
    conditions = [df["Home Goals"] > df["Away Goals"], df["Home Goals"] < df["Away Goals"]]
    df["Result"] = np.select(conditions, ["Win", "Loss"], default="Draw")
    df["match_id"] = np.arange(len(df))
    return df


def add_elo(df):
    """Compute a running Elo rating for every team. Returns df + final ratings dict."""
    elo, home_elo, away_elo = {}, [], []
    for _, r in df.iterrows():
        h, a = r["Home Team"], r["Away Team"]
        eh, ea = elo.get(h, BASE_ELO), elo.get(a, BASE_ELO)
        home_elo.append(eh)
        away_elo.append(ea)                                    # pre-match (no leakage)
        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        if   r["Home Goals"] > r["Away Goals"]: s_h = 1.0
        elif r["Home Goals"] < r["Away Goals"]: s_h = 0.0
        else:                                   s_h = 0.5
        margin = max(np.log(abs(r["Home Goals"] - r["Away Goals"]) + 1), 1)
        elo[h] = eh + K * margin * (s_h - exp_h)
        elo[a] = ea + K * margin * ((1 - s_h) - (1 - exp_h))
    df["Home_Elo"], df["Away_Elo"] = home_elo, away_elo
    df["Elo_Diff"] = df["Home_Elo"] - df["Away_Elo"]
    return df, elo


def add_form_and_h2h(df):
    """Add rolling 5-match form, experience, head-to-head and home advantage.
    Returns df + a 'latest form per team' table + the h2h dict + the long
    (one-row-per-team-per-match) table, which the dashboard needs to draw
    each team's last-5 W/D/L result squares."""
    rows = []
    for _, r in df.iterrows():
        pts_h = 3 if r["Result"] == "Win" else (1 if r["Result"] == "Draw" else 0)
        pts_a = 3 if r["Result"] == "Loss" else (1 if r["Result"] == "Draw" else 0)
        letter_h = "W" if pts_h == 3 else ("D" if pts_h == 1 else "L")
        letter_a = "W" if pts_a == 3 else ("D" if pts_a == 1 else "L")
        rows.append((r["match_id"], r["Date"], r["Home Team"], r["Home Goals"], r["Away Goals"], pts_h, letter_h))
        rows.append((r["match_id"], r["Date"], r["Away Team"], r["Away Goals"], r["Home Goals"], pts_a, letter_a))
    long = pd.DataFrame(
        rows, columns=["match_id", "Date", "Team", "GF", "GA", "Pts", "Result_Letter"]
    ).sort_values("Date").reset_index(drop=True)
    g = long.groupby("Team")
    long["form_pts"] = g["Pts"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    long["form_gf"]  = g["GF"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    long["form_ga"]  = g["GA"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    long["matches_played"] = g.cumcount()

    feat = long.groupby(["match_id", "Team"]).first().reset_index()
    H, A = feat.add_prefix("H_"), feat.add_prefix("A_")
    df = df.merge(H, left_on=["match_id", "Home Team"], right_on=["H_match_id", "H_Team"], how="left")
    df = df.merge(A, left_on=["match_id", "Away Team"], right_on=["A_match_id", "A_Team"], how="left")
    df["Form_Pts_Diff"] = df["H_form_pts"] - df["A_form_pts"]
    df["Form_GF_Diff"]  = df["H_form_gf"]  - df["A_form_gf"]
    df["Form_GA_Diff"]  = df["H_form_ga"]  - df["A_form_ga"]
    df["Exp_Diff"]      = df["H_matches_played"] - df["A_matches_played"]

    # head-to-head differential (home team's historical edge vs this opponent)
    # h2h: (wins_for_key0, total) used as a modelling feature
    # h2h_record: full W/D/L breakdown, keyed the same way, used for display
    h2h, h2h_record, diffs = {}, {}, []
    for _, r in df.iterrows():
        key = tuple(sorted([r["Home Team"], r["Away Team"]]))
        w, t = h2h.get(key, (0, 0))
        if t == 0:
            diffs.append(0.0)
        else:
            rate_first = w / t
            diffs.append((rate_first - 0.5) * 2 if key[0] == r["Home Team"]
                         else ((1 - rate_first) - 0.5) * 2)
        first_win = 1 if ((key[0] == r["Home Team"] and r["Result"] == "Win") or
                          (key[0] == r["Away Team"] and r["Result"] == "Loss")) else 0
        h2h[key] = (w + first_win, t + 1)

        rec = h2h_record.get(key, {"first_wins": 0, "second_wins": 0, "draws": 0, "total": 0})
        if r["Result"] == "Draw":
            rec["draws"] += 1
        elif first_win:
            rec["first_wins"] += 1
        else:
            rec["second_wins"] += 1
        rec["total"] += 1
        h2h_record[key] = rec

    df["H2H_Diff"] = diffs
    df["Home_Adv"] = df["Home Stadium or Not"]

    latest_form = (long.sort_values("Date").groupby("Team")
                   .last()[["form_pts", "form_gf", "form_ga", "matches_played"]])
    return df, latest_form, h2h, long, h2h_record


def train_pipeline(csv_path="international_matches1.csv"):
    """Run the whole pipeline and return everything the dashboard needs."""
    df = load_and_prepare(csv_path)
    df, elo = add_elo(df)
    df, latest_form, h2h, history, h2h_record = add_form_and_h2h(df)

    data = df.dropna(subset=FEATURES + ["Result"]).reset_index(drop=True)
    cut = int(len(data) * 0.8)                       # chronological 80/20 split
    train, test = data.iloc[:cut], data.iloc[cut:]
    model = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   min_samples_leaf=20, random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train["Result"])

    pred = model.predict(test[FEATURES])
    metrics = {"accuracy": accuracy_score(test["Result"], pred),
               "macro_f1": f1_score(test["Result"], pred, average="macro"),
               "baseline": (test["Result"] == "Win").mean(),
               "n_train": len(train), "n_test": len(test)}
    return model, elo, latest_form, metrics, h2h, history, h2h_record


def _form_for(latest_form, team):
    if team in latest_form.index:
        v = latest_form.loc[team]
        return v["form_pts"], v["form_gf"], v["form_ga"], v["matches_played"]
    return 1.0, 1.0, 1.0, 0.0


def _h2h_diff_for(h2h, home, away):
    """Real head-to-head differential for a hypothetical (home, away) matchup,
    using the same convention as training: rate relative to the alphabetically
    first team of the pair, flipped to the home team's perspective."""
    key = tuple(sorted([home, away]))
    w, t = h2h.get(key, (0, 0))
    if t == 0:
        return 0.0
    rate_first = w / t
    return (rate_first - 0.5) * 2 if key[0] == home else ((1 - rate_first) - 0.5) * 2


def h2h_record_for(h2h_record, home, away):
    """Return (home_wins, draws, away_wins, total) for this matchup,
    oriented to home/away regardless of dict key order. All zeros if the
    two teams have never played."""
    key = tuple(sorted([home, away]))
    rec = h2h_record.get(key)
    if rec is None or rec["total"] == 0:
        return 0, 0, 0, 0
    if key[0] == home:
        return rec["first_wins"], rec["draws"], rec["second_wins"], rec["total"]
    else:
        return rec["second_wins"], rec["draws"], rec["first_wins"], rec["total"]


def last5_string(history, team, n=5):
    """Return the team's last n results as a string of W/D/L letters, oldest
    to most recent (left to right), e.g. 'WWDWW'. Empty string if unplayed."""
    if history is None or team not in set(history["Team"]):
        return ""
    rows = history[history["Team"] == team].sort_values("Date").tail(n)
    return "".join(rows["Result_Letter"].tolist())


def predict_match(model, elo, latest_form, home, away, home_advantage=False, h2h=None):
    """Predict a single matchup and return the label, probabilities, and the
    feature values that explain the prediction."""
    eh, ea = elo.get(home, BASE_ELO), elo.get(away, BASE_ELO)
    hp, hgf, hga, hmp = _form_for(latest_form, home)
    ap, agf, aga, amp = _form_for(latest_form, away)
    h2h_diff = _h2h_diff_for(h2h, home, away) if h2h is not None else 0.0
    feat = {
        "Elo_Diff": eh - ea, "Home_Elo": eh, "Away_Elo": ea,
        "Form_Pts_Diff": hp - ap, "Form_GF_Diff": hgf - agf,
        "Form_GA_Diff": hga - aga, "H2H_Diff": h2h_diff,
        "Exp_Diff": hmp - amp, "Home_Adv": 1 if home_advantage else 0,
    }
    X = pd.DataFrame([feat])[FEATURES]
    label = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    readable = {"Win": f"{home} win", "Draw": "Draw", "Loss": f"{away} win"}[label]
    return readable, label, proba, feat


def feature_importance(model):
    return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)


if __name__ == "__main__":
    # quick self-test
    m, elo, lf, metrics, h2h, history, h2h_record = train_pipeline("international_matches1.csv")
    print("Metrics:", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})
    for h, a in [("Brazil", "Qatar"), ("Argentina", "France"), ("Germany", "Japan")]:
        r, lbl, p, f = predict_match(m, elo, lf, h, a, home_advantage=False, h2h=h2h)
        print(f"\n{h} vs {a}: -> {r}")
        print("   probs:", {k: round(v, 2) for k, v in p.items()})
        print("   Elo diff:", round(f["Elo_Diff"], 1), " H2H diff:", round(f["H2H_Diff"], 2))
        print("   last5:", h, last5_string(history, h), "|", a, last5_string(history, a))
        hw, d, aw, tot = h2h_record_for(h2h_record, h, a)
        print(f"   head-to-head: {tot} matches — {h} {hw}W {d}D, {a} {aw}W")
    print("\nTop features:\n", feature_importance(m).round(3).to_string())