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

# Match-importance one-hot columns (Friendly is the dropped/baseline category
# - same convention as the notebook's pd.get_dummies(..., drop_first=True)).
MATCH_TYPE_COLS = ["MatchType_Other Competition", "MatchType_Qualifier", "MatchType_World Cup"]

# The pre-match features the model learns from - same set as the notebook,
# including the two features added later (rest days, match importance) so
# the dashboard and the notebook stay in sync.
FEATURES = ["Elo_Diff", "Home_Elo", "Away_Elo", "Form_Pts_Diff", "Form_GF_Diff",
            "Form_GA_Diff", "H2H_Diff", "Exp_Diff", "Home_Adv",
            "Rest_Days_Diff"] + MATCH_TYPE_COLS

BASE_ELO, K = 1500, 30


def bucket_tournament(t):
    """Collapse the ~200 raw Tournament names into 4 buckets - same rule as
    the notebook's Section 5.5, so 'Match_Type' means the same thing in both
    places."""
    if t == "Friendly":
        return "Friendly"
    if "qualification" in str(t).lower():
        return "Qualifier"
    if t == "FIFA World Cup":
        return "World Cup"
    return "Other Competition"


def add_match_type(df):
    """One-hot encode match importance (Section 5.5 in the notebook).
    Returns df with the MATCH_TYPE_COLS columns added."""
    df["Match_Type"] = df["Tournament"].apply(bucket_tournament)
    dummies = pd.get_dummies(df["Match_Type"], prefix="MatchType", drop_first=True).astype(int)
    missing = [c for c in MATCH_TYPE_COLS if c not in dummies.columns]
    for c in missing:
        dummies[c] = 0  # category not present in this slice of data - keep the column, all zeros
    df = pd.concat([df, dummies[MATCH_TYPE_COLS]], axis=1)
    return df


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
    """Compute a running Elo rating for every team. Returns df + final ratings dict.

    Elo is inherently sequential (each match's update depends on the state
    left by all earlier matches), so this can't be fully vectorised away -
    but iterating over plain numpy arrays (via zip) instead of df.iterrows()
    avoids constructing a pandas Series for every one of the ~49,500 rows,
    which is where iterrows() loses most of its time. Same math, same
    results, much less overhead - this is the main fix for the slow cold
    start (first page load / first Docker container run) that was mistaken
    for a UI bug: the app looked "stuck" because this loop hadn't finished.
    """
    elo, home_elo, away_elo = {}, [], []
    homes = df["Home Team"].to_numpy()
    aways = df["Away Team"].to_numpy()
    home_goals_arr = df["Home Goals"].to_numpy()
    away_goals_arr = df["Away Goals"].to_numpy()
    for h, a, home_goals, away_goals in zip(homes, aways, home_goals_arr, away_goals_arr):
        eh, ea = elo.get(h, BASE_ELO), elo.get(a, BASE_ELO)
        home_elo.append(eh)
        away_elo.append(ea)                                    # pre-match (no leakage)
        exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
        if   home_goals > away_goals: s_h = 1.0
        elif home_goals < away_goals: s_h = 0.0
        else:                         s_h = 0.5
        margin = max(np.log(abs(home_goals - away_goals) + 1), 1)
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
    # Build the "long" (one row per team per match) table with plain pandas
    # ops instead of a Python-level loop appending 2 x len(df) tuples one at
    # a time - same output, no per-row overhead.
    pts_h = np.select([df["Result"] == "Win", df["Result"] == "Draw"], [3, 1], default=0)
    pts_a = np.select([df["Result"] == "Loss", df["Result"] == "Draw"], [3, 1], default=0)
    letter_h = np.select([pts_h == 3, pts_h == 1], ["W", "D"], default="L")
    letter_a = np.select([pts_a == 3, pts_a == 1], ["W", "D"], default="L")

    # "_order" reproduces the exact tie-break order the old row-by-row loop
    # produced (home-row then away-row for each match, in df's original
    # chronological order): df is already sorted by Date, so on matches
    # that share the same calendar date, sort_values("Date") below is a
    # stable sort and needs a same-valued secondary key to land in the same
    # home-then-away, match-by-match order as before - otherwise concat()
    # would group all home rows before all away rows on tied dates instead
    # of interleaving them, which very slightly shifts the rolling-5 form
    # window for the handful of teams that ever played twice on one date.
    home_rows = pd.DataFrame({
        "match_id": df["match_id"], "Date": df["Date"], "Team": df["Home Team"],
        "GF": df["Home Goals"], "GA": df["Away Goals"], "Pts": pts_h, "Result_Letter": letter_h,
        "_order": np.arange(len(df)) * 2,
    })
    away_rows = pd.DataFrame({
        "match_id": df["match_id"], "Date": df["Date"], "Team": df["Away Team"],
        "GF": df["Away Goals"], "GA": df["Home Goals"], "Pts": pts_a, "Result_Letter": letter_a,
        "_order": np.arange(len(df)) * 2 + 1,
    })
    long = (pd.concat([home_rows, away_rows], ignore_index=True)
            .sort_values(["Date", "_order"])
            .drop(columns="_order")
            .reset_index(drop=True))
    g = long.groupby("Team")
    long["form_pts"] = g["Pts"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    long["form_gf"]  = g["GF"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    long["form_ga"]  = g["GA"].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    long["matches_played"] = g.cumcount()
    # Rest days since each team's own previous match - same rule as the
    # notebook's Section 5.3 (capped at 60: beyond ~2 months it reflects
    # inactivity between tournaments, not short-term fatigue).
    long["rest_days"] = g["Date"].diff().dt.days
    long["rest_days"] = long["rest_days"].clip(upper=60)

    feat = long.groupby(["match_id", "Team"]).first().reset_index()
    H, A = feat.add_prefix("H_"), feat.add_prefix("A_")
    df = df.merge(H, left_on=["match_id", "Home Team"], right_on=["H_match_id", "H_Team"], how="left")
    df = df.merge(A, left_on=["match_id", "Away Team"], right_on=["A_match_id", "A_Team"], how="left")
    df["Form_Pts_Diff"] = df["H_form_pts"] - df["A_form_pts"]
    df["Form_GF_Diff"]  = df["H_form_gf"]  - df["A_form_gf"]
    df["Form_GA_Diff"]  = df["H_form_ga"]  - df["A_form_ga"]
    df["Exp_Diff"]      = df["H_matches_played"] - df["A_matches_played"]
    df["Rest_Days_Diff"] = df["H_rest_days"] - df["A_rest_days"]

    # head-to-head differential (home team's historical edge vs this opponent)
    # h2h: (wins_for_key0, total) used as a modelling feature
    # h2h_record: full W/D/L breakdown, keyed the same way, used for display
    # h2h_matches: full list of individual past matches per pair, for a
    # "show me the actual games" expander in the dashboard
    # Head-to-head bookkeeping is stateful (each match's diff depends on every
    # earlier meeting of the same pair), so - like Elo - it can't be fully
    # vectorised, but looping over numpy arrays (instead of df.iterrows(),
    # which builds a pandas Series per row) removes the same per-row
    # overhead that was slowing down the Elo loop.
    h2h, h2h_record, h2h_matches, diffs = {}, {}, {}, []
    home_teams = df["Home Team"].to_numpy()
    away_teams = df["Away Team"].to_numpy()
    results = df["Result"].to_numpy()
    dates = df["Date"].to_numpy()
    home_goals_arr = df["Home Goals"].to_numpy()
    away_goals_arr = df["Away Goals"].to_numpy()

    for home_t, away_t, result, date, hg, ag in zip(
        home_teams, away_teams, results, dates, home_goals_arr, away_goals_arr
    ):
        key = tuple(sorted([home_t, away_t]))
        w, t = h2h.get(key, (0, 0))
        if t == 0:
            diffs.append(0.0)
        else:
            rate_first = w / t
            diffs.append((rate_first - 0.5) * 2 if key[0] == home_t
                         else ((1 - rate_first) - 0.5) * 2)
        first_win = 1 if ((key[0] == home_t and result == "Win") or
                          (key[0] == away_t and result == "Loss")) else 0
        h2h[key] = (w + first_win, t + 1)

        rec = h2h_record.get(key, {"first_wins": 0, "second_wins": 0, "draws": 0, "total": 0})
        if result == "Draw":
            rec["draws"] += 1
        elif first_win:
            rec["first_wins"] += 1
        else:
            rec["second_wins"] += 1
        rec["total"] += 1
        h2h_record[key] = rec

        h2h_matches.setdefault(key, []).append({
            "date": pd.Timestamp(date), "home": home_t, "away": away_t,
            "home_goals": int(hg), "away_goals": int(ag),
        })

    df["H2H_Diff"] = diffs
    df["Home_Adv"] = df["Home Stadium or Not"]

    latest_form = (long.sort_values("Date").groupby("Team")
                   .last()[["form_pts", "form_gf", "form_ga", "matches_played"]])
    return df, latest_form, h2h, long, h2h_record, h2h_matches


def train_pipeline(csv_path="international_matches1.csv"):
    """Run the whole pipeline and return everything the dashboard needs."""
    df = load_and_prepare(csv_path)
    df, elo = add_elo(df)
    df, latest_form, h2h, history, h2h_record, h2h_matches = add_form_and_h2h(df)
    df = add_match_type(df)

    data = df.dropna(subset=FEATURES + ["Result"]).reset_index(drop=True)
    cut = int(len(data) * 0.8)                       # chronological 80/20 split
    train, test = data.iloc[:cut], data.iloc[cut:]
    # n_jobs=2 instead of -1: capped thread count avoids CPU-quota contention
    # inside a resource-limited Docker container (a plausible cause of the
    # occasional hang/disconnect when clicking Predict), while still keeping
    # the RandomForest fast on the un-containerised / local-run case.
    model = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   min_samples_leaf=20, random_state=42, n_jobs=2)
    model.fit(train[FEATURES], train["Result"])

    pred = model.predict(test[FEATURES])
    metrics = {"accuracy": accuracy_score(test["Result"], pred),
               "macro_f1": f1_score(test["Result"], pred, average="macro"),
               "baseline": (test["Result"] == "Win").mean(),
               "n_train": len(train), "n_test": len(test)}
    return model, elo, latest_form, metrics, h2h, history, h2h_record, h2h_matches


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


def predict_match(model, elo, latest_form, home, away, home_advantage=False, h2h=None,
                  match_type="World Cup"):
    """Predict a single matchup and return the label, probabilities, and the
    feature values that explain the prediction.

    match_type: which MATCH_TYPE_COLS bucket to assume for this hypothetical
    fixture ("Friendly", "Qualifier", "Other Competition", or "World Cup").
    Rest days aren't knowable for a hypothetical/future fixture, so - same as
    the notebook's feats_for() - we assume no rest advantage either way
    rather than guess."""
    eh, ea = elo.get(home, BASE_ELO), elo.get(away, BASE_ELO)
    hp, hgf, hga, hmp = _form_for(latest_form, home)
    ap, agf, aga, amp = _form_for(latest_form, away)
    h2h_diff = _h2h_diff_for(h2h, home, away) if h2h is not None else 0.0
    feat = {
        "Elo_Diff": eh - ea, "Home_Elo": eh, "Away_Elo": ea,
        "Form_Pts_Diff": hp - ap, "Form_GF_Diff": hgf - agf,
        "Form_GA_Diff": hga - aga, "H2H_Diff": h2h_diff,
        "Exp_Diff": hmp - amp, "Home_Adv": 1 if home_advantage else 0,
        "Rest_Days_Diff": 0,
    }
    for col in MATCH_TYPE_COLS:
        feat[col] = 1 if col == f"MatchType_{match_type}" else 0
    X = pd.DataFrame([feat])[FEATURES]
    label = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    readable = {"Win": f"{home} win", "Draw": "Draw", "Loss": f"{away} win"}[label]
    return readable, label, proba, feat


def elo_rank(elo, team):
    """Return (rank, total_teams) for this team's current Elo, rank 1 = strongest."""
    if team not in elo:
        return None, len(elo)
    ranked = sorted(elo.items(), key=lambda kv: -kv[1])
    for i, (t, _) in enumerate(ranked, start=1):
        if t == team:
            return i, len(ranked)
    return None, len(ranked)


def h2h_match_list(h2h_matches, home, away, n=5):
    """Return the last n individual past meetings between these two teams,
    most recent first, oriented to (home, away) regardless of which side
    was 'home' historically."""
    key = tuple(sorted([home, away]))
    matches = h2h_matches.get(key, [])
    matches = sorted(matches, key=lambda m: m["date"], reverse=True)[:n]
    out = []
    for m in matches:
        if m["home"] == home:
            gh, ga = m["home_goals"], m["away_goals"]
        else:
            gh, ga = m["away_goals"], m["home_goals"]
        result_for_home = "W" if gh > ga else ("L" if gh < ga else "D")
        out.append({"date": m["date"], "score": f"{gh}-{ga}",
                    "played_at": m["home"], "result_for_home": result_for_home})
    return out


def feature_importance(model):
    return pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)


def narrative_sentence(feat, groups, home, away, label):
    """Turn the raw feature values + grouped importance % into a 1-2 sentence
    plain-English explanation for THIS specific matchup, anchored to the
    model's actual predicted label (not to any single feature's direction,
    since the Random Forest combines features non-linearly)."""
    predicted = {"Win": home, "Loss": away, "Draw": None}[label]
    order = sorted(groups.items(), key=lambda kv: -kv[1])
    reasons = []

    for name, pct in order:
        if name == "Elo rating gap":
            d = feat["Elo_Diff"]
            if abs(d) < 1:
                continue
            leader, trailer = (home, away) if d > 0 else (away, home)
            reasons.append({"pct": pct, "leader": leader,
                             "text": f"{leader}'s Elo rating is {abs(d):.0f} points higher than {trailer}'s"})

        elif name == "Recent form (last 5 matches)":
            d = feat["Form_Pts_Diff"]
            if abs(d) < 0.05:
                continue
            leader, trailer = (home, away) if d > 0 else (away, home)
            reasons.append({"pct": pct, "leader": leader,
                             "text": f"{leader} has been in better recent form, averaging "
                                     f"{abs(d):.1f} more points per match than {trailer} over their last 5 games"})

        elif name == "Home advantage":
            if feat["Home_Adv"] != 1:
                continue
            reasons.append({"pct": pct, "leader": home, "text": f"{home} gets a home-venue boost"})

        elif name == "Fixture context (rest days + match importance)":
            rd = feat.get("Rest_Days_Diff", 0)
            if abs(rd) >= 3:
                leader, trailer = (home, away) if rd > 0 else (away, home)
                reasons.append({"pct": pct, "leader": leader,
                                 "text": f"{leader} has had {abs(rd):.0f} more rest days than {trailer} "
                                         f"before this match"})
            else:
                continue

        elif name == "Other factors (head-to-head, experience)":
            hd = feat["H2H_Diff"]
            if abs(hd) < 0.05:
                continue
            leader, trailer = (home, away) if hd > 0 else (away, home)
            reasons.append({"pct": pct, "leader": leader,
                             "text": f"{leader} has the head-to-head edge over {trailer} in past meetings"})

    if predicted is None:
        if not reasons:
            return "This matchup is very even across the board, which is why the model predicts a draw."
        top = reasons[0]
        return (f"The model predicts a draw because no single factor is decisive here — even the "
                f"biggest one, {top['text']}, only accounts for about {top['pct']:.0f}% of the "
                f"decision, and the rest roughly cancels out.")

    if not reasons:
        return (f"This matchup is genuinely close on paper, but the model still narrowly favours "
                f"{predicted} once all the pre-match signals are combined.")

    supporting = [r for r in reasons if r["leader"] == predicted]
    opposing = [r for r in reasons if r["leader"] != predicted]
    support_total = sum(r["pct"] for r in supporting)
    oppose_total = sum(r["pct"] for r in opposing)

    if supporting and support_total >= oppose_total:
        top = supporting[0]
        sentence = f"{top['text']} ({top['pct']:.0f}% of the decision)."
        if len(supporting) > 1:
            second = supporting[1]
            sentence += f" It also helps that {second['text']} ({second['pct']:.0f}%)."
        if opposing:
            worst = opposing[0]
            sentence += (f" This is despite the fact that {worst['text']}, which points the "
                         f"other way ({worst['pct']:.0f}%) — the factors favouring {predicted} "
                         f"simply carry more combined weight.")
        return sentence

    top = opposing[0] if opposing else reasons[0]
    return (f"Interestingly, the clearest single signal — {top['text']} ({top['pct']:.0f}%) — "
            f"actually favours {top['leader']}, not {predicted}. The model's prediction still tips "
            f"towards {predicted} once all the pre-match signals are combined — a reminder that a "
            f"Random Forest weighs features jointly, not by picking the single largest one.")


def update_after_match(elo, latest_form, h2h, h2h_record, h2h_matches, history,
                       home, away, home_goals, away_goals, date=None):
    """Apply ONE new completed match to the already-fitted feature state,
    without re-running the whole pipeline or retraining the classifier.

    This is the 'incremental update' half of the answer to "can you feed in
    new match data?" — it uses the exact same Elo formula as add_elo() and
    the exact same rolling-form / H2H bookkeeping as add_form_and_h2h(), just
    applied to a single extra match. All five inputs are mutated in place
    AND returned, so the caller can decide whether to keep the update
    (e.g. store it in st.session_state) or discard it.

    Note: home-venue advantage is NOT an input here, on purpose — it is not
    part of the Elo formula in add_elo() either (Elo only reacts to the
    actual score), it is only ever used as a separate feature fed straight
    into the classifier at prediction time. So it has nothing to update here.

    The classifier itself is NOT retrained here — only the pre-match
    features it reads (Elo, form, H2H) are refreshed, so the very next
    prediction for these two teams reflects the new result immediately.
    Periodically retraining the Random Forest on the growing dataset is a
    separate, heavier step (see README / report "future work").
    """
    date = pd.Timestamp(date) if date is not None else pd.Timestamp.today().normalize()

    # --- Elo (same formula as add_elo) ---
    eh, ea = elo.get(home, BASE_ELO), elo.get(away, BASE_ELO)
    exp_h = 1 / (1 + 10 ** ((ea - eh) / 400))
    if   home_goals > away_goals: s_h = 1.0
    elif home_goals < away_goals: s_h = 0.0
    else:                         s_h = 0.5
    margin = max(np.log(abs(home_goals - away_goals) + 1), 1)
    elo[home] = eh + K * margin * (s_h - exp_h)
    elo[away] = ea + K * margin * ((1 - s_h) - (1 - exp_h))

    # --- Recent form (same points/letter convention as add_form_and_h2h) ---
    if home_goals > away_goals:
        pts_h, pts_a = 3, 0
    elif home_goals < away_goals:
        pts_h, pts_a = 0, 3
    else:
        pts_h, pts_a = 1, 1
    letter_h = "W" if pts_h == 3 else ("D" if pts_h == 1 else "L")
    letter_a = "W" if pts_a == 3 else ("D" if pts_a == 1 else "L")

    def _bump_form(team, gf, ga, pts):
        prev = latest_form.loc[team] if team in latest_form.index else pd.Series(
            {"form_pts": 1.0, "form_gf": 1.0, "form_ga": 1.0, "matches_played": 0.0})
        # simple rolling-5 approximation: nudge the average by 1/5th toward the new match
        latest_form.loc[team, "form_pts"] = prev["form_pts"] + (pts - prev["form_pts"]) / 5
        latest_form.loc[team, "form_gf"] = prev["form_gf"] + (gf - prev["form_gf"]) / 5
        latest_form.loc[team, "form_ga"] = prev["form_ga"] + (ga - prev["form_ga"]) / 5
        latest_form.loc[team, "matches_played"] = prev["matches_played"] + 1

    _bump_form(home, home_goals, away_goals, pts_h)
    _bump_form(away, away_goals, home_goals, pts_a)

    # --- Head-to-head (same convention as add_form_and_h2h) ---
    key = tuple(sorted([home, away]))
    w, t = h2h.get(key, (0, 0))
    first_win = 1 if ((key[0] == home and home_goals > away_goals) or
                      (key[0] == away and away_goals > home_goals)) else 0
    h2h[key] = (w + first_win, t + 1)

    rec = h2h_record.get(key, {"first_wins": 0, "second_wins": 0, "draws": 0, "total": 0})
    if home_goals == away_goals:
        rec["draws"] += 1
    elif first_win:
        rec["first_wins"] += 1
    else:
        rec["second_wins"] += 1
    rec["total"] += 1
    h2h_record[key] = rec

    h2h_matches.setdefault(key, []).append({
        "date": date, "home": home, "away": away,
        "home_goals": int(home_goals), "away_goals": int(away_goals),
    })

    # --- "history" table (used for the last-5 W/D/L tick display) ---
    next_id = (history["match_id"].max() + 1) if len(history) else 0
    new_rows = pd.DataFrame([
        {"match_id": next_id, "Date": date, "Team": home, "GF": home_goals, "GA": away_goals,
         "Pts": pts_h, "Result_Letter": letter_h},
        {"match_id": next_id, "Date": date, "Team": away, "GF": away_goals, "GA": home_goals,
         "Pts": pts_a, "Result_Letter": letter_a},
    ])
    history = pd.concat([history, new_rows], ignore_index=True)

    return elo, latest_form, h2h, h2h_record, h2h_matches, history


if __name__ == "__main__":
    # quick self-test
    m, elo, lf, metrics, h2h, history, h2h_record, h2h_matches = train_pipeline("international_matches1.csv")
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
