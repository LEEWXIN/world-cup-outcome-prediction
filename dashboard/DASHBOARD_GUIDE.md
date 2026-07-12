# Dashboard + Docker — How It Works & How to Run It

This folder is the **bonus** part of your project: an interactive Streamlit
dashboard, packaged with Docker. It uses the **same Elo + form model** as your
Jupyter notebook, so you can honestly say *"the dashboard and the notebook are
one and the same analysis — this is just the clickable version."*

---

## 1. What each file does (so you can explain it)

| File | Plain-language job |
|---|---|
| `model.py` | The "brain". Loads the data, builds the Elo + form features, trains the Random Forest, and predicts a matchup. **Same steps as your notebook.** |
| `app.py` | The "face". The Streamlit screen — dropdowns to pick two teams, then it shows the prediction and the *why*. It just calls `model.py`. |
| `requirements.txt` | The exact library versions needed, so it runs the same on any computer. |
| `Dockerfile` | The recipe that packs the app + libraries + data into one **container** that runs anywhere. |
| `international_matches1.csv` | The dataset the model learns from. |

---

## 2. Run it WITHOUT Docker (quick, for development)

```bash
pip install -r requirements.txt
streamlit run app.py
```
A browser tab opens at `http://localhost:8501`. Pick two teams → see the
prediction and the explanation.

---

## 3. Run it WITH Docker (this is the part your lecturer wants you to practice)

Make sure Docker Desktop is installed and running, then from inside this folder:

```bash
# Step 1 — build the image (the "recipe" runs once, may take a few minutes)
docker build -t worldcup-dashboard .

# Step 2 — run a container from that image
docker run -p 8501:8501 worldcup-dashboard
```
Open `http://localhost:8501` in your browser. To stop it, press `Ctrl + C`.

**What just happened, in one sentence you can say out loud:**
> "Docker took my app, the Python libraries, and the dataset, and sealed them
> into one container so it runs identically on any machine — no 'it works on my
> laptop but not yours' problem."

Useful commands to know for questions:
```bash
docker images                 # list the image you built
docker ps                     # list running containers
docker stop <container_id>    # stop a running container
```

---

## 4. What to actually SHOW and SAY in your live demo

1. **Open the dashboard.** "This is the interactive version of my model."
2. **Pick a lopsided match** (e.g. Brazil vs Qatar). "It predicts a Brazil win
   with a high probability." *(Don't memorise an exact number here — read
   whatever the app shows live, since Elo ratings update slightly every time
   the dataset is refreshed. As of this writing it's ~88%, with a ~470-point
   Elo gap, but the demo should always show the real number on screen, not a
   number from this script.)*
3. **Point at the *Why* panel.** "And here's the answer to the key question —
   *why*. Brazil's Elo rating is meaningfully higher; team strength is the
   single biggest driver. Recent form and home advantage adjust it."
4. **The headline moment — France vs Spain.** Select France and Spain, and
   **uncheck "Home venue"** (AT&T Stadium, Dallas is a neutral venue for both
   teams — this must match Section 4.4 of the report, or the numbers won't
   line up). "This is the same France v Spain semifinal I predicted in my
   report, on 12 July, before it was played on the 14th — a genuine forecast,
   not a lookback. My report is honest that the 2022 example earlier used a
   snapshot with future information baked in; this fixes that same problem
   live, in front of you."
5. **Pick a close match** (e.g. Argentina vs France). "Now the probabilities are
   much tighter — the model is appropriately less certain."
6. **Show the feature-importance bar.** "Across all 39,000+ training matches, these are the
   factors that decide outcomes: strength gap first, then defensive form."
7. **Mention Docker.** "I containerised it with Docker so it's fully reproducible."

That sequence covers data → model → prediction → *explanation* → **self-correction** → reproducibility,
which is exactly what the rubric rewards — step 4 in particular is worth lingering
on if you're asked questions, since it's the strongest evidence you understand
your own model's limitations.

---

## 5. Common questions you might get (and honest answers)

- **"Why is the accuracy only ~60%?"** Three-way football prediction is genuinely
  hard and draws are near-random; 60% beats the 48% baseline meaningfully, and
  anything near 90% would mean I accidentally leaked future data.
- **"Where does 'home advantage' come from?"** It's the *Home Stadium or Not*
  flag in the data, exposed as a checkbox so you can toggle it.
- **"Is this the same model as the notebook?"** Yes — `model.py` is the notebook's
  pipeline as functions. The dashboard just adds a UI on top.
- **"Why does your France v Spain number differ slightly from the notebook's
  Section 8b?"** The dashboard trains a Random Forest with fixed parameters
  (`model.py`) for speed; the notebook's is tuned with GridSearchCV. Same
  pipeline, two separately-trained models — expect small (≤1 point) differences,
  not a discrepancy to worry about.
- **"Why trust the France v Spain prediction but not the 2022 one?"** Timing.
  The 2022 example used 2026-dated Elo to "predict" a 2022 match — the model
  already knew the future. France v Spain was predicted on 12 July, before its
  14 July kick-off — nothing about the outcome existed yet when the number was
  generated.
