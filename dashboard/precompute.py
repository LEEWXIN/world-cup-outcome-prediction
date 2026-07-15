"""
precompute.py
-------------
Run ONCE, at Docker image build time (see Dockerfile), to train the model
and pre-compute all Elo / form / head-to-head feature state, then save
everything to model_cache.pkl.

Why this exists: even after vectorising the Elo/form/H2H loops in
model.py (which cut the cold-start time roughly in half), that
computation still has to run from scratch every time a fresh container
starts, because Streamlit's @st.cache_resource cache lives in the
process's memory and is wiped whenever the container restarts. Since
testing/recording a demo means restarting the container over and over,
that repeated cost adds up and looks like the app "hanging" on every
restart. Baking the already-trained result into the image at BUILD time
(this script) means every `docker run` afterwards just loads a pickle
file - near-instant - instead of re-running the pipeline.

If you add new data or change model.py's logic, rebuilding the image
re-runs this script automatically (Docker re-executes RUN steps whose
inputs changed), so the cache can never silently go stale.
"""
import pickle
import model as M

CACHE_PATH = "model_cache.pkl"


def main():
    mdl, elo, latest_form, metrics, h2h, history, h2h_record, h2h_matches = (
        M.train_pipeline("international_matches1.csv")
    )
    teams = sorted(elo.keys())
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {
                "mdl": mdl,
                "elo": elo,
                "latest_form": latest_form,
                "metrics": metrics,
                "h2h": h2h,
                "history": history,
                "h2h_record": h2h_record,
                "h2h_matches": h2h_matches,
                "teams": teams,
            },
            f,
        )
    print(f"Pre-computed model cache written to {CACHE_PATH} "
          f"({metrics['n_train'] + metrics['n_test']:,} matches, "
          f"accuracy {metrics['accuracy']:.3f}).")


if __name__ == "__main__":
    main()
