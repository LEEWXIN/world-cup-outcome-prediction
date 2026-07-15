# ---- Dockerfile: packages the dashboard + all its dependencies into one image ----
# Build (from the repo root):  docker build -t worldcup-dashboard .
# Run:                         docker run -p 8501:8501 worldcup-dashboard
# Then open http://localhost:8501 in your browser.

FROM python:3.12-slim

# 1. Set the working directory inside the container
WORKDIR /app

# 2. Install Python dependencies first (this layer is cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the app code, theme config, and the dataset into the image.
#    Source paths match this repo's folder layout (dashboard/, data/);
#    everything lands flat inside the image so model.py's default,
#    same-directory CSV lookup keeps working unchanged.
COPY dashboard/model.py dashboard/app.py dashboard/precompute.py ./
COPY dashboard/.streamlit ./.streamlit
COPY data/international_matches1.csv ./

# 4. Pre-train the model and pre-compute Elo/form/H2H once, at BUILD time,
#    and bake the result into the image as model_cache.pkl. Without this,
#    that (still nontrivial) computation re-runs from scratch every time a
#    fresh container starts, which is what made every `docker run` feel
#    like it "hangs" for a while before the app responds. Because this is
#    a normal Docker layer, it automatically re-runs (and the cache
#    automatically refreshes) whenever model.py or the CSV change - it
#    can't go silently stale.
RUN python precompute.py

# 5. Streamlit serves on port 8501
EXPOSE 8501

# 6. Start the dashboard, listening on all interfaces so it's reachable from the host
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]