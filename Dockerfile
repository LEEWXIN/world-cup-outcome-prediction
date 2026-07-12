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
COPY dashboard/model.py dashboard/app.py ./
COPY dashboard/.streamlit ./.streamlit
COPY data/international_matches1.csv ./

# 4. Streamlit serves on port 8501
EXPOSE 8501

# 5. Start the dashboard, listening on all interfaces so it's reachable from the host
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]