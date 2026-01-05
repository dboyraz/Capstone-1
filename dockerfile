FROM python:3.13-slim-bookworm

# get uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# runtime deps often needed by sklearn wheels
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install only what's needed for serving
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# copy only the necessary service files + artifact
COPY adult_features.py predict.py serve.py model.joblib ./

ENV MODEL_PATH=/app/model.joblib
EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
