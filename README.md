## Problem

This project predicts whether a person's income is >50K or <=50K using the [Adult Income](https://archive.ics.uci.edu/dataset/2/adult) dataset. It is a binary classification task intended to demonstrate an end to end Machine Learning workflow from exploratory data analysis and feature engineering to model training and serving a prediction API.

## Usage context

Use the trained model to score a single record or batch of records via the FastAPI `/predict` endpoint. A sample request payload is provided in `payload.json` (in the root directory) and the example curl commands below show how to call the service locally.



**Before using these commands make sure you are in the project folder in your terminal.**

1. Build the Docker image:
```
docker build -t adult-income-api .
```
2. Run the container:
```
docker run --rm -p 8000:8000 adult-income-api
```

3. Now you can test the API using the following curl command:

*Windows PowerShell*
```
curl.exe -s -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  --data-binary "@payload.json" 
```



*macOS/Linux*
```
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  --data-binary "@payload.json"
```

Alternatively you can use the built-in [Swagger](http://localhost:8000/docs#/default/predict_endpoint_predict_post) to test the API, after running the container successfully.



## Virtual Environment

This project uses [uv](https://docs.astral.sh/uv/) as package manager.

Windows
```
uv venv
.venv\Scripts\activate
```

macOS/Linux
```
uv venv
source .venv/bin/activate
```

Install dependencies:
```
uv sync
```

If you prefer requirements.txt instead of uv.lock:

```
uv pip install -r requirements.txt
```