# MLOps Assignment 01 - Heart Disease Risk Prediction

This repository implements an end-to-end MLOps pipeline for heart disease risk classification using the Heart Disease UCI dataset.

## Main Features

- Data acquisition script for the UCI Heart Disease dataset
- EDA notebook with histograms, correlation heatmap, class balance and missing value analysis
- Reusable preprocessing pipeline using `ColumnTransformer`
- Logistic Regression and Random Forest models
- Hyperparameter tuning using GridSearchCV
- MLflow experiment tracking with metrics, parameters and artifacts
- FastAPI prediction service with `/predict` and `/metrics`
- Dockerized model serving
- Pytest unit tests
- GitHub Actions CI workflow
- Kubernetes deployment and service manifests
- Basic logging and Prometheus metrics

## Project Structure

```text
├── api/                    # FastAPI serving code
├── data/                   # raw and processed data folders
├── k8s/                    # Kubernetes deployment/service YAML
├── monitoring/             # Prometheus configuration
├── notebooks/              # EDA and training notebook
├── src/                    # training, preprocessing and data scripts
├── tests/                  # unit tests
├── .github/workflows/      # CI pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Download Dataset

```bash
python src/data_acquisition.py
```

## Train Model

```bash
python src/train.py
```

This creates `models/heart_pipeline.joblib` and logs experiments to MLflow.

## Run MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000`.

## Run API Locally

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI at `http://localhost:8000/docs`.

## Sample Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

## Docker

```bash
docker build -t heart-disease-api:latest .
docker run -p 8000:8000 heart-disease-api:latest
```

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods
kubectl get svc
```

## Tests

```bash
pytest tests/ -v
```

