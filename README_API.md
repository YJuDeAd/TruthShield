# TruthShield Backend API

Production-ready REST API for AI-powered misinformation and phishing detection.

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Endpoint Reference](#endpoint-reference)
- [Rate Limits](#rate-limits)
- [Support](#support)
- [Notes](#notes)

## Overview

TruthShield API exposes unified detection services for:

- **News** misinformation detection (RoBERTa-large)
- **SMS/Email** phishing detection (Bi-LSTM + CNN)
- **Multimodal** verification for image + text content (ResNet18 + DistilBERT)

The API includes authentication, async processing, explainability endpoints, user history, feedback collection, and model/admin management tools.

## Tech Stack

- **Framework:** FastAPI
- **Server:** Uvicorn
- **Database:** SQLite
- **Auth:** JWT + API keys
- **Model Serving:** PyTorch-based inference integration

## Project Structure

```text
api/
├── main.py
├── auth.py
├── config.py
├── database.py
├── ml_models.py
├── schemas.py
└── routers/
    ├── auth_router.py
    ├── detection.py
    ├── explainability.py
    ├── feedback.py
    ├── history.py
    ├── jobs.py
    ├── model_management.py
    └── utilities.py
```

## Quick Start

### 1) Install dependencies

```bash
cd api
pip install -r requirements.txt
```

### 2) Configure environment (optional)

```bash
cp .env.example .env            # edit values as needed
```

### 3) Run the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4) Open API docs

- Swagger UI: http://localhost:8000/docs
- OpenAPI JSON: http://localhost:8000/openapi.json

## Endpoint Reference

| Category | Method | Endpoint | Description |
|----------|--------|----------|-------------|
| Detection | POST | `/api/v1/detect/news` | Detect fake news |
| Detection | POST | `/api/v1/detect/sms` | Detect SMS phishing |
| Detection | POST | `/api/v1/detect/multimodal` | Detect fake content (text + image) |
| Detection | POST | `/api/v1/detect` | Auto-route detection |
| Detection | POST | `/api/v1/detect/batch` | Batch detection |
| Async Jobs | POST | `/api/v1/jobs/detect` | Submit async job |
| Async Jobs | GET | `/api/v1/jobs/{job_id}` | Get job status |
| Async Jobs | GET | `/api/v1/jobs/{job_id}/result` | Get job result |
| Async Jobs | DELETE | `/api/v1/jobs/{job_id}` | Cancel job |
| Explainability | POST | `/api/v1/explain/news` | Explain news prediction |
| Explainability | POST | `/api/v1/explain/sms` | Explain SMS prediction |
| Auth | POST | `/api/v1/auth/register` | Register user |
| Auth | POST | `/api/v1/auth/login` | Login |
| Auth | POST | `/api/v1/auth/refresh` | Refresh token |
| Auth | DELETE | `/api/v1/auth/revoke` | Revoke API key |
| Auth | GET | `/api/v1/auth/users/me` | Get current user |
| History | GET | `/api/v1/history` | User history |
| History | GET | `/api/v1/history/{request_id}` | Request detail |
| Analytics | GET | `/api/v1/stats` | User stats |
| Analytics | GET | `/api/v1/stats/global` | Global stats (admin) |
| Feedback | POST | `/api/v1/feedback` | Submit feedback |
| Feedback | GET | `/api/v1/feedback` | Feedback queue (admin) |
| Feedback | POST | `/api/v1/feedback/retrain/trigger` | Trigger retraining (admin) |
| Feedback | GET | `/api/v1/feedback/retrain/status` | Retraining status (admin) |
| Model Mgmt | GET | `/api/v1/models` | List models |
| Model Mgmt | GET | `/api/v1/models/{name}` | Model info |
| Model Mgmt | POST | `/api/v1/models/reload` | Reload models (admin) |
| Utilities | GET | `/api/v1/models/health` | Health check |
| Utilities | GET | `/api/v1/models/metrics` | Prometheus metrics (admin) |
| Utilities | GET | `/api/v1/version` | API version |
| Docs | GET | `/docs` | Swagger UI |

## Rate Limits

- Default: **100 requests/minute**
- Daily quota: **1000 requests** (per user, configurable)
- Admin users: **Unlimited**

## Support

- API docs: http://localhost:8000/docs
- Issues: https://github.com/YJuDeAd/TruthShield/issues

## Notes

- For frontend-specific details, see `README_UI.md`.
- For complete project setup, see the root `README.md`.