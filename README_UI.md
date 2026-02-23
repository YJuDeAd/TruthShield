# TruthShield Frontend (Streamlit)

Modern Streamlit UI for interacting with the TruthShield detection API.

## Table of Contents

- [Overview](#overview)
- [Core Features](#core-features)
- [User Experience](#user-experience)
- [Developer Mode](#developer-mode)
- [Run Locally](#run-locally)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Notes](#notes)

## Overview

TruthShield Frontend provides a user-friendly interface for:

- Fake news detection
- SMS/email phishing detection
- Multimodal verification (text + image)
- Auto-detection routing through backend APIs

Built for both non-technical users and developers, the app supports a clean default mode plus an advanced developer mode.

## Core Features

- **Simple detection workflow** with guided inputs and clear results
- **Visual verdict cards** for quick Real/Fake interpretation
- **Confidence meters and probability breakdowns**
- **Session-based authentication** (optional for most workflows)
- **Detection history, dashboards, and feedback submission**
- **Developer tools** for API-level testing and diagnostics

## User Experience

Default (User) mode includes:

- **Home:** system status and quick stats
- **Detect Content:** News, SMS/Email, Multimodal, or Auto mode
- **My Dashboard:** personal usage analytics
- **History:** previous detection requests
- **Give Feedback:** correct model outputs and improve future retraining

## Developer Mode

Enable **Developer Mode** from the sidebar to unlock advanced features:

- **Auth (Dev):** register/login/refresh/revoke workflows
- **Jobs & APIs (Dev):** async jobs, model operations, utilities
- **Explainability (Dev):** model explanation responses
- **Admin Panel (Dev):** global analytics, feedback queue, retraining actions
- **Raw JSON toggles:** inspect backend payloads directly

## Run Locally

### 1) Start the backend API

From `api/`:

```bash
python main.py
```

### 2) Launch Streamlit

From `frontend/`:

```bash
streamlit run app.py
```

### 3) Open the app

http://localhost:8501

## Quick Start

1. Open the app in your browser.
2. Go to **Detect Content** in the sidebar.
3. Choose News, SMS/Email, Multimodal, or Auto.
4. Submit content for analysis.
5. Review verdict, confidence, and probability details.

## Authentication

Authentication is optional for basic detection and recommended for personalized features.

Login enables:

- Personal dashboard and stats
- Detection history
- Feedback submission
- Quota visibility

Default admin credentials:

- `admin`
- `admin123`

## Notes

- For full backend API details, see `README_API.md`.
- For complete project setup, see the root `README.md`.
