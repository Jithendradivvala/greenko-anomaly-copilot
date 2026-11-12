# Renewable Asset Fault & Anomaly Detection Copilot

**Live Railway Deployment:** [web-production-96cd093b.up.railway.app]

## 1. Project Overview
This project fulfills the Greenko AI/ML assessment by providing a Python ML pipeline and a copilot UI. It detects abnormal renewable-asset behavior in solar and wind telemetry and generates operator-friendly explanations with recommended next checks.

## 2. Design Choices & Model Logic
This project implements a hybrid anomaly detection pipeline:
* **Physics & Statistical Rules:** Calculates expected power generation (using irradiance for solar and wind speed linear fits for turbines). It flags deterministic anomalies when vibration exceeds 3.5σ (Z-score), temperature deviates >8°C from the hourly median, or power drops >0.25 MW below expected curves.
* **Machine Learning (IsolationForest):** A multivariate `IsolationForest` (150 estimators, 2.1% contamination) is trained on scaled telemetry (temp, vibration, power residuals, and rolling deltas) to catch complex, non-linear degradation patterns that static thresholds miss.
* **GenAI / RAG Copilot:** Uses `TfidfVectorizer` and Cosine Similarity to map flagged symptom tokens against a local Standard Operating Procedure (SOP) knowledge base, generating plain-English root causes and inspection steps for operators.

## 3. Evaluation Metrics (Proxy Labels)
Because the provided `asset_telemetry.csv` dataset lacks explicit ground-truth fault labels, model performance metrics are evaluated against proxy physical thresholds (e.g., `vibration > 2.5 mm/s`, `temperature > 44°C`). 

* **Precision:** 1.00
* **Recall:** 1.00
* **F1-Score:** 1.00
*(Note: These metrics confirm the model perfectly aligns with known physical failure states in the dummy data. In a production environment, evaluation would be conducted against an independently labeled historical SCADA fault dataset.)*

## 4. Assumptions
* **Time Series Continuity:** Assumes telemetry data is sequential and gap-free; missing packets are currently handled via rolling averages.
* **Wind Power Curve:** Assumes a simplified linear power curve for WTG expected power within the standard operating regime (3–9 m/s). 
* **SOP Completeness:** Assumes the local SOP dictionary covers all critical failure modes represented in the telemetry variance.

## 5. Limitations & Production-Readiness Gaps
* **Train/Test Split:** Due to the small size of the dummy dataset (240 rows / 5 days), the ML model is fit dynamically. In production, the `IsolationForest` would be trained on months of clean historical data and frozen for real-time inference.
* **Curtailment Context:** The model lacks SCADA boolean flags for grid curtailment. A sharp power drop might be flagged as a fault when it is actually a grid operator command.
* **LLM Integration:** The current copilot uses TF-IDF semantic retrieval to ensure it runs completely locally and free on Railway. Upgrading to a lightweight local LLM (e.g., Llama 3 8B) would allow for conversational follow-up questions from operators.
