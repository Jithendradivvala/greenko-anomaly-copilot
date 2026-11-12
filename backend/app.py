from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

app = FastAPI(title='Renewable Asset Fault & Anomaly Detection Copilot')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
FRONTEND = BASE / 'frontend'


class Query(BaseModel):
    text: str = ''
    asset_id: Optional[str] = None
    contamination: float = 0.021


SOP_KNOWLEDGE_BASE = [
    {
        "doc_id": "SOP-SOLAR-001",
        "asset_type": "SOLAR_INV",
        "title": "Solar Inverter Nighttime Over-Temperature & High Vibration",
        "symptoms": "high temperature_c above 45C at night zero irradiance_wm2 high vibration_mm_s above 3.0 mm/s zero power_mw",
        "root_cause": "Inverter forced-air cooling fan bearing seizure or loose impeller causing mechanical chassis vibration and cabinet thermal buildup during post-sunset discharge/standby.",
        "recommended_checks": [
            "Inspect inverter cooling fan assembly and fan bearings for mechanical play or blockage.",
            "Check cabinet air filters and exhaust louvers for dust/debris obstruction.",
            "Verify auxiliary power supply and contactor relay chatter inside the inverter cabinet.",
            "Review inverter internal IGBT and transformer thermal sensor logs before morning ramp-up."
        ]
    },
    {
        "doc_id": "SOP-SOLAR-002",
        "asset_type": "SOLAR_INV",
        "title": "Solar Inverter Daytime Power Derating & Conversion Loss",
        "symptoms": "high irradiance_wm2 low power_mw negative power_residual_mw high temperature_c",
        "root_cause": "DC string fuse blown, MPPT tracker derating due to thermal limits, or grid curtailment command active.",
        "recommended_checks": [
            "Check combiner box string currents for open circuits or blown fuses.",
            "Verify inverter derating status flags and ambient cabinet temperature.",
            "Confirm SCADA active power setpoint and grid frequency response logs."
        ]
    },
    {
        "doc_id": "SOP-WIND-001",
        "asset_type": "WTG",
        "title": "Wind Turbine Gearbox / Bearing Overheating, Vibration & Power Derating",
        "symptoms": "WTG high vibration_mm_s above 3.0 mm/s high nacelle temperature_c above 44C power_mw drop negative power_residual_mw",
        "root_cause": "Main shaft or gearbox high-speed stage bearing degradation causing mechanical friction, elevated vibration, thermal spike, and automatic PLC protective power derating.",
        "recommended_checks": [
            "Inspect gearbox oil temperature, pressure, and inline metal particle counter alarms.",
            "Analyze Condition Monitoring System (CMS) vibration spectra on high-speed shaft bearings.",
            "Verify pitch angle calibration and check if turbine controller triggered thermal curtailment.",
            "Schedule borescope inspection before returning WTG to full rated load."
        ]
    },
    {
        "doc_id": "SOP-WIND-002",
        "asset_type": "WTG",
        "title": "Wind Turbine Pitch / Anemometer Calibration Drift",
        "symptoms": "WTG abnormal power_residual_mw normal vibration_mm_s normal temperature_c",
        "root_cause": "Nacelle anemometer icing/misalignment or blade pitch actuator lag causing suboptimal aerodynamic torque.",
        "recommended_checks": [
            "Compare nacelle wind speed reading against met-mast and neighboring turbines.",
            "Check blade pitch cylinder hydraulic pressure and angle symmetry across all three blades."
        ]
    }
]


def load_and_engineer_features() -> pd.DataFrame:
    df = pd.read_csv(DATA / 'asset_telemetry.csv')
    df['asset_type'] = df['asset_id'].apply(lambda x: 'SOLAR_INV' if 'SOLAR' in x else 'WTG')
    df['day'] = (df.groupby('asset_id').cumcount() // 24) + 1

    df['expected_power_mw'] = 0.0

    # Solar expected power: linear with irradiance
    solar_mask = df['asset_type'] == 'SOLAR_INV'
    clean_solar = df[solar_mask & (df['vibration_mm_s'] < 2.5)]
    if len(clean_solar) > 0 and clean_solar['irradiance_wm2'].sum() > 0:
        k_solar = (clean_solar['power_mw'] * clean_solar['irradiance_wm2']).sum() / (
            (clean_solar['irradiance_wm2'] ** 2).sum()
        )
    else:
        k_solar = 0.0019
    df.loc[solar_mask, 'expected_power_mw'] = (df.loc[solar_mask, 'irradiance_wm2'] * k_solar).round(3)

    # Wind expected power: linear fit on wind_speed_ms in operating regime
    wind_mask = df['asset_type'] == 'WTG'
    clean_wind = df[wind_mask & (df['vibration_mm_s'] < 2.5)]
    if len(clean_wind) > 1:
        slope, intercept = np.polyfit(clean_wind['wind_speed_ms'], clean_wind['power_mw'], 1)
    else:
        slope, intercept = 0.4436, -1.312
    df.loc[wind_mask, 'expected_power_mw'] = np.maximum(
        0.0, slope * df.loc[wind_mask, 'wind_speed_ms'] + intercept
    ).round(3)

    df['power_residual_mw'] = (df['power_mw'] - df['expected_power_mw']).round(3)
    df['abs_power_residual_mw'] = df['power_residual_mw'].abs()

    for col in ['temperature_c', 'vibration_mm_s', 'power_mw']:
        grp = df.groupby('asset_id')[col]
        df[f'{col}_roll_mean_6h'] = grp.transform(lambda s: s.rolling(6, min_periods=1).mean()).round(3)
        df[f'{col}_delta_1h'] = grp.transform(lambda s: s.diff().fillna(0.0)).round(3)

        med = grp.transform('median')
        mad = grp.transform(lambda s: np.median(np.abs(s - np.median(s))))
        df[f'{col}_zscore'] = ((df[col] - med) / (1.4826 * mad + 1e-6)).round(2)

    hr_grp = df.groupby(['asset_id', 'hour'])['temperature_c']
    hr_med = hr_grp.transform('median')
    df['temp_hourly_deviation_c'] = (df['temperature_c'] - hr_med).round(2)

    return df


def retrieve_sop(asset_type: str, symptom_query: str) -> Dict[str, Any]:
    docs = [d for d in SOP_KNOWLEDGE_BASE if d['asset_type'] == asset_type] or SOP_KNOWLEDGE_BASE
    corpus = [f"{d['title']} {d['symptoms']} {d['root_cause']}" for d in docs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus + [symptom_query])
    sims = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()
    best_idx = int(np.argmax(sims))
    best_doc = docs[best_idx].copy()
    best_doc['similarity_score'] = round(float(sims[best_idx]), 3)
    return best_doc


def run_anomaly_pipeline(contamination: float = 0.021, asset_filter: Optional[str] = None, query_text: str = '') -> Dict[str, Any]:
    df = load_and_engineer_features()

    feature_cols = [
        'temperature_c',
        'vibration_mm_s',
        'abs_power_residual_mw',
        'temp_hourly_deviation_c',
        'temperature_c_delta_1h',
        'vibration_mm_s_delta_1h'
    ]

    df['iforest_anomaly'] = False
    df['iforest_score'] = 0.0

    for asset_id, group in df.groupby('asset_id'):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(group[feature_cols])
        clf = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42
        )
        preds = clf.fit_predict(X_scaled)
        scores = -clf.score_samples(X_scaled)
        df.loc[group.index, 'iforest_anomaly'] = (preds == -1)
        df.loc[group.index, 'iforest_score'] = np.round(scores, 3)

    df['stat_rule_anomaly'] = (
        (df['vibration_mm_s_zscore'].abs() > 3.5) |
        (df['temp_hourly_deviation_c'].abs() > 8.0) |
        (df['abs_power_residual_mw'] > 0.25)
    )

    df['is_anomaly'] = df['iforest_anomaly'] | df['stat_rule_anomaly']

    # Normalize asset_filter strictly from dropdown selection
    if asset_filter in ('ALL', ''):
        asset_filter = None

    if asset_filter is None and query_text:
        q_lower = query_text.lower()
        has_solar = ('solar' in q_lower or 'solar_inv_01' in q_lower)
        has_wind = ('wind' in q_lower or 'wtg' in q_lower or 'turbine' in q_lower)
        if has_solar and not has_wind:
            asset_filter = 'SOLAR_INV_01'
        elif has_wind and not has_solar:
            asset_filter = 'WTG_07'

    view_df = df[df['asset_id'] == asset_filter].copy() if asset_filter else df.copy()

    # Calculate evaluation metrics on the active view
    ground_truth = (
        (view_df['vibration_mm_s'] > 2.5) |
        (view_df['temperature_c'] > 44.0) |
        (view_df['abs_power_residual_mw'] > 0.25)
    ).astype(int)

    y_iforest = view_df['iforest_anomaly'].astype(int)
    y_stat = view_df['stat_rule_anomaly'].astype(int)
    y_hybrid = view_df['is_anomaly'].astype(int)

    tn, fp, fn, tp = confusion_matrix(ground_truth, y_hybrid, labels=[0, 1]).ravel()
    metrics = {
        "hybrid_ensemble": {
            "precision": round(float(precision_score(ground_truth, y_hybrid, zero_division=0)), 3),
            "recall": round(float(recall_score(ground_truth, y_hybrid, zero_division=0)), 3),
            "f1_score": round(float(f1_score(ground_truth, y_hybrid, zero_division=0)), 3),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        },
        "isolation_forest_only": {
            "precision": round(float(precision_score(ground_truth, y_iforest, zero_division=0)), 3),
            "recall": round(float(recall_score(ground_truth, y_iforest, zero_division=0)), 3),
            "f1_score": round(float(f1_score(ground_truth, y_iforest, zero_division=0)), 3)
        },
        "statistical_rules_only": {
            "precision": round(float(precision_score(ground_truth, y_stat, zero_division=0)), 3),
            "recall": round(float(recall_score(ground_truth, y_stat, zero_division=0)), 3),
            "f1_score": round(float(f1_score(ground_truth, y_stat, zero_division=0)), 3)
        }
    }

    anomalies_df = view_df[view_df['is_anomaly']].copy()
    anomaly_records = []

    for _, row in anomalies_df.iterrows():
        reasons = []
        symptom_tokens = [row['asset_type']]

        if row['vibration_mm_s'] > 2.5:
            reasons.append(f"Vibration spiked to {row['vibration_mm_s']} mm/s")
            symptom_tokens.append("high vibration_mm_s above 3.0 mm/s")

        if row['temp_hourly_deviation_c'] > 8.0 or row['temperature_c'] > 44.0:
            reasons.append(f"Temperature reached {row['temperature_c']}°C (+{row['temp_hourly_deviation_c']}°C above baseline)")
            symptom_tokens.append("high temperature_c above 45C")
            if row['asset_type'] == 'SOLAR_INV' and row['irradiance_wm2'] == 0:
                symptom_tokens.append("at night zero irradiance_wm2 zero power_mw")

        if row['abs_power_residual_mw'] > 0.25:
            reasons.append(f"Power dropped to {row['power_mw']} MW vs expected {row['expected_power_mw']} MW")
            symptom_tokens.append("power_mw drop negative power_residual_mw")

        if not reasons:
            reasons.append(f"IsolationForest anomaly score ({row['iforest_score']}) exceeded baseline.")

        sop = retrieve_sop(row['asset_type'], " ".join(symptom_tokens))
        severity = "CRITICAL" if (row['vibration_mm_s'] > 3.0 and (row['temperature_c'] > 45.0 or row['abs_power_residual_mw'] > 0.25)) else "WARNING"

        operator_summary = f"[{severity}] {row['asset_id']} (ID #{int(row['timestamp_id'])}): {'; '.join(reasons)}. Likely Cause: {sop['root_cause']}"

        anomaly_records.append({
            "timestamp_id": int(row['timestamp_id']),
            "asset_id": row['asset_id'],
            "day": int(row['day']),
            "hour": int(row['hour']),
            "severity": severity,
            "telemetry": {
                "irradiance_wm2": float(row['irradiance_wm2']),
                "wind_speed_ms": float(row['wind_speed_ms']),
                "temperature_c": float(row['temperature_c']),
                "vibration_mm_s": float(row['vibration_mm_s']),
                "power_mw": float(row['power_mw']),
                "expected_power_mw": float(row['expected_power_mw']),
                "power_residual_mw": float(row['power_residual_mw'])
            },
            "detection_flags": {
                "isolation_forest": bool(row['iforest_anomaly']),
                "statistical_rule": bool(row['stat_rule_anomaly']),
                "iforest_score": float(row['iforest_score']),
                "vibration_zscore": float(row['vibration_mm_s_zscore']),
                "temp_hourly_deviation_c": float(row['temp_hourly_deviation_c'])
            },
            "operator_explanation": operator_summary,
            "reasons": reasons,
            "retrieved_sop": sop
        })

    # DYNAMIC COPILOT SUMMARY GENERATION
    # DYNAMIC COPILOT SUMMARY GENERATION
    copilot_answer = f"Analyzed {len(view_df)} interval(s) across {view_df['asset_id'].nunique()} asset(s), detecting {len(anomaly_records)} abnormal condition(s). "
    if len(anomaly_records) > 0:
        asset_groups = {}
        for r in anomaly_records:
            if r['asset_id'] not in asset_groups:
                asset_groups[r['asset_id']] = []
            asset_groups[r['asset_id']].append(r)
            
        for asset, records in asset_groups.items():
            primary_sop = records[0]['retrieved_sop']['title']
            copilot_answer += f"On {asset}, {len(records)} anomaly record(s) flagged (starting at ID #{records[0]['timestamp_id']}), requiring field checks for {primary_sop}. "
    else:
        copilot_answer += "No anomalies were detected by the configured detection methods for the selected asset and dataset."

    return {
        "input": query_text,
        "asset_filter": asset_filter or "ALL",
        "summary": {
            "total_records_analyzed": len(view_df),
            "anomalies_detected": len(anomaly_records),
            "anomaly_rate_pct": round(len(anomaly_records) / max(len(view_df), 1) * 100, 2),
            "copilot_executive_summary": copilot_answer.strip()
        },
        "predictions": anomaly_records,
        "retrieval_evidence": list({r['retrieved_sop']['doc_id']: r['retrieved_sop'] for r in anomaly_records}.values()),
        "metrics": metrics,
        "telemetry_series": view_df[[
            'timestamp_id', 'asset_id', 'day', 'hour', 'irradiance_wm2',
            'wind_speed_ms', 'temperature_c', 'vibration_mm_s',
            'power_mw', 'expected_power_mw', 'power_residual_mw',
            'is_anomaly', 'iforest_score'
        ]].to_dict(orient='records')
    }


@app.get('/')
def serve_frontend():
    return FileResponse(FRONTEND / 'index.html')


@app.get('/case_study')
def serve_case_study():
    return FileResponse(BASE / 'case_study.html')


@app.get('/api/demo')
def demo():
    pipeline_output = run_anomaly_pipeline(contamination=0.021)
    return {
        'case_study': 'Renewable Asset Fault & Anomaly Detection Copilot',
        'status': 'ML & Copilot Pipeline Active',
        'expected_outcome': 'A Python ML pipeline that flags anomalies in solar/wind asset telemetry and a copilot UI that explains likely causes and recommended next checks.',
        'summary': pipeline_output['summary'],
        'metrics': pipeline_output['metrics']
    }


@app.post('/api/run')
def run_case(query: Query):
    return run_anomaly_pipeline(
        contamination=query.contamination,
        asset_filter=query.asset_id,
        query_text=query.text
    )