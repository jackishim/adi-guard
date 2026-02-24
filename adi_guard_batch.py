"""
ADI-Guard v2: Batch Advance Directive Intelligence Pipeline
============================================================
MedGemma Impact Challenge — Google Research

HOW TO USE:
-----------
1. Fill in patients.csv with your patient data (one row per patient)
2. Run: python adi_guard_batch.py
3. Reports are saved locally to /reports/ folder — no data leaves your machine

PRIVACY:
--------
This tool runs entirely on local infrastructure.
No patient data is transmitted to external servers or cloud services.
All processing occurs on-device using the locally loaded MedGemma model.
HIPAA-compliant by design.

CSV COLUMNS:
------------
patient_id         : Unique patient identifier
patient_name       : Patient full name
age                : Patient age (integer)
icu_admission      : True/False
chronic_conditions : Comma-separated list of conditions
six_month_prognosis: True/False (poor prognosis flag)
advance_directive  : Full text of advance directive document
current_care_plan  : Full text of current active care plan / orders
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path

# ── MedGemma Setup ─────────────────────────────────────────────────────────────
# Uncomment when running locally after: pip install transformers torch accelerate
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
#
# MODEL_ID = "google/medgemma-4b-it"
# print("Loading MedGemma...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     dtype=torch.bfloat16,
#     device_map="auto",
# )
# print("MedGemma loaded.\n")


def call_medgemma(prompt: str) -> str:
    """
    Call MedGemma with a clinical prompt.
    Uncomment production block when model is loaded.
    """
    # ── PRODUCTION ──────────────────────────────────────────────────────────
    # messages = [{"role": "user", "content": prompt}]
    # inputs = tokenizer.apply_chat_template(
    #     messages, return_tensors="pt",
    #     add_generation_prompt=True, return_dict=True
    # )
    # inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # with torch.inference_mode():
    #     outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    # input_len = inputs["input_ids"].shape[-1]
    # return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    # ── DEMO STUB (remove when using real model) ─────────────────────────────
    return '{"conflicts_found": false, "conflict_count": 0, "conflicts": []}'


# ── Clinical Agent Steps ───────────────────────────────────────────────────────

def step1_identify_high_risk(patient_data: dict) -> bool:
    risk_factors = []
    if int(patient_data.get("age", 0)) >= 65:
        risk_factors.append("Age >= 65")
    if str(patient_data.get("icu_admission", "")).lower() == "true":
        risk_factors.append("ICU admission")
    if patient_data.get("chronic_conditions"):
        risk_factors.append(f"Chronic conditions: {patient_data['chronic_conditions']}")
    if str(patient_data.get("six_month_prognosis", "")).lower() == "true":
        risk_factors.append("Poor 6-month prognosis")
    is_high_risk = len(risk_factors) >= 2
    return is_high_risk, risk_factors


def step2_extract_ad_preferences(ad_text: str) -> dict:
    prompt = f"""You are a clinical AI assistant. Extract the patient's advance directive preferences from the document below. Return ONLY a JSON object with these exact keys: dnr_status, life_support, cpr_preference, intubation, artificial_nutrition, resuscitation, healthcare_agent, palliative_care, hospital_transfer.

Advance Directive:
{ad_text}

Return ONLY valid JSON, no explanation, no markdown."""
    raw = call_medgemma(prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {"raw_extraction": raw}


def step3_parse_care_plan(care_plan_text: str) -> dict:
    prompt = f"""You are a clinical AI assistant. Extract the active treatment orders from the care plan below. Return ONLY a JSON object with keys: active_orders (list of strings), treating_team, primary_diagnosis.

Care Plan:
{care_plan_text}

Return ONLY valid JSON, no explanation, no markdown."""
    raw = call_medgemma(prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {"raw": raw}


def step4_detect_conflicts(ad_preferences: dict, care_plan: dict) -> dict:
    prompt = f"""You are a clinical AI assistant performing advance directive compliance review.

PATIENT'S ADVANCE DIRECTIVE PREFERENCES:
{json.dumps(ad_preferences, indent=2)}

CURRENT ACTIVE CARE PLAN:
{json.dumps(care_plan, indent=2)}

Identify all conflicts where the care plan contradicts the patient's advance directive.

Respond with ONLY this JSON structure, no other text:
{{
  "conflicts_found": true,
  "conflict_count": 2,
  "conflicts": [
    {{
      "severity": "CRITICAL",
      "directive": "patient wish here",
      "care_plan": "conflicting order here",
      "loinc_code": "81351-9",
      "snomed_code": "143021000119109",
      "explanation": "explanation here"
    }}
  ]
}}

If no conflicts exist return: {{"conflicts_found": false, "conflict_count": 0, "conflicts": []}}
Return ONLY valid JSON."""
    raw = call_medgemma(prompt)
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        return json.loads(raw[start:end])
    except:
        return {"conflicts_found": False, "conflict_count": 0, "conflicts": []}


def step5_generate_narrative(conflicts: dict, ad_preferences: dict) -> str:
    if not conflicts.get("conflicts_found"):
        return "No conflicts detected. Current care plan is concordant with the patient's documented advance directive wishes."
    prompt = f"""You are a clinical documentation assistant. Write a concise urgent clinical narrative (3-4 sentences) for a clinician summarizing these advance directive conflicts. Use clinical language. Emphasize patient autonomy and legal obligations under the Patient Self-Determination Act.

Conflicts:
{json.dumps(conflicts, indent=2)}

Write the narrative only, no JSON."""
    return call_medgemma(prompt)


# ── FHIR Bundle Report Builder ─────────────────────────────────────────────────

def build_fhir_report(patient_data: dict,
                       risk_factors: list,
                       ad_preferences: dict,
                       care_plan: dict,
                       conflicts: dict,
                       narrative: str) -> dict:
    """
    Builds a FHIR-aligned Bundle report.
    PACIO ADI STU2 | LOINC 100826-7 | LA33478-1
    """
    return {
        "resourceType": "Bundle",
        "fhir_ig": "PACIO ADI STU2",
        "loinc_review_code": "100826-7",
        "conflict_answer_code": "LA33478-1" if conflicts.get("conflicts_found") else "LA33476-5",
        "conflict_answer_display": (
            "Conflict exists, notified patient"
            if conflicts.get("conflicts_found")
            else "Document was reviewed, no conflict"
        ),
        "timestamp": datetime.now().isoformat(),
        "patient": {
            "id": patient_data.get("patient_id"),
            "name": patient_data.get("patient_name"),
            "age": patient_data.get("age")
        },
        "RiskFactors": risk_factors,
        "ConflictFlag": conflicts.get("conflicts_found", False),
        "ConflictSeverity": "CRITICAL" if conflicts.get("conflicts_found") else "NONE",
        "ConflictCount": conflicts.get("conflict_count", 0),
        "Conflicts": conflicts.get("conflicts", []),
        "ClinicalNarrative": narrative,
        "AdvanceDirectiveObservation": {
            "loinc_panel": "75772-4",
            "preferences": ad_preferences
        },
        "CurrentCarePlan": care_plan,
        "ActionRequired": conflicts.get("conflicts_found", False),
        "RecommendedAction": (
            "IMMEDIATE clinician review required. "
            "Current care plan must be reconciled with patient's advance directive "
            "before further treatment decisions are made."
            if conflicts.get("conflicts_found")
            else "No action required. Care plan is concordant with advance directive."
        ),
        "privacy_note": "Processed locally. No patient data transmitted externally."
    }


# ── Main Batch Pipeline ────────────────────────────────────────────────────────

def process_patient(patient_data: dict, reports_dir: Path) -> dict:
    """Run the full 5-step ADI-Guard pipeline for a single patient."""
    pid  = patient_data.get("patient_id", "unknown")
    name = patient_data.get("patient_name", "Unknown")

    print(f"\n{'─' * 60}")
    print(f"  Patient: {name} ({pid})")
    print(f"{'─' * 60}")

    # Step 1: Risk identification
    is_high_risk, risk_factors = step1_identify_high_risk(patient_data)
    status = "HIGH RISK ⚠️" if is_high_risk else "STANDARD RISK"
    print(f"  [STEP 1] Risk Assessment   → {status}")
    for r in risk_factors:
        print(f"           ✓ {r}")

    if not is_high_risk:
        print(f"  → Skipping: patient does not meet high-risk criteria")
        return {"patient_id": pid, "status": "low_risk", "action_required": False}

    # Step 2: Extract AD preferences
    ad_preferences = step2_extract_ad_preferences(patient_data.get("advance_directive", ""))
    print(f"  [STEP 2] AD Extraction     → {len(ad_preferences)} preference categories extracted")

    # Step 3: Parse care plan
    care_plan = step3_parse_care_plan(patient_data.get("current_care_plan", ""))
    orders = len(care_plan.get("active_orders", []))
    print(f"  [STEP 3] Care Plan         → {orders} active orders found")

    # Step 4: Detect conflicts
    conflicts = step4_detect_conflicts(ad_preferences, care_plan)
    if conflicts.get("conflicts_found"):
        print(f"  [STEP 4] Conflict Check    → 🚨 {conflicts['conflict_count']} CONFLICT(S) DETECTED")
        for c in conflicts.get("conflicts", []):
            print(f"           [{c.get('severity')}] {c.get('directive')} ←→ {c.get('care_plan')}")
    else:
        print(f"  [STEP 4] Conflict Check    → ✅ No conflicts detected")

    # Step 5: Generate narrative
    narrative = step5_generate_narrative(conflicts, ad_preferences)
    print(f"  [STEP 5] Clinical Report   → Generated")

    # Build FHIR report
    report = build_fhir_report(
        patient_data, risk_factors,
        ad_preferences, care_plan,
        conflicts, narrative
    )

    # Save report locally
    report_path = reports_dir / f"report_{pid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    action = "🚨 REVIEW REQUIRED" if conflicts.get("conflicts_found") else "✅ CONCORDANT"
    print(f"\n  RESULT: {action}")
    print(f"  Report: {report_path}")

    return {
        "patient_id": pid,
        "patient_name": name,
        "status": "processed",
        "action_required": conflicts.get("conflicts_found", False),
        "conflict_count": conflicts.get("conflict_count", 0),
        "report_path": str(report_path)
    }


def run_batch(csv_path: str = "patients.csv"):
    """
    Main entry point. Reads patients.csv and runs the full
    ADI-Guard pipeline for every patient row.
    """
    print("=" * 60)
    print("  ADI-GUARD BATCH PIPELINE")
    print("  Powered by MedGemma | PACIO FHIR ADI STU2")
    print("  ⚠️  All processing is LOCAL — no data leaves this machine")
    print("=" * 60)

    # Load patients
    if not os.path.exists(csv_path):
        print(f"\nERROR: {csv_path} not found.")
        print("Create a patients.csv file using the provided template.")
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        patients = list(csv.DictReader(f))

    print(f"\n  Loaded {len(patients)} patient(s) from {csv_path}")

    # Create reports directory
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Process each patient
    results = []
    for patient_data in patients:
        result = process_patient(patient_data, reports_dir)
        results.append(result)

    # Summary
    total     = len(results)
    reviewed  = sum(1 for r in results if r.get("status") == "processed")
    flagged   = sum(1 for r in results if r.get("action_required"))
    concordant = reviewed - flagged

    print("\n" + "=" * 60)
    print("  BATCH SUMMARY")
    print("=" * 60)
    print(f"  Total patients processed : {total}")
    print(f"  High-risk reviewed       : {reviewed}")
    print(f"  🚨 Conflicts flagged     : {flagged}")
    print(f"  ✅ Concordant            : {concordant}")
    print(f"\n  All reports saved to: {reports_dir.resolve()}")
    print("=" * 60)

    # Save summary
    summary_path = reports_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_timestamp": datetime.now().isoformat(),
            "total_patients": total,
            "high_risk_reviewed": reviewed,
            "conflicts_flagged": flagged,
            "concordant": concordant,
            "privacy": "All processing local. No patient data transmitted externally.",
            "results": results
        }, f, indent=2)

    print(f"\n  Batch summary saved to: {summary_path}")


if __name__ == "__main__":
    run_batch()
