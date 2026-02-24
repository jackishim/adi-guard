"""
ADI-Guard: Agentic Advance Directive Intelligence for Goal-Concordant Care
Built for the MedGemma Impact Challenge (Google Research)

Requirements:
    pip install transformers torch accelerate huggingface_hub

To run:
    python adi_guard.py

MedGemma access:
    1. Go to https://huggingface.co/google/medgemma-4b-it
    2. Accept the license agreement
    3. Run: huggingface-cli login
"""

import json
import re
from datetime import datetime

# ── MedGemma setup ────────────────────────────────────────────────────────────
# Uncomment these when running locally after: pip install transformers torch accelerate
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
#
# MODEL_ID = "google/medgemma-4b-it"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )


def call_medgemma(prompt: str) -> str:
    """
    Call MedGemma with a clinical prompt.
    In production: replace the stub below with the real model call.
    """
    # ── PRODUCTION (uncomment when model is loaded) ──────────────────────────
    # messages = [{"role": "user", "content": prompt}]
    # inputs = tokenizer.apply_chat_template(
    #     messages, return_tensors="pt", add_generation_prompt=True
    # ).to(model.device)
    # with torch.inference_mode():
    #     outputs = model.generate(inputs, max_new_tokens=512, do_sample=False)
    # return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

    # ── DEMO STUB (remove when using real model) ─────────────────────────────
    # Returns realistic outputs so the demo pipeline runs end-to-end
    if "extract" in prompt.lower() and "directive" in prompt.lower():
        return json.dumps({
            "dnr_status": "YES - Do Not Resuscitate",
            "life_support": "NO - Patient refuses life-sustaining treatment",
            "cpr_preference": "NO CPR",
            "intubation": "NO - Refuses mechanical ventilation",
            "artificial_nutrition": "NO - Refuses tube feeding",
            "resuscitation": "NO - Do Not Attempt Resuscitation (DNAR)",
            "healthcare_agent": "Mary Johnson (daughter) - primary",
            "palliative_care": "YES - Comfort measures only",
            "hospital_transfer": "NO - Prefers to remain in current facility"
        })
    elif "care plan" in prompt.lower() or "current treatment" in prompt.lower():
        return json.dumps({
            "active_orders": [
                "Full resuscitation protocol active",
                "Mechanical ventilation - ongoing",
                "IV nutrition support initiated",
                "ICU escalation plan in place",
                "CPR order: FULL CODE"
            ],
            "treating_team": "ICU - Dr. Patel",
            "admission_date": "2026-02-20",
            "primary_diagnosis": "Acute respiratory failure"
        })
    elif "inconsistenc" in prompt.lower() or "conflict" in prompt.lower():
        return json.dumps({
            "conflicts_found": True,
            "conflict_count": 3,
            "conflicts": [
                {
                    "severity": "CRITICAL",
                    "directive": "DNR - Do Not Resuscitate",
                    "care_plan": "Full resuscitation protocol active / CPR order: FULL CODE",
                    "loinc_code": "81351-9",
                    "snomed_code": "143021000119109",
                    "explanation": "Patient explicitly refused CPR. Current order mandates full resuscitation."
                },
                {
                    "severity": "CRITICAL",
                    "directive": "NO mechanical ventilation",
                    "care_plan": "Mechanical ventilation - ongoing",
                    "loinc_code": "75787-2",
                    "snomed_code": "425921008",
                    "explanation": "Patient refused intubation and mechanical ventilation. Ventilator currently in use."
                },
                {
                    "severity": "HIGH",
                    "directive": "NO artificial nutrition",
                    "care_plan": "IV nutrition support initiated",
                    "loinc_code": "75788-0",
                    "snomed_code": "415068001",
                    "explanation": "Patient refused tube feeding and artificial nutrition. IV nutrition support active."
                }
            ]
        })
    elif "report" in prompt.lower() or "summar" in prompt.lower():
        return (
            "URGENT CLINICAL ALERT: This patient has a documented Advance Directive "
            "that directly conflicts with their current care plan. Three critical discrepancies "
            "were identified: (1) A DNR order exists but a Full Code resuscitation protocol is "
            "active. (2) The patient refused mechanical ventilation but is currently intubated. "
            "(3) The patient refused artificial nutrition but IV nutrition support has been "
            "initiated. Immediate clinician review is required to align care with the patient's "
            "documented wishes. Failure to act may constitute a violation of patient autonomy "
            "and legal rights under the Patient Self-Determination Act."
        )
    return "Unable to process prompt."


# ── FHIR-aligned data structures ──────────────────────────────────────────────

def build_fhir_advance_directive_observation(extracted: dict) -> dict:
    """
    Structures extracted AD preferences as FHIR-aligned resources.
    Maps to HL7 FHIR ADI Implementation Guide (STU2) / PACIO Community standards.
    LOINC panel: 75772-4 (Advance Directive panel)
    """
    return {
        "resourceType": "AdvanceDirectiveObservation",
        "fhir_ig": "PACIO ADI STU2",
        "loinc_panel": "75772-4",
        "status": "final",
        "effectiveDateTime": datetime.now().isoformat(),
        "PersonalInterventionPreferences": {
            "dnr_status": {
                "value": extracted.get("dnr_status"),
                "loinc": "81351-9",
                "snomed": "143021000119109"
            },
            "life_support": {
                "value": extracted.get("life_support"),
                "loinc": "75789-8"
            },
            "cpr_preference": {
                "value": extracted.get("cpr_preference"),
                "loinc": "75779-9"
            },
            "intubation": {
                "value": extracted.get("intubation"),
                "loinc": "75787-2"
            },
            "artificial_nutrition": {
                "value": extracted.get("artificial_nutrition"),
                "loinc": "75788-0"
            },
            "resuscitation": {
                "value": extracted.get("resuscitation"),
                "loinc": "81329-5"
            }
        },
        "CareExperiencePreferences": {
            "palliative_care": extracted.get("palliative_care"),
            "hospital_transfer": extracted.get("hospital_transfer"),
            "loinc": "81380-8"
        },
        "HealthcareAgent": {
            "primary_agent": extracted.get("healthcare_agent"),
            "loinc": "75783-1"
        }
    }


def build_conflict_report(patient_id: str,
                           patient_name: str,
                           conflicts: dict,
                           narrative: str,
                           ad_resource: dict,
                           care_plan: dict) -> dict:
    """
    Generates a FHIR Bundle containing the full discrepancy report.
    Maps to LOINC 100826-7: Portable medical order & advance directive review.
    Conflict answer: LA33478-1 'Conflict exists, notified patient'
    """
    return {
        "resourceType": "Bundle",
        "fhir_ig": "PACIO ADI STU2",
        "type": "document",
        "timestamp": datetime.now().isoformat(),
        "meta": {
            "loinc_review_code": "100826-7",
            "conflict_answer_code": "LA33478-1",
            "conflict_answer_display": "Conflict exists, notified patient"
        },
        "patient": {
            "id": patient_id,
            "name": patient_name
        },
        "ConflictFlag": conflicts.get("conflicts_found", False),
        "ConflictSeverity": "CRITICAL" if conflicts.get("conflicts_found") else "NONE",
        "ConflictCount": conflicts.get("conflict_count", 0),
        "Conflicts": conflicts.get("conflicts", []),
        "ClinicalNarrative": narrative,
        "AdvanceDirectiveObservation": ad_resource,
        "CurrentCarePlan": care_plan,
        "ActionRequired": conflicts.get("conflicts_found", False),
        "RecommendedAction": (
            "IMMEDIATE clinician review required. Current care plan must be "
            "reconciled with patient's advance directive before further "
            "treatment decisions are made."
            if conflicts.get("conflicts_found")
            else "No action required. Care plan is concordant with advance directive."
        )
    }


# ── Agent Pipeline ─────────────────────────────────────────────────────────────

def step1_identify_high_risk(patient_data: dict) -> bool:
    """
    Step 1: Identify whether this patient is high-risk and needs AD review.
    In production: MedGemma analyzes patient history, diagnosis, age, prognosis.
    """
    print("\n[STEP 1] Identifying high-risk patient...")
    risk_factors = []

    if patient_data.get("age", 0) >= 65:
        risk_factors.append("Age >= 65")
    if patient_data.get("icu_admission"):
        risk_factors.append("ICU admission")
    if patient_data.get("chronic_conditions"):
        risk_factors.append(f"Chronic conditions: {', '.join(patient_data['chronic_conditions'])}")
    if patient_data.get("six_month_prognosis"):
        risk_factors.append("Poor 6-month prognosis")

    is_high_risk = len(risk_factors) >= 2
    status = "HIGH RISK" if is_high_risk else "STANDARD RISK"
    print(f"  Status: {status}")
    for r in risk_factors:
        print(f"  ✓ {r}")
    return is_high_risk


def step2_extract_ad_preferences(ad_text: str) -> dict:
    """
    Step 2: Use MedGemma to extract structured preferences from AD document text.
    Works on unstructured paper-based documents, PDFs, legacy formats.
    """
    print("\n[STEP 2] Extracting advance directive preferences with MedGemma...")

    prompt = f"""
You are a clinical AI assistant. Extract and structure the patient's advance directive 
preferences from the following document. Return a JSON object with these keys:
dnr_status, life_support, cpr_preference, intubation, artificial_nutrition, 
resuscitation, healthcare_agent, palliative_care, hospital_transfer.

Advance Directive Document:
{ad_text}

Return ONLY valid JSON, no explanation.
"""
    raw = call_medgemma(prompt)

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = {"raw_extraction": raw}

    print(f"  Extracted {len(extracted)} preference categories")
    return extracted


def step3_get_care_plan(patient_id: str) -> dict:
    """
    Step 3: Retrieve current active care plan from EHR.
    In production: FHIR RESTful API call to EHR system.
    GET [base]/CarePlan?patient={patient_id}&status=active
    """
    print("\n[STEP 3] Retrieving current care plan from EHR...")

    prompt = f"""
Retrieve the current active care plan and treatment orders for patient {patient_id}.
Return a JSON object with keys: active_orders (list), treating_team, 
admission_date, primary_diagnosis.
Return ONLY valid JSON.
"""
    raw = call_medgemma(prompt)

    try:
        care_plan = json.loads(raw)
    except json.JSONDecodeError:
        care_plan = {"raw": raw}

    print(f"  Found {len(care_plan.get('active_orders', []))} active orders")
    return care_plan


def step4_detect_conflicts(ad_preferences: dict, care_plan: dict) -> dict:
    """
    Step 4: Use MedGemma to detect inconsistencies between AD and care plan.
    Maps conflicts to LOINC and SNOMED-CT codes per PACIO ADI STU2.
    """
    print("\n[STEP 4] Detecting conflicts between advance directive and care plan...")

    prompt = f"""
You are a clinical AI assistant performing advance directive compliance review.

PATIENT'S ADVANCE DIRECTIVE PREFERENCES:
{json.dumps(ad_preferences, indent=2)}

CURRENT ACTIVE CARE PLAN:
{json.dumps(care_plan, indent=2)}

Identify all inconsistencies where the current care plan contradicts the patient's 
documented advance directive wishes. For each conflict provide: severity (CRITICAL/HIGH/MEDIUM),
the directive statement, the conflicting care plan order, relevant LOINC code, 
SNOMED code, and a plain-language explanation.

Return ONLY valid JSON with keys: conflicts_found (bool), conflict_count (int), conflicts (list).
"""
    raw = call_medgemma(prompt)

    try:
        conflicts = json.loads(raw)
    except json.JSONDecodeError:
        conflicts = {"conflicts_found": False, "conflict_count": 0, "conflicts": []}

    if conflicts.get("conflicts_found"):
        print(f"  ⚠️  {conflicts['conflict_count']} conflict(s) detected")
        for c in conflicts.get("conflicts", []):
            print(f"  [{c['severity']}] {c['directive']} ←→ {c['care_plan']}")
    else:
        print("  ✓ No conflicts detected. Care plan is concordant with advance directive.")

    return conflicts


def step5_generate_report(patient_id: str,
                           patient_name: str,
                           ad_preferences: dict,
                           care_plan: dict,
                           conflicts: dict) -> dict:
    """
    Step 5: Generate a structured FHIR Bundle report for clinician review.
    Human clinician always makes the final call — AI surfaces the issue only.
    """
    print("\n[STEP 5] Generating clinical discrepancy report...")

    prompt = f"""
You are a clinical documentation assistant. Write a concise, urgent clinical narrative 
(3-4 sentences) summarizing the advance directive conflicts found for a clinician to review.
Be direct, use clinical language, and emphasize patient autonomy and legal obligations.

Conflicts identified:
{json.dumps(conflicts, indent=2)}
"""
    narrative = call_medgemma(prompt)

    ad_resource = build_fhir_advance_directive_observation(ad_preferences)

    report = build_conflict_report(
        patient_id=patient_id,
        patient_name=patient_name,
        conflicts=conflicts,
        narrative=narrative,
        ad_resource=ad_resource,
        care_plan=care_plan
    )

    print("  Report generated successfully.")
    return report


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_adi_guard(patient_data: dict, ad_text: str) -> dict:
    """
    Full ADI-Guard agentic pipeline.
    Runs all 5 steps sequentially and returns the final FHIR Bundle report.
    """
    print("=" * 60)
    print("  ADI-GUARD: Advance Directive Intelligence Agent")
    print("  Powered by MedGemma | PACIO FHIR ADI STU2 Aligned")
    print("=" * 60)

    patient_id   = patient_data["id"]
    patient_name = patient_data["name"]

    # Step 1: Risk identification
    is_high_risk = step1_identify_high_risk(patient_data)
    if not is_high_risk:
        print("\n  Patient is not high-risk. Standard monitoring protocol applies.")
        return {"status": "low_risk", "patient_id": patient_id}

    # Step 2: Extract AD preferences
    ad_preferences = step2_extract_ad_preferences(ad_text)

    # Step 3: Get current care plan
    care_plan = step3_get_care_plan(patient_id)

    # Step 4: Detect conflicts
    conflicts = step4_detect_conflicts(ad_preferences, care_plan)

    # Step 5: Generate report
    report = step5_generate_report(
        patient_id, patient_name,
        ad_preferences, care_plan, conflicts
    )

    print("\n" + "=" * 60)
    if report.get("ConflictFlag"):
        print("  🚨 RESULT: CRITICAL CONFLICTS DETECTED — CLINICIAN REVIEW REQUIRED")
    else:
        print("  ✅ RESULT: CARE PLAN IS CONCORDANT WITH ADVANCE DIRECTIVE")
    print("=" * 60)

    return report


# ── Demo ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Sample patient (based on PACIO ADI use case — Frank)
    patient_data = {
        "id": "patient-001",
        "name": "Frank Morrison",
        "age": 78,
        "icu_admission": True,
        "chronic_conditions": ["end-stage kidney disease", "dialysis-dependent"],
        "six_month_prognosis": True
    }

    # Sample advance directive text (unstructured, as found in real practice)
    advance_directive_text = """
    ADVANCE DIRECTIVE - Frank Morrison - DOB: 03/14/1948

    I, Frank Morrison, being of sound mind, hereby state my wishes for medical care
    if I become unable to communicate or make decisions for myself.

    RESUSCITATION: I do NOT want cardiopulmonary resuscitation (CPR) performed.
    Do Not Resuscitate (DNR). Do Not Attempt Resuscitation (DNAR).

    LIFE-SUSTAINING TREATMENT: I do not want life-sustaining treatment if there is
    no reasonable chance of recovery. This includes mechanical ventilation,
    dialysis beyond my current regimen, and artificial nutrition or hydration.

    COMFORT CARE: I want comfort measures only — pain management and palliative care.
    I do not want to be transferred to a hospital if my condition worsens.
    I wish to remain in my current care setting.

    HEALTHCARE AGENT: My daughter Mary Johnson (555-0142) is my primary healthcare
    agent and has full authority to make medical decisions on my behalf.

    Signed: Frank Morrison    Date: January 10, 2026
    Witness: Dr. Sarah Chen   Notary: James Wilson
    """

    # Run the full pipeline
    report = run_adi_guard(patient_data, advance_directive_text)

    # Save output
    output_path = "adi_guard_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Full FHIR Bundle report saved to: {output_path}")
    print("\n  KEY CONFLICTS FOUND:")
    for conflict in report.get("Conflicts", []):
        print(f"\n  [{conflict['severity']}]")
        print(f"  Directive : {conflict['directive']}")
        print(f"  Care Plan : {conflict['care_plan']}")
        print(f"  LOINC     : {conflict['loinc_code']}")
        print(f"  SNOMED    : {conflict['snomed_code']}")
        print(f"  Reason    : {conflict['explanation']}")

    print("\n  RECOMMENDED ACTION:")
    print(f"  {report.get('RecommendedAction')}")
