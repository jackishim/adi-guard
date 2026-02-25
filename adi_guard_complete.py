"""
ADI-Guard: Complete Clinical Interface
=======================================
Run with: python adi_guard_complete.py

Setup:
    1. pip install transformers torch accelerate huggingface_hub
    2. Visit https://huggingface.co/google/medgemma-4b-it and accept the license
    3. Run: huggingface-cli login
    4. Uncomment the MedGemma Setup block below
"""

import json
import torch
from datetime import datetime

# ── MedGemma Setup ─────────────────────────────────────────────────────────────
# Uncomment after completing setup steps above
#
# from transformers import AutoTokenizer, AutoModelForCausalLM
# MODEL_ID = "google/medgemma-4b-it"
# print("Loading MedGemma (this may take a few minutes)...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )
# print("MedGemma loaded.\n")


# ── MedGemma call ─────────────────────────────────────────────────────────────

def call_medgemma(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


# ── Patient AD Registry ────────────────────────────────────────────────────────
# In production: replaced with FHIR RESTful API call to MyDirectives / EHR
# GET [base]/DocumentReference?patient={name}&type=advance-directive

AD_REGISTRY = {
    "frank morrison": {
        "patient_id": "patient-001",
        "patient_name": "Frank Morrison",
        "age": 78,
        "advance_directive": """
        ADVANCE DIRECTIVE - Frank Morrison - DOB: 03/14/1948

        I, Frank Morrison, being of sound mind, hereby state my wishes for medical
        care if I become unable to communicate or make decisions for myself.

        RESUSCITATION: I do NOT want cardiopulmonary resuscitation (CPR) performed.
        Do Not Resuscitate (DNR). Do Not Attempt Resuscitation (DNAR).

        LIFE-SUSTAINING TREATMENT: I do not want life-sustaining treatment if there
        is no reasonable chance of recovery. This includes mechanical ventilation,
        dialysis beyond my current regimen, and artificial nutrition or hydration.

        COMFORT CARE: I want comfort measures only — pain management and palliative
        care. I do not want to be transferred to a hospital if my condition worsens.

        HEALTHCARE AGENT: My daughter Mary Johnson (555-0142) is my primary
        healthcare agent and has full authority to make medical decisions on my
        behalf.

        Signed: Frank Morrison    Date: January 10, 2026
        Witness: Dr. Sarah Chen   Notary: James Wilson
        """
    },
    "eleanor hayes": {
        "patient_id": "patient-002",
        "patient_name": "Eleanor Hayes",
        "age": 82,
        "advance_directive": """
        ADVANCE DIRECTIVE - Eleanor Hayes

        I want full life-sustaining treatment including CPR and mechanical
        ventilation if needed. I want every measure taken to keep me alive.
        I want to be hospitalized and treated aggressively if my condition worsens.

        Healthcare Agent: Robert Hayes (husband) 555-0188.

        Signed: Eleanor Hayes    Date: March 3, 2025
        """
    },
    "robert okafor": {
        "patient_id": "patient-005",
        "patient_name": "Robert Okafor",
        "age": 85,
        "advance_directive": """
        ADVANCE DIRECTIVE - Robert Okafor

        I do not want any life-prolonging treatment under any circumstances.
        No CPR. No ventilator. No feeding tube. No nasogastric tube.
        No hospitalization if avoidable. No IV nutrition.

        I want to be kept comfortable and pain-free only.
        Comfort and palliative care measures only.

        No healthcare agent designated. My wishes as written are to be followed.

        Signed: Robert Okafor    Date: November 15, 2025
        """
    }
}


# ── Agent Steps ────────────────────────────────────────────────────────────────

def lookup_patient(name_or_id: str):
    key = name_or_id.strip().lower()
    if key in AD_REGISTRY:
        return AD_REGISTRY[key]
    for k, v in AD_REGISTRY.items():
        if key in k or key == v.get("patient_id", "").lower():
            return v
    return None


def extract_ad_preferences(ad_text: str) -> dict:
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


def detect_conflicts(ad_preferences: dict, proposed_treatment: str) -> dict:
    prompt = f"""You are a clinical AI assistant performing advance directive compliance review.

A clinician is proposing the following treatment:
PROPOSED TREATMENT: {proposed_treatment}

The patient's documented advance directive preferences are:
{json.dumps(ad_preferences, indent=2)}

Identify ALL conflicts where the proposed treatment contradicts the patient's documented wishes.

Respond with ONLY this JSON structure, no other text:
{{
  "conflicts_found": true,
  "conflict_count": 2,
  "conflicts": [
    {{
      "severity": "CRITICAL",
      "directive": "exact patient wish from AD",
      "proposed_treatment": "exact conflicting treatment proposed",
      "loinc_code": "81351-9",
      "snomed_code": "143021000119109",
      "explanation": "plain language explanation",
      "legal_risk": "specific legal or ethical risk"
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
        return {
            "conflicts_found": True,
            "conflict_count": 1,
            "conflicts": [{
                "severity": "CRITICAL",
                "directive": "See advance directive",
                "proposed_treatment": proposed_treatment,
                "loinc_code": "100826-7",
                "snomed_code": "143021000119109",
                "explanation": raw[:500],
                "legal_risk": "Review required — potential Patient Self-Determination Act violation"
            }]
        }


def generate_clinical_alert(conflicts: dict, patient_name: str, proposed_treatment: str) -> str:
    if not conflicts.get("conflicts_found"):
        return f"No conflicts detected. Proposed treatment is concordant with {patient_name}'s advance directive."
    prompt = f"""You are a clinical documentation assistant. Write an urgent clinical alert (3-4 sentences) for a physician about to perform: "{proposed_treatment}"

Conflicts with the patient's advance directive:
{json.dumps(conflicts, indent=2)}

Be direct and urgent. Reference patient autonomy and legal obligations under the Patient Self-Determination Act."""
    return call_medgemma(prompt)


def print_report(patient, proposed_treatment, ad_preferences, conflicts, alert):
    print("\n" + "═" * 65)
    print(f"  ADI-GUARD CLINICAL ALERT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 65)
    print(f"\n  Patient           : {patient['patient_name']} (Age {patient['age']})")
    print(f"  Patient ID        : {patient['patient_id']}")
    print(f"  Proposed Treatment: {proposed_treatment}")

    print(f"\n  AD PREFERENCES EXTRACTED BY MEDGEMMA:")
    for k, v in ad_preferences.items():
        if k != "raw_extraction":
            print(f"    {k:<25}: {v}")

    print(f"\n{'─' * 65}")
    if conflicts.get("conflicts_found"):
        print(f"  🚨 {conflicts['conflict_count']} CONFLICT(S) DETECTED")
        print(f"{'─' * 65}")
        for i, c in enumerate(conflicts.get("conflicts", []), 1):
            print(f"\n  CONFLICT {i} [{c.get('severity', 'UNKNOWN')}]")
            print(f"  Patient's AD      : {c.get('directive', 'N/A')}")
            print(f"  Proposed Action   : {c.get('proposed_treatment', 'N/A')}")
            print(f"  LOINC Code        : {c.get('loinc_code', 'N/A')}")
            print(f"  SNOMED Code       : {c.get('snomed_code', 'N/A')}")
            print(f"  Explanation       : {c.get('explanation', 'N/A')}")
            if c.get('legal_risk'):
                print(f"  Legal Risk        : {c.get('legal_risk')}")
    else:
        print(f"  ✅ NO CONFLICTS — Treatment concordant with advance directive")

    print(f"\n{'─' * 65}")
    print(f"  CLINICAL ALERT (MedGemma):")
    print(f"{'─' * 65}")
    print(f"\n  {alert}\n")
    print("═" * 65)
    print(f"  ⚠️  Processed locally. No patient data transmitted externally.")
    print(f"  ⚠️  Clinician makes the final decision.")
    print("═" * 65)


# ── Main Interface ─────────────────────────────────────────────────────────────

def run_clinical_check():
    print("═" * 65)
    print("  ADI-GUARD: Pre-Treatment Advance Directive Check")
    print("  Powered by MedGemma | PACIO FHIR ADI STU2 Aligned")
    print("  All processing LOCAL — no data leaves this machine")
    print("═" * 65)

    print("\nAvailable demo patients: Frank Morrison, Eleanor Hayes, Robert Okafor")
    patient_input = input("\nEnter patient name or ID: ").strip()
    patient = lookup_patient(patient_input)

    if not patient:
        print(f"\n  ❌ No advance directive found for '{patient_input}'")
        print(f"  Patient may not have a registered AD or name may be incorrect.")
        print(f"  Recommend initiating advance care planning conversation.")
        return

    print(f"\n  ✓ Advance directive found for {patient['patient_name']}")
    print(f"  AD document length: {len(patient['advance_directive'])} characters")

    print(f"\nDescribe the proposed treatment for {patient['patient_name']}.")
    print("Example: 'Full resuscitation, mechanical ventilation, IV nutrition support'")
    proposed_treatment = input("\nProposed treatment: ").strip()

    if not proposed_treatment:
        print("  No treatment entered.")
        return

    print(f"\n  [1/3] Reading advance directive with MedGemma...")
    ad_preferences = extract_ad_preferences(patient["advance_directive"])
    print(f"  ✓ Extracted {len(ad_preferences)} preference categories")

    print(f"  [2/3] Comparing proposed treatment against advance directive...")
    conflicts = detect_conflicts(ad_preferences, proposed_treatment)
    status = f"🚨 {conflicts['conflict_count']} conflict(s)" if conflicts.get("conflicts_found") else "✅ No conflicts"
    print(f"  ✓ Conflict check complete: {status}")

    print(f"  [3/3] Generating clinical alert...")
    alert = generate_clinical_alert(conflicts, patient["patient_name"], proposed_treatment)
    print(f"  ✓ Alert generated")

    print_report(patient, proposed_treatment, ad_preferences, conflicts, alert)

    report = {
        "resourceType": "Bundle",
        "fhir_ig": "PACIO ADI STU2",
        "loinc_review_code": "100826-7",
        "timestamp": datetime.now().isoformat(),
        "patient": {"id": patient["patient_id"], "name": patient["patient_name"]},
        "proposed_treatment": proposed_treatment,
        "ConflictFlag": conflicts.get("conflicts_found", False),
        "ConflictCount": conflicts.get("conflict_count", 0),
        "Conflicts": conflicts.get("conflicts", []),
        "ClinicalAlert": alert,
        "ADPreferences": ad_preferences,
        "privacy_note": "Processed locally. No patient data transmitted externally."
    }
    fname = f"adi_guard_alert_{patient['patient_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved locally: {fname}")


run_clinical_check()
