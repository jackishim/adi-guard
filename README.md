# ADI-Guard 🏥
### Agentic Advance Directive Intelligence for Goal-Concordant Care
**MedGemma Impact Challenge — Google Research**

---

## The Problem

Every day, patients receive care that directly contradicts what they legally documented they wanted. A 78-year-old with a DNR order gets resuscitated. A patient who refused mechanical ventilation gets intubated. Not because doctors don't care — but because the system fails to connect the document to the decision.

**65-76% of physicians are entirely unaware when their patient has an advance directive.** Among patients admitted to emergency departments, only **4.8%** have an accessible advance care plan. When patients transfer between facilities, their documented wishes routinely disappear.

Tools like MyDirectives solve the storage problem — they make the document accessible. **ADI-Guard solves what comes next:** the moment a doctor is about to make a treatment decision and needs to know immediately whether it violates their patient's legal wishes.

---

## The Solution

ADI-Guard is an intelligent clinical layer powered by **MedGemma** that sits between the doctor and the patient's advance directive. Before any treatment is administered, the doctor checks with ADI-Guard. It reads the full AD document, compares it against the proposed care plan, and flags every conflict — in real time, entirely on local infrastructure.

**No patient data leaves the hospital network. Ever.**

---

## Architecture

```
Doctor proposes treatment
        │
        ▼
┌─────────────────────────┐
│  Patient Lookup         │  Query AD registry / EHR / MyDirectives
│  FHIR RESTful API       │  GET /DocumentReference?patient={id}
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  AD Extraction          │  MedGemma reads full document
│  MedGemma NLP           │  Extracts structured preferences
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Conflict Detection     │  MedGemma compares AD vs proposed treatment
│  MedGemma Reasoning     │  Maps to LOINC + SNOMED-CT codes
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Clinical Alert         │  FHIR Bundle report generated locally
│  Clinician Review       │  Doctor makes the final decision
└─────────────────────────┘
```

---

## Three Ways to Use ADI-Guard

### 1. Pre-Treatment Clinical Check (`adi_guard_complete.py`)
The primary interface. Doctor enters a patient name and proposed treatment. ADI-Guard reads the full advance directive and returns a conflict report before any treatment is administered.

```
Enter patient name or ID: Frank Morrison
Proposed treatment: Full resuscitation, mechanical ventilation, IV nutrition support

[1/3] Reading advance directive with MedGemma...
[2/3] Comparing proposed treatment against advance directive...
[3/3] Generating clinical alert...

🚨 2 CRITICAL CONFLICTS DETECTED

CONFLICT 1 [CRITICAL]
Patient's AD    : Do NOT want mechanical ventilation
Proposed Action : Mechanical ventilation
LOINC           : 81351-9
Legal Risk      : Violation of patient autonomy

CONFLICT 2 [CRITICAL]
Patient's AD    : Do NOT want CPR / DNR on file
Proposed Action : Full resuscitation protocol
LOINC           : 81351-9
Legal Risk      : Potential Patient Self-Determination Act violation

MEDGEMMA CLINICAL ALERT:
URGENT: Patient's advance directive explicitly prohibits mechanical
ventilation and CPR. Proceeding will directly conflict with documented
wishes and may violate the Patient Self-Determination Act.
```

### 2. Batch Pipeline (`adi_guard_batch.py`)
Runs the full pipeline on every patient in `patients.csv`. A hospital can run this every morning across all admitted patients and catch conflicts before rounds begin.

```
python adi_guard_batch.py

Loaded 5 patient(s) from patients.csv
🚨 Conflicts flagged : 3
✅ Concordant        : 2
All reports saved to: /reports/
```

### 3. Single Patient Pipeline (`adi_guard.py`)
The original five-step agentic pipeline demonstrating the full MedGemma workflow for a single patient case.

---

## FHIR Alignment

Built to the **HL7 FHIR Advance Healthcare Directive Interoperability (ADI) Implementation Guide STU2**, developed by the PACIO Community with support from CMS and ONC.

| Component | Standard |
|-----------|----------|
| Output format | FHIR Bundle resource |
| AD preference panel | LOINC 75772-4 |
| DNR status | LOINC 81351-9 / SNOMED 143021000119109 |
| Intubation preference | LOINC 75787-2 |
| Nutrition preference | LOINC 75788-0 |
| Review code | LOINC 100826-7 |
| Conflict flag | LA33478-1 "Conflict exists, notified patient" |

---

## Setup & Installation

### 1. Install dependencies
```bash
pip install transformers torch accelerate huggingface_hub
```

### 2. Get MedGemma access
- Visit: https://huggingface.co/google/medgemma-4b-it
- Accept the license agreement
- Authenticate: `huggingface-cli login`

### 3. Enable the model
In `adi_guard_complete.py`, the model loads automatically. 
In `adi_guard.py` and `adi_guard_batch.py`, uncomment lines 23-30 
(the model loading block) and lines 42-48 (inside `call_medgemma`).


### 4. Run
```bash
# Pre-treatment clinical check (primary interface)
python adi_guard_complete.py

# Batch processing from CSV
python adi_guard_batch.py

# Single patient demo
python adi_guard.py
```

### Google Colab (Recommended for Demo)
See the included Colab notebook. Load MedGemma on a T4 GPU and run `adi_guard_complete.py` as a cell. Free tier works with Colab Pro.

---

## Privacy & Security

ADI-Guard is designed for deployment on local hospital infrastructure:

- All MedGemma inference runs on-device
- No patient data is transmitted to external servers
- No cloud API calls during inference
- HIPAA-compliant by architecture
- Reports saved locally in FHIR-aligned JSON format

---

## patients.csv Format

Hospitals add patients by filling in one row per patient:

| Column | Description |
|--------|-------------|
| patient_id | Unique identifier |
| patient_name | Full name |
| age | Integer |
| icu_admission | True/False |
| chronic_conditions | Comma-separated list |
| six_month_prognosis | True/False |
| advance_directive | Full text of AD document |
| current_care_plan | Full text of active orders |

---

## Impact

| Metric | Value |
|--------|-------|
| Physicians unaware of patient's AD | 65-76% |
| Emergency patients without accessible AD | 95.2% |
| Time saved vs. manual chart review | ~115 min/patient |
| Institutional cost savings (reported) | Up to 25% |
| Penn Medicine ACP conversation increase | 4x |

---

## Team

**Jack Saleeby** — Developer & Technical Lead
AI engineer and quantitative researcher. Built the MedGemma pipeline, agentic architecture, and FHIR-aligned output structure.

**Dr. Trish Saleeby** — Domain Expert & Clinical Validator
Social work researcher specializing in health interoperability, care coordination, and advance care planning across marginalized populations. Active participant in PACIO FHIR ADI interoperability community. Validated all clinical workflow logic and output accuracy.

---

## References

- Harrison et al. (2016). Low Completion and Disparities in Advance Care Planning. *JAMA Internal Medicine*, 176(12).
- Wilkinson, Wenger & Shugarman (2007). Literature Review on Advance Directives. RAND/HHS.
- Portanova & Enguidanos. Ethnic Differences in Advance Directive Completion. *Journal of Pain and Symptom Management*, 51(2).
- Shankar, Devi & Xu (2026). Effectiveness of digital tools in promoting advance care planning. *Frontiers in Medicine*, 12.
- PACIO Community. HL7 FHIR Advance Healthcare Directive Interoperability IG STU2. https://build.fhir.org/ig/HL7/fhir-pacio-adi/
- Yadav et al. Epidemiology of advance directives in a large patient population. *JAMA Internal Medicine*.
- Penn Medicine AI-Driven ACP Intervention. Quadrupling ACP conversations in oncology.
- NYU Langone Health. Real-Time Mortality Risk Alerts for ACP.
