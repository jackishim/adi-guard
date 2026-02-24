# ADI-Guard 🏥
### Agentic Advance Directive Intelligence for Goal-Concordant Care
**MedGemma Impact Challenge — Google Research**

---

## The Problem

Between **65-76% of physicians whose patients have an advance directive are entirely unaware of its existence.** Among patients admitted to emergency departments, only **4.8%** have an accessible advance care plan. When patients are transferred between facilities, their documented end-of-life wishes routinely disappear — leading to patients receiving care that directly contradicts what they legally documented they wanted.

This is not a documentation problem. It is a **clinical intelligence problem.**

---

## The Solution

ADI-Guard is an agentic AI application built on **MedGemma** that autonomously:

1. **Identifies** high-risk patients who need advance directive review
2. **Extracts** structured preferences from unstructured AD documents (paper, PDF, legacy formats)
3. **Detects** internal inconsistencies within the directive itself
4. **Cross-references** the advance directive against the patient's active care plan
5. **Generates** a structured FHIR-aligned discrepancy report for clinician review

The human clinician always makes the final call. ADI-Guard ensures nothing falls through the cracks.

---

## Architecture

```
Patient Data + AD Document
        │
        ▼
┌───────────────────────┐
│  STEP 1: Risk ID      │  MedGemma analyzes patient history
│  High-risk flagging   │  age, diagnosis, prognosis
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  STEP 2: AD Extraction│  MedGemma parses unstructured text
│  NLP preference parse │  → structured FHIR resources
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  STEP 3: Care Plan    │  FHIR RESTful API query
│  EHR retrieval        │  GET /CarePlan?patient={id}&status=active
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  STEP 4: Conflict     │  MedGemma detects discrepancies
│  Detection            │  Maps to LOINC + SNOMED-CT codes
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│  STEP 5: FHIR Report  │  Structured Bundle output
│  Clinician alert      │  LOINC 100826-7 / LA33478-1
└───────────────────────┘
```

---

## FHIR Alignment

Built to the **HL7 FHIR Advance Healthcare Directive Interoperability (ADI) Implementation Guide STU2**, developed by the PACIO Community with support from CMS and ONC.

| Component | Standard |
|-----------|----------|
| Output format | FHIR Bundle resource |
| AD preferences | LOINC panel 75772-4 |
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
- Accept the license agreement (requires HuggingFace account)
- Authenticate: `huggingface-cli login`

### 3. Enable the model in adi_guard.py
Uncomment the model loading block at the top of `adi_guard.py`:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "google/medgemma-4b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```
And uncomment the `call_medgemma()` production block.

### 4. Run the demo
```bash
python adi_guard.py
```

---

## Example Output

```
============================================================
  ADI-GUARD: Advance Directive Intelligence Agent
  Powered by MedGemma | PACIO FHIR ADI STU2 Aligned
============================================================

[STEP 1] Identifying high-risk patient...
  Status: HIGH RISK
  ✓ Age >= 65
  ✓ ICU admission
  ✓ Chronic conditions: end-stage kidney disease, dialysis-dependent
  ✓ Poor 6-month prognosis

[STEP 2] Extracting advance directive preferences with MedGemma...
  Extracted 9 preference categories

[STEP 3] Retrieving current care plan from EHR...
  Found 5 active orders

[STEP 4] Detecting conflicts between advance directive and care plan...
  ⚠️  3 conflict(s) detected
  [CRITICAL] DNR - Do Not Resuscitate ←→ Full resuscitation protocol active
  [CRITICAL] NO mechanical ventilation ←→ Mechanical ventilation - ongoing
  [HIGH]     NO artificial nutrition  ←→ IV nutrition support initiated

[STEP 5] Generating clinical discrepancy report...
  Report generated successfully.

============================================================
  🚨 RESULT: CRITICAL CONFLICTS DETECTED — CLINICIAN REVIEW REQUIRED
============================================================
```

---

## Impact

| Metric | Value |
|--------|-------|
| Physicians unaware of patient's AD | 65-76% |
| Emergency patients without accessible AD | 95.2% |
| Time saved vs. manual chart review | ~115 min/patient |
| Institutional cost savings (reported) | Up to 25% |
| Penn Medicine ACP conversation increase | 4x (quadrupled) |

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
