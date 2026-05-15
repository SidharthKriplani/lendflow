"""
LendFlow — Synthetic Test Data Generator
Generates 20 realistic financial documents (text format) with ground truth JSON.
Run: python scripts/generate_test_data.py

Output:
  tests/fixtures/applications/APP_xx.txt       — raw document text
  tests/fixtures/ground_truth/APP_xx_gt.json   — ground truth labels
"""
import json
import random
from pathlib import Path

random.seed(42)

APPS_DIR = Path("tests/fixtures/applications")
GT_DIR   = Path("tests/fixtures/ground_truth")
APPS_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fake_pan():    return f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}{random.randint(1000,9999)}{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=1))}"
def fake_aadhaar():return f"{random.randint(1000,9999)} {random.randint(1000,9999)} {random.randint(1000,9999)}"
def fake_phone():  return f"+91 {random.choice([6,7,8,9])}{''.join([str(random.randint(0,9)) for _ in range(9)])}"
def fake_ifsc():   return f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))}0{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))}"
def inr(n):        return f"₹{n:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
# BANK STATEMENTS (5)
# ══════════════════════════════════════════════════════════════════════════════

bank_cases = [
    # (label, income, emi, employment_type, red_flags, routing, reason_key)
    ("SALARIED_CLEAR",   85000, 22000, "SALARIED",     [],                           "APPROVE",  "All fields verified"),
    ("SALARIED_HI_FOIR", 60000, 36000, "SALARIED",     [],                           "REJECT",   "FOIR exceeds threshold"),
    ("SELF_EMP_EDGE",    55000, 26500, "SELF_EMPLOYED", [],                           "REJECT",   "FOIR exceeds 0.50 limit for self-employed"),
    ("THIN_FILE",        40000,  8000, "UNKNOWN",       [],                           "ESCALATE", "Employment type unverifiable"),
    ("RED_FLAG",         90000, 18000, "SALARIED",     ["round-trip transactions detected", "large cash withdrawal spike"], "REJECT", "Suspicious transaction patterns"),
]

for i, (label, income, emi, emp_type, flags, routing, reason) in enumerate(bank_cases, 1):
    foir = round(emi / income, 3)
    app_id = f"APP_{i:02d}"
    name = random.choice(["Rahul Sharma", "Priya Mehta", "Arjun Nair", "Sunita Patel", "Vikram Singh"])
    pan  = fake_pan(); aadhaar = fake_aadhaar(); phone = fake_phone(); ifsc = fake_ifsc()

    doc = f"""BANK ACCOUNT STATEMENT
Account Holder: {name}
PAN: {pan}
Mobile: {phone}
IFSC Code: {ifsc}
Account Number: {random.randint(10000000000, 99999999999)}
Branch: Mumbai Main Branch
Statement Period: January 2026 - March 2026

TRANSACTION SUMMARY
Opening Balance (Jan 1):  {inr(random.randint(5000, 30000))}
Total Credits (3 months): {inr(income * 3)}
Total Debits  (3 months): {inr((income - emi) * 3 + emi * 3)}
Closing Balance (Mar 31): {inr(random.randint(5000, 40000))}

Average Monthly Credits: {inr(income)}
Average Monthly Debits:  {inr(income - random.randint(2000, 8000))}
Estimated Monthly Income: {inr(income)}
Identified EMI Outflows:  {inr(emi)}
FOIR (calculated): {foir:.2%}
Employment Type (inferred): {emp_type}
Account Vintage: 36 months

SELECTED TRANSACTIONS
Date        | Description                      | Debit     | Credit
------------|----------------------------------|-----------|----------
01 Jan 2026 | SALARY CREDIT - {emp_type[:4]}  |           | {inr(income)}
05 Jan 2026 | EMI - HOME LOAN                  | {inr(emi//2)}  |
10 Jan 2026 | EMI - PERSONAL LOAN              | {inr(emi//2)}  |
{f'15 Jan 2026 | CASH WITHDRAWAL - UNUSUAL        | ' + inr(income * 0.8) + '  |' if "cash withdrawal" in str(flags) else '15 Jan 2026 | UTILITY PAYMENT                  | ' + inr(random.randint(500, 3000)) + '   |'}
{f'20 Jan 2026 | TRANSFER IN + TRANSFER OUT SAME  | ' + inr(income * 0.5) + '  | ' + inr(income * 0.5) if "round-trip" in str(flags) else '20 Jan 2026 | GROCERY / DAILY EXPENSES         | ' + inr(random.randint(3000, 8000)) + '   |'}
01 Feb 2026 | SALARY CREDIT - {emp_type[:4]}  |           | {inr(income)}

{f"RED FLAGS DETECTED: {'; '.join(flags)}" if flags else "No anomalies detected."}
"""

    gt = {
        "application_id": app_id,
        "doc_type": "bank_statement",
        "label": label,
        "extracted_fields": {
            "estimated_monthly_income": income,
            "emi_obligations": emi,
            "foir": foir,
            "employment_type": emp_type,
            "cash_flow_volatility": "HIGH" if flags else "LOW",
            "red_flags": flags,
            "account_vintage_months": 36,
        },
        "expected_routing": routing,
        "primary_reason": reason,
        "critical_policy_rules": ["FOIR_LIMIT", "RED_FLAGS"] if flags else ["FOIR_LIMIT"],
    }

    (APPS_DIR / f"{app_id}.txt").write_text(doc)
    (GT_DIR / f"{app_id}_gt.json").write_text(json.dumps(gt, indent=2))
    print(f"✓ {app_id} — {label} — {routing}")


# ══════════════════════════════════════════════════════════════════════════════
# SALARY SLIPS (5)
# ══════════════════════════════════════════════════════════════════════════════

salary_cases = [
    ("SALARIED_HIGH",   150000, 45000, "SALARIED",  "Software Engineer",      "APPROVE"),
    ("SALARIED_MID",     65000, 20000, "SALARIED",  "Operations Manager",     "APPROVE"),
    ("CONTRACT_UNCLEAR", 48000, 24000, "CONTRACT",  "Consultant",             "ESCALATE"),
    ("SALARIED_LOW",     28000,  8000, "SALARIED",  "Junior Executive",       "ESCALATE"),
    ("SALARIED_NOEMP",   55000,     0, "SALARIED",  "Data Analyst",           "APPROVE"),
]

for i, (label, gross, emi, emp_type, designation, routing) in enumerate(salary_cases, 1):
    app_id = f"APP_{i+5:02d}"
    net = round(gross * 0.78)
    name = random.choice(["Kavya Reddy", "Manish Joshi", "Deepa Iyer", "Rohan Gupta", "Anita Desai"])
    employer = random.choice(["TechCorp India Pvt Ltd", "Infosys BPM", "HDFC Bank", "Reliance Industries", "Wipro Ltd"])
    pan = fake_pan()

    doc = f"""SALARY SLIP — {random.choice(['January','February','March'])} 2026

Employee Name: {name}
Employee ID:   EMP{random.randint(10000, 99999)}
Designation:   {designation}
Department:    {random.choice(['Engineering', 'Operations', 'Finance', 'HR', 'Analytics'])}
PAN:           {pan}
Bank Account:  ****{random.randint(1000,9999)}

Employer: {employer}
Employment Type: {emp_type}

EARNINGS
Basic Pay:              {inr(gross * 0.40)}
House Rent Allowance:   {inr(gross * 0.20)}
Special Allowance:      {inr(gross * 0.25)}
Medical Allowance:      {inr(gross * 0.05)}
Transport Allowance:    {inr(gross * 0.05)}
Other Allowances:       {inr(gross * 0.05)}
━━━━━━━━━━━━━━━━━━━━━
GROSS SALARY:           {inr(gross)}

DEDUCTIONS
Provident Fund (12%):   {inr(gross * 0.12)}
TDS:                    {inr(gross * 0.05)}
Professional Tax:       {inr(200)}
{'EMI Recovery:           ' + inr(emi) if emi > 0 else 'No loan deductions.'}
━━━━━━━━━━━━━━━━━━━━━
TOTAL DEDUCTIONS:       {inr(gross * 0.17 + (emi if emi > 0 else 0))}

NET TAKE-HOME PAY:      {inr(net)}

Authorised Signatory: HR Department, {employer}
"""

    gt = {
        "application_id": app_id,
        "doc_type": "salary_slip",
        "label": label,
        "extracted_fields": {
            "gross_salary": gross,
            "net_salary": net,
            "emi_deductions": emi if emi > 0 else None,
            "employment_type": emp_type,
            "designation": designation,
        },
        "expected_routing": routing,
        "primary_reason": "Income verified" if routing == "APPROVE" else "Manual verification needed",
        "critical_policy_rules": ["INCOME_VERIFIABLE"],
    }

    (APPS_DIR / f"{app_id}.txt").write_text(doc)
    (GT_DIR / f"{app_id}_gt.json").write_text(json.dumps(gt, indent=2))
    print(f"✓ {app_id} — {label} — {routing}")


# ══════════════════════════════════════════════════════════════════════════════
# KYC DOCUMENTS (5)
# ══════════════════════════════════════════════════════════════════════════════

kyc_cases = [
    ("KYC_COMPLETE",      True,  True,  True,  "VERIFIED",  "APPROVE"),
    ("KYC_NO_PAN",        True,  False, True,  "PENDING",   "ESCALATE"),
    ("KYC_MISMATCH",      True,  True,  True,  "MISMATCH",  "ESCALATE"),
    ("KYC_NO_AADHAAR",    False, True,  True,  "PENDING",   "ESCALATE"),
    ("KYC_MINIMAL",       True,  True,  False, "PENDING",   "ESCALATE"),
]

for i, (label, has_aadhaar, has_pan, has_address, v_status, routing) in enumerate(kyc_cases, 1):
    app_id = f"APP_{i+10:02d}"
    name = random.choice(["Suresh Kumar", "Nandini Rao", "Ajay Verma", "Pooja Shah", "Rakesh Tiwari"])
    dob  = f"{random.randint(1975,2000)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    aadhaar = fake_aadhaar() if has_aadhaar else None
    pan     = fake_pan()     if has_pan     else None

    doc = f"""KYC DOCUMENT PACKAGE

APPLICANT DETAILS
Full Name:     {name}
Date of Birth: {dob}
Father's Name: {random.choice(['Ramesh Kumar', 'Sunil Sharma', 'Mohan Das', 'Vijay Patel'])}

IDENTITY PROOF
{'Aadhaar Number: ' + aadhaar if aadhaar else 'Aadhaar: NOT PROVIDED'}
{'PAN Card: ' + pan           if pan     else 'PAN Card: NOT PROVIDED'}

{'ADDRESS PROOF' if has_address else ''}
{'Flat 4B, Green Valley Apartments, Bengaluru - 560001' if has_address else 'Address proof: NOT PROVIDED'}

KYC VERIFICATION STATUS: {v_status}
{"NOTE: Name mismatch between Aadhaar and PAN records." if v_status == "MISMATCH" else ""}
{"Documents submitted for manual review." if v_status == "PENDING" else ""}

Submission Date: {random.choice(['Jan','Feb','Mar'])} 2026
"""

    kyc_complete = has_aadhaar and has_pan and v_status == "VERIFIED"
    gt = {
        "application_id": app_id,
        "doc_type": "kyc",
        "label": label,
        "extracted_fields": {
            "aadhaar_placeholder": "[AADHAAR_1]" if has_aadhaar else None,
            "pan_placeholder":     "[IN_PAN_1]"  if has_pan     else None,
            "address_present":     has_address,
            "kyc_complete":        kyc_complete,
            "verification_status": v_status,
        },
        "expected_routing": routing,
        "primary_reason": "KYC verified" if routing == "APPROVE" else "KYC incomplete or mismatched",
        "critical_policy_rules": ["KYC_COMPLETE"],
    }

    (APPS_DIR / f"{app_id}.txt").write_text(doc)
    (GT_DIR / f"{app_id}_gt.json").write_text(json.dumps(gt, indent=2))
    print(f"✓ {app_id} — {label} — {routing}")


# ══════════════════════════════════════════════════════════════════════════════
# VEHICLE INSPECTION REPORTS (5)
# ══════════════════════════════════════════════════════════════════════════════

vehicle_cases = [
    ("VEH_CLEAR_A",       True,  "A", False, 850000, 35000, "APPROVE"),
    ("VEH_ENCUMBERED",    True,  "B", True,  620000, 48000, "REJECT"),
    ("VEH_POOR_GRADE",    False, "D", False, 280000, 60000, "REJECT"),
    ("VEH_BORDERLINE_B",  True,  "B", False, 480000, 42000, "ESCALATE"),
    ("VEH_CLEAR_HIGH",    True,  "A", False,1200000, 80000, "APPROVE"),
]

makes = [("Maruti", "Swift Dzire"), ("Honda", "City"), ("Hyundai", "Creta"),
         ("Toyota", "Innova"), ("Tata", "Nexon")]

for i, (label, passed, grade, encumbered, value, km, routing) in enumerate(vehicle_cases, 1):
    app_id = f"APP_{i+15:02d}"
    make, model = makes[i-1]
    year = random.randint(2018, 2022)

    doc = f"""VEHICLE INSPECTION REPORT
Inspection Agency: Profecto Vehicle Intelligence
Report ID: PROF-2026-{random.randint(100000,999999)}

VEHICLE DETAILS
Make & Model:   {make} {model}
Manufacture Year: {year}
Odometer Reading: {km:,} km
Chassis Number:   MHRS{random.randint(10000000,99999999)}
Engine Number:    K12N{random.randint(10000000,99999999)}
RC Number:        KA{random.randint(10,99)} AB {random.randint(1000,9999)}

TITLE STATUS
Registration Certificate: {'CLEAR — No existing charge/hypothecation' if not encumbered else 'ENCUMBERED — Existing hypothecation found (Lender: XYZ Finance)'}
Hypothecation:  {'NONE' if not encumbered else 'ACTIVE — Must be cleared before new loan'}

INSPECTION RESULTS
Overall Grade:         {grade}
Inspection Passed:     {'YES' if passed else 'NO — Vehicle does not meet minimum standards'}
Engine Condition:      {'Good' if grade in ('A','B') else 'Poor — Significant wear detected'}
Body Condition:        {'Good' if grade == 'A' else ('Minor dents' if grade == 'B' else 'Major damage')}
Tyre Condition:        {'Good' if grade in ('A','B') else 'Needs replacement'}
Interior Condition:    {'Good' if grade in ('A','B') else 'Poor'}

VALUATION
Assessed Market Value: {inr(value)}
Recommended LTV (80%): {inr(value * 0.80)}
Maximum Loan Amount:   {inr(value * 0.80)}

Inspector: {random.choice(['Rajan Mehta', 'Sonia Krishnan', 'Amit Jain'])}
Date: {random.choice(['Jan','Feb','Mar'])} 2026
"""

    gt = {
        "application_id": app_id,
        "doc_type": "vehicle_report",
        "label": label,
        "extracted_fields": {
            "vehicle_make":     make,
            "vehicle_model":    model,
            "manufacture_year": year,
            "assessed_value":   value,
            "condition_grade":  grade,
            "rc_encumbrance":   encumbered,
            "odometer_km":      km,
            "inspection_passed": passed,
        },
        "expected_routing": routing,
        "primary_reason": (
            "RC encumbrance must be cleared" if encumbered
            else "Inspection failed" if not passed
            else "Vehicle value borderline" if routing == "ESCALATE"
            else "Vehicle clear and valued"
        ),
        "critical_policy_rules": ["RC_ENCUMBRANCE"],
    }

    (APPS_DIR / f"{app_id}.txt").write_text(doc)
    (GT_DIR / f"{app_id}_gt.json").write_text(json.dumps(gt, indent=2))
    print(f"✓ {app_id} — {label} — {routing}")


print(f"\n✅ Generated 20 synthetic applications in {APPS_DIR}")
print(f"✅ Generated 20 ground truth files in {GT_DIR}")
print("\nRouting distribution:")
import glob
approve = escalate = reject = 0
for gt_file in sorted(glob.glob(str(GT_DIR / "*.json"))):
    gt = json.loads(Path(gt_file).read_text())
    r = gt["expected_routing"]
    if r == "APPROVE":   approve  += 1
    elif r == "ESCALATE": escalate += 1
    else:                reject   += 1
print(f"  APPROVE:  {approve}/20")
print(f"  ESCALATE: {escalate}/20")
print(f"  REJECT:   {reject}/20")
