"""
Quick diagnostic: test LM Studio connectivity and raw JSON extraction.
Run: python scripts/test_llm.py
"""
import os, sys, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("LLM_BASE_URL", "http://localhost:1234/v1")
os.environ.setdefault("LLM_API_KEY",  "lm-studio")
os.environ.setdefault("LLM_MODEL",    "google/gemma-3-4b")

from openai import OpenAI
import config

client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

print(f"Testing LM Studio at {config.LLM_BASE_URL} with model {config.LLM_MODEL}\n")

# Test 1: basic JSON output
print("=== Test 1: Basic JSON ===")
try:
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "Respond with ONLY a JSON object. No markdown, no extra text. Start with { and end with }."},
            {"role": "user",   "content": 'Return {"status": "ok", "value": 42}'},
        ],
        temperature=0,
        max_tokens=100,
    )
    raw = resp.choices[0].message.content or ""
    print(f"Raw output: {repr(raw)}")
    try:
        parsed = json.loads(raw.strip())
        print(f"Parsed OK: {parsed}")
    except Exception:
        # Try regex
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            parsed = json.loads(m.group())
            print(f"Parsed via regex: {parsed}")
        else:
            print("FAILED to parse JSON from output")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: extraction on a tiny bank statement snippet
print("\n=== Test 2: Mini extraction ===")
snippet = """
BANK STATEMENT — [BANK_1]
Account Holder: [PERSON_1]
Account Number: [ACCOUNT_1]

Date       Description          Debit      Credit     Balance
01-Jan-24  Opening Balance                            50,000
05-Jan-24  EMI Payment          15,000                35,000
15-Jan-24  Salary Credit                   80,000    115,000
25-Jan-24  EMI Payment          15,000                100,000
Closing Balance: 100,000
"""

schema_hint = '{"account_type": "string", "estimated_monthly_income": "number or null", "emi_obligations": "number or null", "foir": "number or null", "suspicious_transactions": "boolean", "employment_type": "SALARIED or SELF_EMPLOYED or null"}'

try:
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a financial extraction assistant. Respond with ONLY a JSON object. No markdown fences. Start with { end with }."},
            {"role": "user", "content": f"Extract from this bank statement. Schema: {schema_hint}\n\nDocument:\n{snippet}\n\nRespond with ONLY the JSON object."},
        ],
        temperature=0,
        max_tokens=300,
    )
    raw = resp.choices[0].message.content or ""
    print(f"Raw output:\n{raw}")
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        print(f"\nParsed: {json.loads(m.group())}")
    else:
        print("No JSON found in output")
except Exception as e:
    print(f"ERROR: {e}")
