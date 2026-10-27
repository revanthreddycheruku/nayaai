"""Live API verification script testing exact section pinpointing for Aviation, E-Commerce, and Insurance."""
import json
import httpx

BASE_URL = "http://127.0.0.1:8000"

QUERIES = [
    {
        "name": "1. Aviation: Lost Baggage",
        "message": "My baggage was lost on an airline flight from Delhi to Mumbai and the airline refuses to compensate me.",
        "expected_domain": "aviation",
        "expected_issue": "aviation_baggage",
        "exact_primary_section": "Part VI & Carriage by Air Act, 1972",
        "forbidden_in_primary": ["Flight Cancellations", "Denied Boarding"],
        "required_statutes": ["DGCA CAR", "Carriage by Air Act"],
        "required_criteria": ["20,000", "PIR"],
        "escalation_keywords": ["Nodal Officer", "AirSewa", "e-Jagriti"],
        "forbidden_domain": "insurance"
    },
    {
        "name": "2. E-Commerce: Damaged Laptop Refund Refusal",
        "message": "I purchased a laptop from an online seller on a marketplace platform, it arrived damaged, and the seller refused to take it back or provide a refund.",
        "expected_domain": "ecommerce",
        "expected_issue": "ecommerce_return_refusal",
        "exact_primary_section": "Rule 6(2)",
        "forbidden_in_primary": ["Free-Look", "Flight Cancellations"],
        "required_statutes": ["Consumer Protection (E-Commerce) Rules, 2020", "Consumer Protection Act, 2019"],
        "required_criteria": ["Section 2(47)(viii)", "30 days"],
        "escalation_keywords": ["Grievance Officer", "National Consumer Helpline", "e-Jagriti"],
        "forbidden_domain": "insurance"
    },
    {
        "name": "3. Insurance: Claim Delay",
        "message": "My health insurance claim has been pending without payment for over two months after submitting all necessary hospital bills and discharge summary.",
        "expected_domain": "insurance",
        "expected_issue": "insurance_delay",
        "exact_primary_section": "Regulation 11",
        "forbidden_in_primary": ["Free-Look", "Flight Cancellations"],
        "required_statutes": ["IRDAI Regulations, 2024"],
        "required_criteria": ["Regulation 14", "30", "Bank Rate"],
        "escalation_keywords": ["Grievance Redressal Officer", "Ombudsman", "e-Jagriti"],
        "forbidden_domain": "aviation"
    },
    {
        "name": "4. E-Commerce: 'Company Policy' Return Refusal",
        "message": "Online seller delivered a defective laptop and refused my return request citing company policy.",
        "expected_domain": "ecommerce",
        "expected_issue": "ecommerce_return_refusal",
        "exact_primary_section": "Rule 6(2)",
        "forbidden_in_primary": ["Free-Look", "Flight Cancellations", "Regulation 11"],
        "required_statutes": ["Consumer Protection (E-Commerce) Rules, 2020"],
        "required_criteria": ["Section 2(47)(viii)", "Rule 6(2)"],
        "escalation_keywords": ["Grievance Officer", "National Consumer Helpline", "e-Jagriti"],
        "forbidden_domain": "insurance"
    },
    {
        "name": "5. Aviation: Overnight Flight Delay",
        "message": "Flight delayed by 7 hours overnight at Delhi airport, airline refused hotel accommodation.",
        "expected_domain": "aviation",
        "expected_issue": "aviation_delay",
        "exact_primary_section": "Part IV",
        "forbidden_in_primary": ["Part VI", "Rule 6(2)", "Regulation 11"],
        "required_statutes": ["DGCA CAR Section 3, Series M, Part IV"],
        "required_criteria": ["Clause 3.4", "Hotel Accommodation"],
        "escalation_keywords": ["AirSewa", "DGCA", "e-Jagriti"],
        "forbidden_domain": "insurance"
    },
    {
        "name": "6. Aviation: Denied Boarding (Overbooking)",
        "message": "Airline overbooked flight and denied me boarding despite having a valid confirmed ticket.",
        "expected_domain": "aviation",
        "expected_issue": "aviation_denied_boarding",
        "exact_primary_section": "Clause 3.2",
        "forbidden_in_primary": ["Part VI", "Rule 6(2)"],
        "required_statutes": ["DGCA CAR Section 3, Series M, Part IV"],
        "required_criteria": ["Denied Boarding", "20,000"],
        "escalation_keywords": ["Duty Manager", "AirSewa", "e-Jagriti"],
        "forbidden_domain": "insurance"
    },
    {
        "name": "7. Insurance: Claim Repudiation",
        "message": "Health insurer rejected my surgery reimbursement claim alleging pre-existing condition.",
        "expected_domain": "insurance",
        "expected_issue": "insurance_repudiation",
        "exact_primary_section": "Regulation 11",
        "forbidden_in_primary": ["Part VI", "Rule 6(2)"],
        "required_statutes": ["IRDAI Regulations, 2024", "Insurance Ombudsman Rules, 2017"],
        "required_criteria": ["Ombudsman", "Burden of Proof"],
        "escalation_keywords": ["Grievance Redressal Officer", "Ombudsman", "e-Jagriti"],
        "forbidden_domain": "aviation"
    }
]

def run_verification():
    print(f"Connecting to {BASE_URL}...")
    with httpx.Client(base_url=BASE_URL, timeout=60) as client:
        health = client.get("/health")
        print("Health check status:", health.status_code, health.json())
        assert health.status_code == 200

        all_passed = True
        for q in QUERIES:
            print(f"\n=======================================================")
            print(f"TESTING: {q['name']}")
            print(f"User Query: \"{q['message']}\"")
            print(f"-------------------------------------------------------")

            # Test Search Endpoint
            search_res = client.post("/api/retrieval/search", json={"message": q["message"]})
            assert search_res.status_code == 200
            search_hits = search_res.json()["results"]
            print(f"Retrieved {len(search_hits)} chunks from /api/retrieval/search")
            for h in search_hits[:3]:
                print(f"  - [{h.get('domain')}] {h.get('statute')} | {h.get('section')}")
                if q["forbidden_domain"] in str(h.get("domain")):
                    print(f"  FAILED: Forbidden domain {q['forbidden_domain']} chunk present!")
                    all_passed = False

            # Test Chat Endpoint
            chat_res = client.post("/api/chat", json={"message": q["message"]})
            assert chat_res.status_code == 200
            data = chat_res.json()
            domain = data.get("domain")
            issue = data.get("issue")
            answer = data.get("answer", "")
            citations = data.get("citations", [])

            print(f"\nAPI Response:")
            print(f"  Domain: {domain} (Expected: {q['expected_domain']})")
            print(f"  Specific Issue: {issue} (Expected: {q['expected_issue']})")
            if domain != q["expected_domain"]:
                print(f"  FAILED: Domain mismatch! Expected {q['expected_domain']}, got {domain}")
                all_passed = False
            if issue != q["expected_issue"]:
                print(f"  FAILED: Issue mismatch! Expected {q['expected_issue']}, got {issue}")
                all_passed = False

            # Check Citations for cross-contamination
            for c in citations:
                if c.get("domain") == q["forbidden_domain"]:
                    print(f"  FAILED: Citation cross-contamination! Found chunk with domain {c.get('domain')}")
                    all_passed = False

            # Check exact primary governing section in answer
            if q["exact_primary_section"].lower() not in answer.lower():
                print(f"  FAILED: Exact primary section '{q['exact_primary_section']}' missing from answer!")
                all_passed = False
            else:
                print(f"  PASSED: Exact primary section '{q['exact_primary_section']}' found in answer.")

            # Check forbidden terms in primary section
            primary_part = answer.split("**2.")[0] if "**2." in answer else answer[:400]
            for forb in q["forbidden_in_primary"]:
                if forb.lower() in primary_part.lower():
                    print(f"  FAILED: Unrelated section '{forb}' incorrectly cited in primary section!")
                    all_passed = False

            # Check required criteria
            for crit in q["required_criteria"]:
                if crit.lower() not in answer.lower():
                    print(f"  FAILED: Required criterion '{crit}' not found in answer!")
                    all_passed = False
                else:
                    print(f"  PASSED: Criterion '{crit}' present.")

            # Check 3-Tier Escalation
            found_escalation = [esc for esc in q["escalation_keywords"] if esc.lower() in answer.lower()]
            print(f"  Escalation Nodes Cited: {found_escalation} (Required: {q['escalation_keywords']})")
            if len(found_escalation) < 2:
                print(f"  FAILED: Incomplete 3-tier escalation path!")
                all_passed = False

            print(f"\nAnswer Excerpt:\n{answer[:650]}...\n")

        print("=======================================================")
        if all_passed:
            print("ALL LIVE EXACT SECTION VERIFICATION TESTS PASSED SUCCESSFULLY!")
        else:
            print("SOME VERIFICATION CHECKS FAILED.")
            raise SystemExit(1)

if __name__ == "__main__":
    run_verification()
