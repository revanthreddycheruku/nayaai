import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app, classify, classify_domain, classify_issue, retrieve

client = TestClient(app)


def test_topics_and_health():
    assert client.get('/health').status_code == 200
    health_data = client.get('/health').json()
    assert health_data['status'] == 'ok'
    assert health_data['vector_store_exists'] is True
    topics = client.get('/api/topics').json()['topics']
    assert 'insurance claim disputes' in topics and len(topics) >= 9


def test_classification():
    assert classify('Airline cancelled my flight and did not refund me') == 'flight cancellation and delay disputes'
    assert classify('Insurer denied my health insurance claim') == 'insurance claim disputes'
    assert classify('Online seller delivered wrong product') == 'e-commerce disputes'
    assert classify('The laptop screen is defective and shattered') == 'defective products'


def test_domain_and_issue_routing():
    # Aviation
    assert classify_domain('Airline lost my baggage on flight from Delhi') == 'aviation'
    assert classify_issue('Airline lost my baggage on flight from Delhi', 'aviation') == 'aviation_baggage'
    assert classify_issue('Flight was cancelled without 24 hours notice', 'aviation') == 'aviation_cancellation'
    assert classify_issue('Stuck at airport due to 6 hour flight delay', 'aviation') == 'aviation_delay'
    assert classify_issue('Airline denied boarding due to overbooking', 'aviation') == 'aviation_denied_boarding'

    # E-Commerce
    assert classify_domain('Online marketplace seller refused to refund damaged laptop') == 'ecommerce'
    assert classify_issue('Online marketplace seller refused to refund damaged laptop', 'ecommerce') == 'ecommerce_return_refusal'
    assert classify_issue('Defective charger caught fire and caused injury', 'ecommerce') == 'ecommerce_product_liability'

    # Insurance
    assert classify_domain('Insurer delayed paying health insurance claim for 60 days') == 'insurance'
    assert classify_issue('Insurer delayed paying health insurance claim for 60 days', 'insurance') == 'insurance_delay'
    assert classify_issue('I want to cancel the insurance policy during free look period', 'insurance') == 'insurance_free_look'


def test_analyze_does_not_require_every_field():
    response = client.post('/api/complaint/analyze', json={'message': 'A seller delivered a defective phone for ₹12,000.'})
    assert response.status_code == 200
    data = response.json()
    assert data['domain'] == 'ecommerce'
    assert data['complaint']['amount_paid'] == '₹12,000'


def test_draft_uses_only_provided_facts():
    response = client.post('/api/complaint/draft', json={'details': {'company': 'Example Shop', 'chronology': 'Phone arrived cracked'}})
    assert response.status_code == 200
    draft = response.json()['draft']
    assert 'Example Shop' in draft and 'Phone arrived cracked' in draft
    assert '[Please add]' in draft


def test_empty_verified_kb_does_not_invent_answer(monkeypatch):
    from app import main
    monkeypatch.setattr(main, 'retrieve', lambda query, k=6, domain_override=None: [])
    response = client.post('/api/chat', json={'message': 'I was denied a refund for a defective appliance.'})
    assert response.status_code == 200
    assert 'no eligible independently verified sources' in response.json()['answer']
    assert response.json()['citations'] == []


def test_aviation_lost_baggage_exact_retrieval():
    query = 'My baggage was lost on an Indigo flight from Delhi to Bangalore and airline refuses to compensate'
    hits = retrieve(query)
    assert len(hits) > 0, 'Should retrieve aviation chunks'
    for h in hits:
        assert h.get('domain') == 'aviation', f"Expected aviation domain, got {h.get('domain')}"
        assert 'insurance' not in h.get('title', '').lower()

    # The top chunk must be the exact Baggage Rules section
    top_hit = hits[0]
    assert 'Part VI' in top_hit.get('section', '') or 'Baggage' in top_hit.get('section', '')
    assert 'DGCA CAR' in top_hit.get('statute', '') or 'Carriage by Air' in top_hit.get('statute', '')


def test_ecommerce_damaged_laptop_exact_retrieval():
    query = 'I purchased a laptop from an online seller, it arrived damaged, and the seller refused to take it back or refund'
    hits = retrieve(query)
    assert len(hits) > 0, 'Should retrieve e-commerce chunks'
    for h in hits:
        assert h.get('domain') in {'ecommerce', 'general'}
        assert 'aviation' != h.get('domain')

    # Top hit must be Rule 6 return refusal prohibition or Section 2(47)
    top_hit = hits[0]
    assert ('Rule 6' in top_hit.get('section', '') or '2(47)' in top_hit.get('section', '') or '2(11)' in top_hit.get('section', ''))


def test_insurance_delay_exact_retrieval():
    query = 'My health insurance claim has been pending without payment for over two months after submitting all documents'
    hits = retrieve(query)
    assert len(hits) > 0, 'Should retrieve insurance chunks'
    for h in hits:
        assert h.get('domain') == 'insurance'

    top_sections = [h.get('section', '') for h in hits[:2]]
    joined = " ".join(top_sections)
    assert '11' in joined or '14' in joined or 'Settlement' in joined or 'Interest' in joined


def test_chat_api_exact_sections_for_each_issue():
    # 1. Aviation Lost Baggage -> Must give Part VI & Carriage by Air Act, ₹20,000 / SDRs, PIR
    resp_av = client.post('/api/chat', json={'message': 'My baggage was lost on an airline flight and they refuse to compensate me'})
    assert resp_av.status_code == 200
    data_av = resp_av.json()
    assert data_av['domain'] == 'aviation'
    assert data_av['issue'] == 'aviation_baggage'
    ans_av = data_av['answer']
    assert 'Part VI' in ans_av
    assert 'Carriage by Air Act, 1972' in ans_av
    assert '20,000' in ans_av
    assert 'Property Irregularity Report' in ans_av or 'PIR' in ans_av
    assert 'Tier 1' in ans_av and 'Nodal Officer' in ans_av
    assert 'Tier 2' in ans_av and 'AirSewa' in ans_av
    assert 'Tier 3' in ans_av and 'e-Jagriti' in ans_av
    assert 'Executive Legal Summary' in ans_av
    assert 'Strategic Suggestions' in ans_av

    # 2. E-Commerce Damaged Laptop Return Refusal -> Must give Rule 6(2), Section 2(47)(viii), Section 2(11)
    resp_ec = client.post('/api/chat', json={'message': 'I purchased a laptop from an online seller, it arrived damaged, and the seller refused to take it back or refund'})
    assert resp_ec.status_code == 200
    data_ec = resp_ec.json()
    assert data_ec['domain'] == 'ecommerce'
    assert data_ec['issue'] == 'ecommerce_return_refusal'
    ans_ec = data_ec['answer']
    assert 'Rule 6(2)' in ans_ec or 'Rule 6' in ans_ec
    assert 'Section 2(47)(viii)' in ans_ec or 'Section 2(47)' in ans_ec
    assert 'Tier 1' in ans_ec and 'Grievance Officer' in ans_ec
    assert 'Tier 2' in ans_ec and 'National Consumer Helpline' in ans_ec
    assert 'Tier 3' in ans_ec and 'e-Jagriti' in ans_ec
    assert 'Executive Legal Summary' in ans_ec
    assert 'Strategic Suggestions' in ans_ec

    # 3. Insurance Claim Delay -> Must give Regulation 11 (30-day TAT) and Regulation 14 (Bank Rate + 2% penal interest)
    resp_ins = client.post('/api/chat', json={'message': 'My health insurance claim has been pending without payment for over two months after submitting all documents'})
    assert resp_ins.status_code == 200
    data_ins = resp_ins.json()
    assert data_ins['domain'] == 'insurance'
    assert data_ins['issue'] == 'insurance_delay'
    ans_ins = data_ins['answer']
    assert 'Regulation 11' in ans_ins
    assert 'Regulation 14' in ans_ins
    assert '30' in ans_ins
    assert 'Bank Rate' in ans_ins
    assert 'Tier 1' in ans_ins and 'Grievance Redressal Officer' in ans_ins
    assert 'Tier 2' in ans_ins and ('Ombudsman' in ans_ins or 'Bima Bharosa' in ans_ins)
    assert 'Tier 3' in ans_ins and 'e-Jagriti' in ans_ins
    assert 'Executive Legal Summary' in ans_ins
    assert 'Strategic Suggestions' in ans_ins

    # 4. E-Commerce "citing company policy" return refusal -> Must NOT misclassify as insurance
    resp_pol = client.post('/api/chat', json={'message': 'Online seller delivered a defective laptop and refused my return request citing company policy'})
    assert resp_pol.status_code == 200
    data_pol = resp_pol.json()
    assert data_pol['domain'] == 'ecommerce'
    assert data_pol['issue'] == 'ecommerce_return_refusal'
    assert 'Rule 6(2)' in data_pol['answer']

    # 5. Aviation Flight Delay -> Must give Part IV Clause 3.4
    resp_delay = client.post('/api/chat', json={'message': 'Flight delayed by 7 hours overnight at Delhi airport, airline refused hotel accommodation'})
    assert resp_delay.status_code == 200
    data_delay = resp_delay.json()
    assert data_delay['domain'] == 'aviation'
    assert data_delay['issue'] == 'aviation_delay'
    assert 'Part IV' in data_delay['answer']
    assert 'Clause 3.4' in data_delay['answer'] or 'Flight Delays' in data_delay['answer']

    # 6. Aviation Denied Boarding -> Must give Part IV Clause 3.2
    resp_db = client.post('/api/chat', json={'message': 'Airline overbooked flight and denied me boarding despite having a valid ticket'})
    assert resp_db.status_code == 200
    data_db = resp_db.json()
    assert data_db['domain'] == 'aviation'
    assert data_db['issue'] == 'aviation_denied_boarding'
    assert 'Clause 3.2' in data_db['answer'] or 'Denied Boarding' in data_db['answer']

    # 7. Insurance Repudiation -> Must give Regulation 11 & Ombudsman Rule 13
    resp_rep = client.post('/api/chat', json={'message': 'Health insurer rejected my surgery reimbursement claim alleging pre-existing condition'})
    assert resp_rep.status_code == 200
    data_rep = resp_rep.json()
    assert data_rep['domain'] == 'insurance'
    assert data_rep['issue'] == 'insurance_repudiation'
    assert 'Ombudsman' in data_rep['answer']
