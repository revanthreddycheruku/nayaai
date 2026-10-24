import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app, classify

client=TestClient(app)

def test_topics_and_health():
    assert client.get('/health').status_code==200
    topics=client.get('/api/topics').json()['topics']
    assert 'insurance claim disputes' in topics and len(topics)>=9

def test_classification():
    assert classify('Airline cancelled my flight and did not refund me')=='flight cancellation and delay disputes'
    assert classify('Insurer denied my health insurance claim')=='insurance claim disputes'

def test_analyze_does_not_require_every_field():
    response=client.post('/api/complaint/analyze',json={'message':'A seller delivered a defective phone for ₹12,000.'})
    assert response.status_code==200
    assert response.json()['topic']=='e-commerce disputes'
    assert response.json()['complaint']['amount_paid']=='₹12,000'

def test_draft_uses_only_provided_facts():
    response=client.post('/api/complaint/draft',json={'details':{'company':'Example Shop','chronology':'Phone arrived cracked'}})
    assert response.status_code==200
    draft=response.json()['draft']
    assert 'Example Shop' in draft and 'Phone arrived cracked' in draft
    assert '[Please add]' in draft

def test_empty_verified_kb_does_not_invent_answer(monkeypatch):
    from app import main
    monkeypatch.setattr(main,'retrieve',lambda query:[])
    response=client.post('/api/chat',json={'message':'I was denied a refund for a defective appliance.'})
    assert response.status_code==200
    assert 'no eligible independently verified sources' in response.json()['answer']
    assert response.json()['citations']==[]
