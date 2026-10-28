from __future__ import annotations
import os, re, logging
from pathlib import Path
from typing import Any, Literal
from functools import lru_cache
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
log = logging.getLogger("consumer-law-rag")

app = FastAPI(
    title="Consumer Rights AI",
    version="0.3.0",
    description="Evidence-grounded consumer law information with domain routing and exact statutory section pinpointing"
)

cors_env = os.getenv("CORS_ORIGINS", "*")
if "*" in cors_env:
    origins = ["*"]
else:
    origins = [x.strip() for x in cors_env.split(",") if x.strip()]
    for extra in ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"]:
        if extra not in origins:
            origins.append(extra)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

MAX_CHARS = 12000
TOPICS = [
    "defective products",
    "refund and cancellation disputes",
    "misleading advertisements",
    "unfair trade practices",
    "e-commerce disputes",
    "deficiency in service",
    "insurance claim disputes",
    "flight cancellation and delay disputes",
    "product liability",
    "other or uncertain consumer issues"
]

SYSTEM_PROMPT = """You are an evidence-grounded Indian Consumer Law AI specialist. You provide structured, actionable consumer legal information based strictly on official statutes, regulations, and verified legal frameworks.

CRITICAL INSTRUCTION - EXACT SECTION PINPOINTING:
Do NOT list all sections across a sector. You MUST pinpoint and cite the EXACT section that directly governs the user's specific grievance:
- Lost / Delayed / Damaged Baggage: Cite DGCA CAR Section 3, Series M, Part VI & Carriage by Air Act, 1972 (Third Schedule / Montreal Convention - Baggage Rules). Do NOT cite Flight Cancellations or Denied Boarding.
- Flight Cancellation: Cite DGCA CAR Section 3, Series M, Part IV (Flight Cancellations & Mandatory Compensation).
- Flight Delay: Cite DGCA CAR Section 3, Series M, Part IV (Flight Delays, Meals, Refreshments, and Hotel Accommodation).
- Ticket Refund / Non-refundable taxes: Cite DGCA CAR Section 3, Series M, Part II (Refund of Airline Tickets).
- Denied Boarding: Cite DGCA CAR Section 3, Series M, Part IV (Denied Boarding Compensation).
- E-Commerce Damaged / Defective Product Return Refusal: Cite Consumer Protection (E-Commerce) Rules, 2020, Rule 6(2) (Prohibition of Return Refusal) and Consumer Protection Act, 2019, Section 2(47)(viii) (Unfair Trade Practice: Refusal to refund within 30 days) and Section 2(11) (Deficiency in Service).
- Defective Product Caused Injury / Property Damage: Cite Consumer Protection Act, 2019, Chapter VI (Sections 82-87 Product Liability).
- E-Commerce Dark Patterns / Unfair Charges: Cite Consumer Protection (E-Commerce) Rules, 2020, Rule 4(2) & 4(11) and Consumer Protection Act, 2019, Section 2(47).
- Insurance Claim Processing Delay: Cite IRDAI Regulations, 2024, Regulation 11 & Schedule (Mandatory 30-day Settlement TAT) and Regulation 14 & Schedule (Penal Interest at Bank Rate + 2%). Do NOT cite Free-Look Period for a claim processing delay.
- Insurance Free-Look / Cancellation: Cite IRDAI Regulations, 2024, Regulation 8 & 9 (30-Day Mandatory Free-Look Period).
- Insurance Claim Repudiation / Rejection: Cite IRDAI Regulations, 2024, Regulation 11 & Schedule (Reasoned Written Order within 30 days) and Insurance Ombudsman Rules, 2017, Rule 13(1)(a).

Structure your response with:
1. Exact Governing Statutory Section for this Issue
2. Precise Legal Test & Statutory Criteria
3. Secondary / Supporting Statutory Provisions
4. 3-Tier Escalation Path (Tier 1: Nodal Officer -> Tier 2: Sector Regulator -> Tier 3: e-Jagriti)
5. Executive Legal Summary (concise summary of the legal standing and core remedies)
6. Strategic Suggestions & Proactive Steps Beyond Immediate Escalation (evidence preservation, Section 69 limitation period, formal pre-litigation notice, e-Jagriti self-representation)
Cite source labels [S#]."""


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=3000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=3, max_length=MAX_CHARS)
    conversation_id: str | None = Field(default=None, max_length=128)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    document_id: str
    title: str
    source_url: str | None = None
    document_type: str = "unknown"
    verification_status: str = "unknown"
    legal_provisions: list[str] = Field(default_factory=list)
    paragraph_reference: str | None = None
    excerpt: str = ""
    score: float | None = None
    domain: str | None = None
    statute: str | None = None
    section: str | None = None


class ChatResponse(BaseModel):
    answer: str
    topic: str
    domain: str | None = None
    issue: str | None = None
    follow_up_questions: list[str] = Field(default_factory=list)
    citations: list[Source] = Field(default_factory=list)
    limitation: str | None = None


class AnalyzeRequest(BaseModel):
    message: str = Field(min_length=3, max_length=MAX_CHARS)


class SearchRequest(BaseModel):
    message: str = Field(min_length=3, max_length=MAX_CHARS)
    domain: str | None = None


class DraftRequest(BaseModel):
    details: dict[str, str] = Field(default_factory=dict)


def classify_domain(text: str) -> str:
    """Route queries into specific legal domains: aviation, insurance, ecommerce, or general.
    Uses negative context filtering and weighted scoring to eliminate false cross-domain collisions."""
    t = text.lower()

    # Negative filters for insurance:
    # Terms like "company policy", "store policy", "return policy", "baggage claim" are common false positives.
    has_insurance_negatives = bool(re.search(
        r"\b(company policy|store policy|return policy|refund policy|exchange policy|cancellation policy|baggage policy|seller policy|terms and policy|baggage claim|warranty claim|damage claim|claim a refund|claim for refund)\b",
        t
    ))

    # Aviation scoring
    aviation_strong = len(re.findall(
        r"\b(flight|flights|airline|airlines|boarding|baggage|luggage|dgca|airport|air ticket|aircraft|carrier by air|carriage by air|pir|airsewa|flyer|indigo|air india|spicejet|vistara|akasa)\b",
        t
    ))
    aviation_score = aviation_strong * 3

    # Insurance scoring
    insurance_strong = len(re.findall(
        r"\b(insurance|insurer|insurers|insured|policyholder|policyholders|premium|tpa|cashless|mediclaim|sum insured|bima bharosa|bima|ombudsman|repudiat\w*)\b",
        t
    ))
    insurance_context = len(re.findall(
        r"\b(health claim|life claim|motor claim|medical claim|claim settlement|settle the claim|hospitalization|surveyor report|free-look|free look)\b",
        t
    ))
    insurance_weak = len(re.findall(r"\b(policy|policies|claim|claims)\b", t))
    insurance_score = (insurance_strong * 3) + (insurance_context * 3)
    if not has_insurance_negatives:
        insurance_score += insurance_weak * 1

    # E-commerce scoring
    ecommerce_strong = len(re.findall(
        r"\b(e-commerce|ecommerce|online shopping|marketplace|seller|sellers|flipkart|amazon|myntra|meesho|order|ordered|delivery|delivered|return request|refund request|courier|package|parcel)\b",
        t
    ))
    ecommerce_medium = len(re.findall(
        r"\b(laptop|phone|refrigerator|appliance|product|goods|defective|damaged product|broken item|wrong item|warranty|return\b|refund)\b",
        t
    ))
    ecommerce_score = (ecommerce_strong * 3) + (ecommerce_medium * 2)

    scores = {"aviation": aviation_score, "insurance": insurance_score, "ecommerce": ecommerce_score}
    best_domain, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score >= 2:
        return best_domain
    return "general"


def classify_issue(text: str, domain: str) -> str:
    """Classify the exact specific issue within a domain to pinpoint governing statutes."""
    t = text.lower()
    if domain == "aviation":
        if re.search(r"\b(baggage|luggage|suitcase|suitcases|bag|bags|lost bag|damaged bag|pir|cargo)\b", t):
            return "aviation_baggage"
        if re.search(r"\b(deni\w*\s+.*?boarding|boarding\s+deni\w*|overbook\w*|bumped)\b", t):
            return "aviation_denied_boarding"
        if re.search(r"\b(cancel\w*)\b", t):
            return "aviation_cancellation"
        if re.search(r"\b(delay|delayed|delaying|wait|hours|stuck|tarmac|hotel|refreshment)\b", t):
            return "aviation_delay"
        if re.search(r"\b(refund|ticket|fare|tax|taxes|credit shell|voucher)\b", t):
            return "aviation_refund"
        return "aviation_general"

    if domain == "ecommerce":
        if re.search(r"\b(harm|injur\w*|fire|explod\w*|burn|shock|hazardous|unsafe|accident|hospital|damage to property)\b", t):
            return "ecommerce_product_liability"
        if re.search(r"\b(refus\w*.*(return|refund)|return.*refus\w*|refund.*refus\w*|no return|no refund|take back|policy.*return)\b", t):
            return "ecommerce_return_refusal"
        if re.search(r"\b(dark pattern|hidden charge|pre-ticked|unilateral|fake review|misleading|cancellation fee|opt-out)\b", t):
            return "ecommerce_unfair_practice"
        if re.search(r"\b(defective|damaged|faulty|broken|laptop|phone|wrong item|screen|battery|counterfeit|fake|spurious|not working)\b", t):
            return "ecommerce_return_refusal"
        return "ecommerce_general"

    if domain == "insurance":
        if re.search(r"\b(free look|free-look|cooling off|cancel policy|cancellation|mis-sold|mis sold|return policy)\b", t):
            return "insurance_free_look"
        if re.search(r"\b(delay|delayed|delaying|pending|overdue|unprocessed|tat|penal interest|interest on delay|30 days|45 days|months)\b", t):
            return "insurance_delay"
        if re.search(r"\b(reject|repudiat\w*|denied|denial|frivolous|pre-existing|clause)\b", t):
            return "insurance_repudiation"
        return "insurance_general"

    return "general"


ISSUE_PRIMARY_TARGETS = {
    "aviation_baggage": ["part vi", "baggage", "carriage by air"],
    "aviation_cancellation": ["part iv (flight cancellations", "cancellation"],
    "aviation_delay": ["part iv (flight delays", "delays"],
    "aviation_denied_boarding": ["denied boarding", "part iv"],
    "aviation_refund": ["part ii", "refund of airline tickets"],
    "ecommerce_return_refusal": ["rule 6", "return refusal", "section 2(47)", "section 2(11)"],
    "ecommerce_product_liability": ["chapter vi", "product liability", "section 84", "section 86"],
    "ecommerce_unfair_practice": ["rule 4", "dark pattern", "section 2(47)"],
    "insurance_delay": ["regulation 11", "regulation 14", "claim settlement", "penal interest", "tat"],
    "insurance_repudiation": ["regulation 11", "ombudsman", "repudiation"],
    "insurance_free_look": ["regulation 8", "free look", "free-look"],
}


def classify(text: str) -> str:
    """Classify consumer issues into topic taxonomy."""
    t = text.lower()
    domain = classify_domain(text)

    if domain == "aviation":
        return "flight cancellation and delay disputes"

    if domain == "insurance":
        return "insurance claim disputes"

    if domain == "ecommerce":
        if re.search(r"\b(harm|injur\w*|fire|explod\w*|burn|shock|hazardous|unsafe|accident)\b", t):
            return "product liability"
        if re.search(r"\b(fake review|misleading|false claim|advert|dark pattern|hidden charge|pre-ticked)\b", t):
            return "unfair trade practices"
        if re.search(r"\b(seller|marketplace|online|delivery|delivered)\b", t):
            return "e-commerce disputes"
        if re.search(r"\b(defective|faulty|damaged|broken|screen|battery|laptop|phone|appliance|refrigerator|hardware)\b", t):
            return "defective products"
        if re.search(r"\b(cancel|cancellation|refund|return)\b", t):
            return "refund and cancellation disputes"
        return "e-commerce disputes"

    if re.search(r"\b(advert|misleading|false claim|endorsement)\b", t):
        return "misleading advertisements"
    if re.search(r"\b(unfair trade|hidden charge|overcharg|deceptive)\b", t):
        return "unfair trade practices"
    if re.search(r"\b(defective|faulty|damaged|warranty)\b", t):
        return "defective products"
    if re.search(r"\b(service|repair|bank|hotel|telecom)\b", t):
        return "deficiency in service"
    return "other or uncertain consumer issues"


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"))


def _execute_hybrid_retrieval(
    collection: Any,
    query: str,
    allowed_domains: list[str],
    k: int = 6
) -> list[dict[str, Any]]:
    """Execute domain-filtered hybrid retrieval (Semantic vector search + BM25 keyword search + RRF).
    Optimized for low-memory cloud runtimes (e.g. Render 512MB free tier) with safe neural fallback."""
    if len(allowed_domains) == 1:
        where_filter = {"domain": allowed_domains[0]}
    else:
        where_filter = {"domain": {"$in": allowed_domains}}

    domain_corpus = collection.get(where=where_filter, include=["documents", "metadatas"])
    doc_ids = domain_corpus.get("ids", [])
    if not doc_ids:
        return []

    n_results = min(max(k * 4, 16), len(doc_ids))
    semantic_ids = []

    # Only run heavy PyTorch neural embeddings if explicitly enabled, preventing Render 512MB OOM crashes
    enable_neural = os.getenv("ENABLE_NEURAL_EMBEDDINGS", "false").lower() in ("true", "1")
    if enable_neural:
        try:
            q_emb = embedder().encode([query], normalize_embeddings=True).tolist()
            semantic = collection.query(
                query_embeddings=q_emb,
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            semantic_ids = semantic.get("ids", [[]])[0]
        except Exception as e:
            log.warning(f"Neural embeddings bypassed ({e}); using high-speed statutory BM25 ranking.")

    from rank_bm25 import BM25Okapi
    tokenized = [re.findall(r"[a-z0-9]+", text.lower()) for text in domain_corpus.get("documents", [])]
    bm25 = BM25Okapi(tokenized)
    query_tokens = re.findall(r"[a-z0-9]+", query.lower())
    scores = bm25.get_scores(query_tokens)
    kw = [(float(score), i) for i, score in enumerate(scores) if score > 0]
    kw.sort(reverse=True)
    kwrank = {i: r + 1 for r, (_, i) in enumerate(kw[:n_results])}

    byid = {}
    metas = domain_corpus.get("metadatas", [])
    docs = domain_corpus.get("documents", [])

    for rank, i in enumerate(semantic_ids, 1):
        j = doc_ids.index(i) if i in doc_ids else -1
        if j < 0:
            continue
        rrf_score = 1 / (60 + rank) + 1 / (60 + kwrank.get(j, 9999))
        byid[i] = (rrf_score, dict(metas[j] or {}), docs[j])

    for j, r in kwrank.items():
        ident = doc_ids[j]
        if ident not in byid:
            byid[ident] = (1 / (60 + r), dict(metas[j] or {}), docs[j])

    # If no direct keyword hits matched, fallback to top domain statute documents
    if not byid:
        for j, ident in enumerate(doc_ids[:k]):
            byid[ident] = (1 / (60 + j + 1), dict(metas[j] or {}), docs[j])

    found = []
    for score, meta, text in sorted(byid.values(), key=lambda x: x[0], reverse=True)[:k]:
        meta["excerpt"] = text[:1200]
        meta["score"] = score
        if isinstance(meta.get("legal_provisions"), str):
            meta["legal_provisions"] = [x.strip() for x in meta["legal_provisions"].split("|") if x.strip()]
        if meta.get("independently_verified") is True or os.getenv("ALLOW_UNVERIFIED_CONTEXT", "false").lower() == "true":
            found.append(meta)

    return found


def retrieve(query: str, k: int = 6, domain_override: str | None = None) -> list[dict[str, Any]]:
    """Retrieve verified legal chunks with domain routing, issue pinpointing, and CPA fallback."""
    db = Path(os.getenv("CHROMA_PATH", "./vector_store"))
    if not db.exists():
        return []
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(db))
        collection = client.get_collection("consumer_law")
        if collection.count() == 0:
            return []

        domain = domain_override or classify_domain(query)
        issue = classify_issue(query, domain)

        if domain == "aviation":
            allowed_domains = ["aviation"]
        elif domain == "ecommerce":
            allowed_domains = ["ecommerce", "general"]
        elif domain == "insurance":
            allowed_domains = ["insurance"]
        else:
            allowed_domains = ["general", "ecommerce"]

        # Step 1: Perform domain-filtered retrieval
        hits = _execute_hybrid_retrieval(collection, query, allowed_domains, k=12)

        # Step 2: Fall back to general Consumer Protection Act 2019 provisions if domain-specific rule is absent
        if not hits and domain != "general":
            log.info("Domain-specific rule absent for %s; falling back to general CPA 2019", domain)
            hits = _execute_hybrid_retrieval(collection, query, ["general"], k=12)

        # Step 3: Issue-specific pinpointing & re-ranking
        targets = ISSUE_PRIMARY_TARGETS.get(issue, [])
        if targets and hits:
            def matches_target(h):
                txt = (str(h.get("section", "")) + " " + str(h.get("title", "")) + " " + str(h.get("statute", ""))).lower()
                return any(t in txt for t in targets)

            matching = [h for h in hits if matches_target(h)]
            non_matching = [h for h in hits if not matches_target(h)]
            if matching:
                hits = matching + non_matching

        return hits[:k]
    except Exception as e:
        log.exception("Retrieval unavailable")
        return []


def extract_complaint_facts(text: str, domain: str) -> dict[str, Any]:
    """Extract named enterprises, disputed items, monetary amounts, and reference IDs."""
    t = text.lower()

    # 1. Opposite Party / Enterprise Identification
    company = None
    if re.search(r"\b(indigo|goindigo)\b", t):
        company = "IndiGo Airlines"
    elif re.search(r"\b(air india|airindia)\b", t):
        company = "Air India"
    elif re.search(r"\b(spicejet)\b", t):
        company = "SpiceJet"
    elif re.search(r"\b(vistara)\b", t):
        company = "Vistara (Air India Group)"
    elif re.search(r"\b(akasa|akasa air)\b", t):
        company = "Akasa Air"
    elif re.search(r"\b(amazon)\b", t):
        company = "Amazon India"
    elif re.search(r"\b(flipkart)\b", t):
        company = "Flipkart"
    elif re.search(r"\b(myntra)\b", t):
        company = "Myntra"
    elif re.search(r"\b(meesho)\b", t):
        company = "Meesho"
    elif re.search(r"\b(blinkit)\b", t):
        company = "Blinkit"
    elif re.search(r"\b(zepto)\b", t):
        company = "Zepto"
    elif re.search(r"\b(swiggy)\b", t):
        company = "Swiggy"
    elif re.search(r"\b(zomato)\b", t):
        company = "Zomato"
    elif re.search(r"\b(star health)\b", t):
        company = "Star Health and Allied Insurance"
    elif re.search(r"\b(hdfc ergo)\b", t):
        company = "HDFC ERGO General Insurance"
    elif re.search(r"\b(icici lombard)\b", t):
        company = "ICICI Lombard General Insurance"
    elif re.search(r"\b(care health)\b", t):
        company = "Care Health Insurance"
    elif re.search(r"\b(niva bupa|max bupa)\b", t):
        company = "Niva Bupa Health Insurance"
    elif re.search(r"\b(tata aig)\b", t):
        company = "Tata AIG General Insurance"
    elif re.search(r"\b(lic|life insurance corporation)\b", t):
        company = "Life Insurance Corporation of India (LIC)"
    elif re.search(r"\b(new india assurance)\b", t):
        company = "The New India Assurance Co."
    elif re.search(r"\b(united india)\b", t):
        company = "United India Insurance Co."
    elif re.search(r"\b(national insurance)\b", t):
        company = "National Insurance Co."
    elif re.search(r"\b(oriental insurance)\b", t):
        company = "The Oriental Insurance Co."
    else:
        if domain == "aviation":
            company = "the airline carrier"
        elif domain == "insurance":
            company = "the insurance company"
        elif domain == "ecommerce":
            company = "the e-commerce marketplace seller / platform"
        else:
            company = "the opposite party service provider"

    # 2. Disputed Item / Service
    item = None
    item_match = re.search(
        r"\b(laptop|macbook|phone|smartphone|mobile|iphone|refrigerator|fridge|tv|television|washing machine|air conditioner|ac|watch|smartwatch|shoes|sneakers|dress|shirt|clothing|tablet|ipad|charger|headphone|earbuds|battery|power bank|luggage|baggage|suitcase|bag|flight ticket|flight booking|health insurance policy|mediclaim policy|life policy|motor insurance|hospital bill|claim)\b",
        t
    )
    if item_match:
        item = item_match.group(1)
    else:
        item = "the purchased product / booked service"

    # 3. Monetary Amount
    amount = None
    amt_match = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)|([\d,]+)\s*(?:rs|rupees|inr|lakhs?|lacs?)", t)
    if amt_match:
        amount = amt_match.group(0).strip(",.")

    # 4. Flight / Booking / Reference ID
    ref_id = None
    ref_match = re.search(r"\b([0-9a-z]{2,3}[-\s]?[0-9]{3,4}|pnr\s*[:#]?\s*[0-9a-z]+|order\s*#?\s*[0-9a-z-]+)\b", t)
    if ref_match:
        ref_id = ref_match.group(1).upper()

    return {
        "company": company,
        "item": item,
        "amount": amount,
        "ref_id": ref_id
    }


def detect_sub_issue(text: str, domain: str, coarse_issue: str) -> str:
    """Identify nuanced sub-issue to deliver distinct legal tests within the same sector."""
    t = text.lower()
    if domain == "aviation":
        if re.search(r"\b(damaged bag|broken wheel|torn|cracked bag|damage to luggage|zipper|broken bag|cracked)\b", t):
            return "aviation_baggage_damaged"
        if re.search(r"\b(delayed bag|delayed luggage|baggage arrived late|waiting for bag|bag not received)\b", t):
            return "aviation_baggage_delayed"
        if re.search(r"\b(lost baggage|lost luggage|baggage lost|missing bag|lost suitcase|untraceable|not found)\b", t):
            return "aviation_baggage_lost"
        if coarse_issue == "aviation_baggage":
            return "aviation_baggage_lost"
        if re.search(r"\b(deni\w*\s+.*?boarding|boarding\s+deni\w*|overbook\w*|bumped)\b", t):
            return "aviation_denied_boarding"
        if re.search(r"\b(hotel|overnight|night flight|sleep|stuck at airport|6 hours|7 hours|8 hours|12 hours|24 hours)\b", t) and re.search(r"\b(delay|delayed)\b", t):
            return "aviation_delay_night"
        if re.search(r"\b(5 hours|more than 5 hours|refund on delay)\b", t) and re.search(r"\b(delay|delayed)\b", t):
            return "aviation_delay_major"
        if re.search(r"\b(delay|delayed|delaying|wait|hours|stuck|tarmac|refreshment|food|meal|water)\b", t):
            return "aviation_delay_refreshments"
        if re.search(r"\b(cancel\w*)\b", t):
            return "aviation_cancellation_unnotified"
        if re.search(r"\b(refund|ticket|fare|tax|taxes|credit shell|voucher)\b", t):
            return "aviation_refund"
        return "aviation_general"

    if domain == "ecommerce":
        if re.search(r"\b(harm|injur\w*|fire|explod\w*|burn|shock|hazardous|unsafe|accident|hospital|damage to property)\b", t):
            return "ecommerce_product_liability"
        if re.search(r"\b(fake|counterfeit|replica|spurious|first copy|knockoff|duplicate)\b", t):
            return "ecommerce_spurious"
        if re.search(r"\b(wrong item|wrong product|wrong size|empty box|soap in box|different item|different model)\b", t):
            return "ecommerce_wrong_item"
        if re.search(r"\b(late delivery|delivered late|delayed delivery|past delivery date|not delivered on time)\b", t):
            return "ecommerce_late_delivery"
        if re.search(r"\b(dark pattern|hidden charge|pre-ticked|cancellation fee|opt-out)\b", t):
            return "ecommerce_dark_patterns"
        if re.search(r"\b(policy|citing policy|company policy|store policy|no return|no refund)\b", t):
            return "ecommerce_invoice_policy"
        if re.search(r"\b(defective|damaged|faulty|broken|cracked|screen|battery|compressor|not working|stopped working|dead on arrival)\b", t):
            return "ecommerce_damaged_defective"
        if re.search(r"\b(refus\w*.*(return|refund)|return.*refus\w*|refund.*refus\w*|take back)\b", t):
            return "ecommerce_return_refusal"
        return "ecommerce_damaged_defective"

    if domain == "insurance":
        if re.search(r"\b(cashless|discharge|pre-auth|pre authorization|hospital delay|tpa delay|detained)\b", t):
            return "insurance_cashless_delay"
        if re.search(r"\b(free look|free-look|cooling off|cancel policy|cancellation|mis-sold|mis sold|return policy)\b", t):
            return "insurance_free_look"
        if re.search(r"\b(reject|repudiat\w*|denied|denial|frivolous|pre-existing|clause)\b", t):
            return "insurance_repudiation"
        if re.search(r"\b(delay|delayed|delaying|pending|overdue|unprocessed|tat|penal interest|interest on delay|30 days|45 days|months)\b", t):
            return "insurance_delay_standard"
        return "insurance_delay_standard"

    return "general"


def synthesize_legal_response(
    query: str,
    hits: list[dict[str, Any]],
    domain: str,
    issue: str | None = None
) -> str:
    """Generate structured, factual, case-tailored legal assessment citing exact governing sections and legal tests."""
    if not hits:
        return "I cannot provide a source-grounded legal assessment because no eligible verified legal provisions were retrieved."

    if not issue:
        issue = classify_issue(query, domain)

    facts = extract_complaint_facts(query, domain)
    sub = detect_sub_issue(query, domain, issue)
    company = facts["company"]
    item = facts["item"]
    ref_id = facts["ref_id"]
    amount = facts["amount"]

    item_disp = f"{item} ({ref_id})" if ref_id else item
    amount_disp = f" | Claimed Consideration: {amount}" if amount else ""

    # Top hit excerpt summary for direct grounding
    top_hit = hits[0]
    top_title = top_hit.get("title", "Statutory Regulations")
    top_statute = top_hit.get("statute", "Consumer Protection Law")
    top_excerpt = top_hit.get("excerpt", "").strip().replace("\n\n", "\n")
    top_excerpt_summary = top_excerpt[:280] + "..." if len(top_excerpt) > 280 else top_excerpt

    # --- 1. AVIATION SUB-ISSUES ---
    if sub == "aviation_baggage_lost":
        factual_eval = (
            f"Regarding the lost {item_disp} on {company}: Failure by the carrier to locate and deliver your checked baggage "
            "within established transit periods constitutes an actionable deficiency in service. Domestic carriage baggage liability is statutorily governed by DGCA CAR Part VI."
        )
        primary_section = (
            "- **DGCA CAR Section 3, Series M, Part VI & Carriage by Air Act, 1972 (Third Schedule / Montreal Convention)**: "
            "*Baggage Rules and Statutory Compensation for Lost, Damaged, or Delayed Luggage*"
        )
        legal_test = (
            "- **Domestic Checked Baggage Liability (DGCA CAR Part VI):** Airline liability for lost, delayed, or damaged checked luggage "
            "is statutory compensation up to **₹20,000 per passenger** (calculated at statutory per-kg rates of ₹350–₹450 per kg up to ₹20,000, "
            "or the actual proved value of lost contents). Consumer Commissions regularly award the full value of lost contents plus compensation for mental agony.\n"
            "- **International Baggage Liability (Carriage by Air Act, 1972):** Governed by the Third Schedule (Montreal Convention 1999), capping liability at "
            "**1,288 Special Drawing Rights (SDRs)** (approx. **₹1,40,000 to ₹1,50,000**) per passenger unless a higher value was declared.\n"
            "- **Mandatory Procedural Requirements (PIR):** Passenger MUST file a **Property Irregularity Report (PIR)** before leaving the airport arrival terminal. "
            "Written complaint must be submitted to the airline within **7 days** for damaged baggage, and within **21 days** for delayed baggage."
        )
        supporting = (
            "- **Consumer Protection Act, 2019 — Section 2(11):** Failure by airline staff to safeguard, trace, or deliver checked baggage constitutes actionable Deficiency in Service.\n"
            "- **Consumer Protection Act, 2019 — Section 39:** Empowers Consumer Commissions to order compensation, reimbursement of lost luggage value, and litigation costs."
        )
        actions = (
            f"1. Obtain and safeguard your **Property Irregularity Report (PIR)** reference number issued at the {company} baggage service desk.\n"
            f"2. Submit a formal written claim letter to {company} attaching baggage tags, boarding passes, and an itemized list of contents with valuation receipts.\n"
            f"3. Demand statutory compensation up to ₹20,000 (domestic) or 1,288 SDRs (international) within 14 days of notice."
        )
        tier1 = f"{company} Baggage Services Counter & {company} Nodal Officer / Grievance Officer (quote your PIR reference number)"
        tier2 = "Ministry of Civil Aviation AirSewa Portal (https://airsewa.gov.in/) & DGCA Passenger Grievance Cell"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    elif sub == "aviation_baggage_damaged":
        factual_eval = (
            f"Regarding the damaged {item_disp} on {company}: Delivering checked luggage in a broken, torn, or damaged condition "
            "violates the carrier's statutory bailee duty under carriage regulations. The carrier cannot disclaim liability through arbitrary exclusions."
        )
        primary_section = (
            "- **DGCA CAR Section 3, Series M, Part VI & Carriage by Air Act, 1972 (Third Schedule / Montreal Convention)**: "
            "*Clause 3.2 — Damaged Checked Baggage Liability, Repair Costs, and Replacement Value*"
        )
        legal_test = (
            "- **Carrier Liability for Baggage Damage:** The airline is strictly liable for any damage, destruction, or tearing sustained by checked baggage "
            "while in the carrier's charge onboard the aircraft or during airport ground handling.\n"
            "- **Repair or Replacement Mandate:** Carrier must reimburse actual verified repair costs or provide direct replacement of the damaged suitcase/bag.\n"
            "- **7-Day Mandatory Notice Window:** The passenger MUST submit a written complaint to the airline within **seven (7) days** from the date of receipt of the damaged luggage, supported by the Property Irregularity Report (PIR)."
        )
        supporting = (
            "- **Consumer Protection Act, 2019 — Section 2(11):** Damaging passenger property in transit constitutes gross Deficiency in Service.\n"
            "- **Carriage by Air Act, 1972 — Third Schedule (Rule 17):** Strict liability of carrier for checked baggage damage."
        )
        actions = (
            f"1. File a Property Irregularity Report (PIR) or Damage Report at the airport arrival terminal before exiting.\n"
            f"2. Take high-resolution photographs of damaged wheels, torn fabric, broken zippers, and airline baggage tags.\n"
            f"3. Submit formal claim notice to {company} within 7 days demanding repair or replacement invoice reimbursement."
        )
        tier1 = f"{company} Airport Station Manager & {company} Baggage Services Grievance Officer"
        tier2 = "AirSewa Portal (https://airsewa.gov.in/) & DGCA Passenger Grievance Cell"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    elif sub == "aviation_delay_night":
        factual_eval = (
            f"Regarding the extended overnight flight delay on {company}: Retaining passengers at the airport for over 6 hours during night hours "
            "without providing hotel accommodation constitutes an explicit non-compliance with statutory passenger amenity mandates."
        )
        primary_section = (
            "- **DGCA CAR Section 3, Series M, Part IV — Clause 3.4.2**: "
            "*Flight Delays Exceeding 6 Hours Overnight — Mandatory Free Hotel Accommodation and Ground Transfers*"
        )
        legal_test = (
            "- **Mandatory Hotel Accommodation:** Airlines MUST provide free hotel accommodation and ground transport for flights delayed by more than "
            "six (6) hours scheduled between 20:00 hours and 08:00 hours, or for any delay exceeding twenty-four (24) hours.\n"
            "- **No Forced Airport Stays:** The airline cannot compel passengers to remain seated at departure gates or airport terminals overnight.\n"
            "- **Right to Full Refund:** If total flight delay exceeds 5 hours, the passenger has an unconditional statutory right to cancel the flight and receive a **100% full refund** without any cancellation penalty."
        )
        supporting = (
            "- **DGCA CAR Section 3, Series M, Part II:** 100% refund of all statutory taxes, UDF, ADF, and passenger service fees.\n"
            "- **Consumer Protection Act, 2019 — Section 2(11):** Failure to provide statutory amenities during prolonged disruption constitutes Deficiency in Service."
        )
        actions = (
            f"1. Approach the {company} Duty Manager at the airport and demand written confirmation of delay and hotel accommodation voucher.\n"
            f"2. If {company} fails to provide hotel rooms, retain hotel and taxi receipts for reimbursement.\n"
            f"3. Lodge an AirSewa complaint quoting flight number and boarding pass details."
        )
        tier1 = f"{company} Airport Duty Manager & {company} Nodal Officer"
        tier2 = "AirSewa Portal (https://airsewa.gov.in/) & DGCA Passenger Grievance Cell"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - Consumer Commission"

    elif sub == "aviation_delay_refreshments":
        factual_eval = (
            f"Regarding the flight delay on {company}: Operating airlines are legally bound to supply mandatory free meals and refreshments "
            "scaled to the flight duration and delay threshold."
        )
        primary_section = (
            "- **DGCA CAR Section 3, Series M, Part IV — Clause 3.4.1**: "
            "*Flight Delays — Mandatory Free Meals and Refreshments*"
        )
        legal_test = (
            "- **Mandatory Refreshments:** Free meals and refreshments must be provided: 2+ hrs delay (flights <= 2.5 hrs block time); "
            "3+ hrs delay (flights 2.5 to 5 hrs block time); 4+ hrs delay (all other flights).\n"
            "- **Passenger Information:** Airlines must inform passengers of the delay at least every 30 minutes with updated departure estimates."
        )
        supporting = "- **Consumer Protection Act, 2019 — Section 2(11):** Deficiency in service for denial of statutory passenger amenities."
        actions = (
            f"1. Demand food and beverage coupons from {company} boarding gate agents.\n"
            f"2. Keep food and beverage receipts if carrier denies on-ground refreshments.\n"
            f"3. Submit grievance to {company} customer service citing CAR Part IV Clause 3.4.1."
        )
        tier1 = f"{company} Airport Station Manager & Customer Support"
        tier2 = "AirSewa Portal (https://airsewa.gov.in/) & DGCA"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - Consumer Commission"

    elif sub == "aviation_cancellation_unnotified":
        factual_eval = (
            f"Regarding the cancelled {item_disp} on {company}: Unnotified cancellations deprive passengers of planned travel. "
            "Carriers cannot evade statutory cash compensation unless an extraordinary circumstance (e.g. weather, ATC) is officially proved."
        )
        primary_section = (
            "- **DGCA CAR Section 3, Series M, Part IV — Clause 3.3**: "
            "*Flight Cancellations & Mandatory Compensation*"
        )
        legal_test = (
            "- **Notice Period:** Airline must inform passenger of cancellation at least 2 weeks before departure. If informed between 2 weeks and 24 hours, airline must offer alternate flight or refund.\n"
            "- **Compensation for Unnotified Cancellation:** If cancelled without 24 hours notice, airline must provide alternate flight or 100% refund PLUS statutory compensation: "
            "₹2,000 (block time <= 1 hr), ₹3,000 (1 to 2 hrs), ₹4,000 (> 2 hrs), up to ₹10,000 when alternate flight is not accepted.\n"
            "- **Extraordinary Circumstances Exception:** Compensation is exempt only for extraordinary circumstances beyond airline control (weather, ATC, security). Operational crew shortages do NOT qualify."
        )
        supporting = "- **Consumer Protection Act, 2019 — Section 2(11):** Deficiency in service for unnotified flight disruption."
        actions = (
            f"1. Demand written reason for cancellation from {company}.\n"
            f"2. Reject travel credit shells if you prefer a 100% full bank refund.\n"
            f"3. Claim statutory compensation of ₹2,000 to ₹10,000 based on flight block time under Clause 3.3.2."
        )
        tier1 = f"{company} Nodal Officer / Grievance Officer"
        tier2 = "AirSewa Portal (https://airsewa.gov.in/) & DGCA Passenger Grievance Cell"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - Consumer Commission"

    elif sub == "aviation_denied_boarding":
        factual_eval = (
            f"Regarding denied boarding on {company} despite a valid ticket: Involuntary denied boarding due to airline overbooking "
            "triggers immediate statutory compensation up to ₹20,000 plus full ticket refund options."
        )
        primary_section = (
            "- **DGCA CAR Section 3, Series M, Part IV — Clause 3.2**: "
            "*Denied Boarding Compensation and Alternate Flight Arrangements*"
        )
        legal_test = (
            "- **Call for Volunteers:** Airline must first call for volunteers to surrender seats before denying boarding involuntarily.\n"
            "- **Alternate Flight within 1 Hour:** If passenger is provided an alternate flight departing within 1 hour of original time, no statutory compensation is payable.\n"
            "- **Statutory Compensation for Involuntary Denied Boarding:** If alternate flight is arranged within 24 hours of original departure: **200% of booked one-way basic fare plus airline fuel charge**, up to a maximum of **₹10,000**.\n"
            "- **Severe Delay / Alternate > 24 Hours:** If alternate flight is arranged beyond 24 hours of original departure: **400% of booked one-way basic fare plus airline fuel charge**, up to a maximum of **₹20,000**.\n"
            "- **Right to Decline Alternate Flight:** If passenger declines alternate travel, airline MUST provide a **100% full refund of ticket** PLUS **400% of basic fare + fuel charge (capped at ₹20,000)**."
        )
        supporting = (
            "- **Consumer Protection Act, 2019 — Section 2(11):** Involuntary denied boarding on a confirmed ticket constitutes actionable Deficiency in Service.\n"
            "- **DGCA CAR Section 3, Series M, Part II:** Immediate full refund without any administrative or processing fee deduction."
        )
        actions = (
            f"1. Demand a Denied Boarding Certificate / written endorsement from the {company} Airport Duty Manager.\n"
            f"2. Elect whether to accept alternate flight or demand 100% full refund plus 400% statutory compensation.\n"
            f"3. Escalate to DGCA AirSewa portal attaching confirmed boarding pass and refusal evidence."
        )
        tier1 = f"{company} Airport Duty Manager / Station Manager & {company} Nodal Officer"
        tier2 = "AirSewa Portal (https://airsewa.gov.in/) & DGCA Passenger Grievance Cell"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    elif sub == "aviation_refund":
        factual_eval = (
            f"Regarding ticket refund and deductions by {company}: Airlines are statutorily prohibited from forcing travel vouchers / credit shells "
            "or retaining statutory airport taxes on cancelled bookings."
        )
        primary_section = (
            "- **DGCA CAR Section 3, Series M, Part II — Issue II**: "
            "*Refund of Airline Tickets to Passengers of Delinquent Airlines*"
        )
        legal_test = (
            "- **Timelines:** Refund must be completed within 7 working days for credit card bookings, and immediately for cash bookings.\n"
            "- **100% Refund of Taxes:** On non-refundable tickets, all statutory taxes, User Development Fees (UDF), Airport Development Fees (ADF), and Passenger Service Fees (PSF) must be 100% refunded.\n"
            "- **Fee Cap:** Cancellation fee cannot exceed basic fare + fuel surcharge. No additional cancellation processing fees are permitted.\n"
            "- **No Forced Credit Shells:** Airline cannot force passenger to accept travel vouchers or credit shells without explicit consent."
        )
        supporting = "- **Consumer Protection Act, 2019 — Section 2(47):** Unfair trade practice for retaining statutory taxes or forcing credit shells."
        actions = (
            f"1. Submit written refund demand to {company} for 100% refund of all statutory taxes and fees.\n"
            f"2. Explicitly decline any credit shell or travel voucher in writing.\n"
            f"3. Escalate to AirSewa if refund is not credited within 7 working days."
        )
        tier1 = f"{company} Nodal Officer / Grievance Officer"
        tier2 = "AirSewa Portal (https://airsewa.gov.in/) & DGCA"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - Consumer Commission"

    # --- 2. E-COMMERCE SUB-ISSUES ---
    elif sub == "ecommerce_spurious":
        factual_eval = (
            f"Regarding the counterfeit / spurious {item_disp} received from {company}: Supplying counterfeit replicas instead of genuine authentic "
            "goods is an aggravated statutory offense under consumer protection law, exposing the seller and marketplace to strict liability and CCPA penalties."
        )
        primary_section = (
            "- **Consumer Protection (E-Commerce) Rules, 2020 — Rule 6(2)(b)**: "
            "*Duties of Sellers — Absolute Prohibition of Sale and Return Refusal of Spurious / Counterfeit Goods*\n"
            "- **Consumer Protection Act, 2019 — Section 2(47)(i)(a) & Section 89**: "
            "*Unfair Trade Practice and Penal Action for Manufacture / Sale of Spurious Goods*"
        )
        legal_test = (
            "- **Ban on Spurious Goods (Rule 6(2)(b)):** Sellers and platforms are strictly barred from selling spurious goods or refusing returns of counterfeit products.\n"
            "- **Marketplace Liability:** E-commerce platforms that host or promote spurious goods cannot evade intermediary liability under Section 79 of IT Act if they fail to expeditiously remove spurious listings and refund consumers.\n"
            "- **Penal Consequences (Section 89):** Manufacturing or selling spurious goods attracts CCPA prosecution, fines up to ₹50 Lakhs, and imprisonment."
        )
        supporting = (
            "- **Consumer Protection Act, 2019 — Section 2(47):** Falsely representing spurious products as authentic goods is an Unfair Trade Practice.\n"
            "- **Consumer Protection Act, 2019 — Section 21:** CCPA powers to order recall and mandatory refund."
        )
        actions = (
            f"1. Preserve the product packaging, brand tags, barcodes, invoice, and courier delivery slip.\n"
            f"2. Submit formal return and refund request through {company} alleging counterfeit / spurious delivery.\n"
            f"3. File a complaint on National Consumer Helpline (1915) and CCPA if refund is resisted."
        )
        tier1 = f"{company} Marketplace Grievance Officer (grievance mechanism under Rule 4)"
        tier2 = "National Consumer Helpline (NCH - 1915) & Central Consumer Protection Authority (CCPA)"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    elif sub == "ecommerce_wrong_item":
        factual_eval = (
            f"Regarding the incorrect / non-conforming {item_disp} received from {company}{amount_disp}: Delivering an item that does not match "
            "the characteristics, model, or order placed constitutes non-delivery of contractual consideration and an unfair trade practice."
        )
        primary_section = (
            "- **Consumer Protection (E-Commerce) Rules, 2020 — Rule 6(2)(d)**: "
            "*Prohibition of Return / Refund Refusal for Goods Non-Conforming to Advertised Specifications*\n"
            "- **Consumer Protection Act, 2019 — Section 2(47)(viii)**: "
            "*Unfair Trade Practice: Prohibition of Refusal to Refund Consideration within 30 Days*"
        )
        legal_test = (
            "- **Non-Conforming Goods Bar (Rule 6(2)(d)):** Sellers on e-commerce marketplaces cannot refuse return or refund if delivered goods fail to meet the exact characteristics, specifications, brand, or model ordered.\n"
            "- **Refund Window (Section 2(47)(viii)):** Refusal to take back the wrong goods and refund payment within 30 days constitutes a per se Unfair Trade Practice.\n"
            "- **No Shifting to Seller:** The e-commerce marketplace is legally bound under Rule 4 to provide an active grievance mechanism to resolve wrong shipment disputes."
        )
        supporting = "- **Consumer Protection Act, 2019 — Section 2(11):** Shortcoming in delivery and order fulfillment constitutes actionable Deficiency in Service."
        actions = (
            f"1. Take clear unboxing photos/video showing the package label, tracking number, and wrong product received.\n"
            f"2. Lodge an immediate return ticket on {company} quoting order reference.\n"
            f"3. If return is rejected, escalate to {company} Grievance Officer citing Rule 6(2)(d)."
        )
        tier1 = f"{company} Grievance Officer / Nodal Redressal Team"
        tier2 = "National Consumer Helpline (NCH - 1915 / consumerhelpline.gov.in) & CCPA"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    elif sub in {"ecommerce_return_refusal", "ecommerce_damaged_defective", "ecommerce_invoice_policy"}:
        factual_eval = (
            f"Regarding your {item_disp} purchased on {company}{amount_disp}: The seller's refusal to accept return or refund "
            "of a defective or damaged product—whether citing internal company policy or a 'no return' invoice disclaimer—violates mandatory statutory protections."
        )
        primary_section = (
            "- **Consumer Protection (E-Commerce) Rules, 2020 — Rule 6(2)**: "
            "*Duties of Sellers on Marketplace — Absolute Prohibition of Return and Refund Refusal*\n"
            "- **Consumer Protection Act, 2019 — Section 2(47)(viii)**: "
            "*Unfair Trade Practice: Prohibition of Refusal to Take Back Defective Goods or Refund Consideration within 30 Days*"
        )
        legal_test = (
            "- **Statutory Bar Against Return Refusal (Rule 6(2)):** No seller on an e-commerce marketplace shall refuse to take back goods, "
            "or refuse to discontinue services, or refuse to refund consideration paid, if such goods or services are defective, damaged, deficient, "
            "spurious, delivered late, or non-conforming to the advertised specifications.\n"
            "- **Invalidity of 'No Return' Policies:** Blanket 'No Return' or 'No Refund' clauses printed on invoices or website disclaimers are legally void "
            "and cannot override statutory obligations under Rule 6(2).\n"
            "- **Unfair Trade Practice Bar (Section 2(47)(viii)):** Refusal to withdraw defective goods and refund payment within the period stipulated "
            "in the invoice, or within thirty (30) days if no period is stipulated, constitutes a per se Unfair Trade Practice."
        )
        supporting = (
            "- **Consumer Protection Act, 2019 — Section 2(11):** Statutory Definition of Deficiency in Service (fault, shortcoming, or inadequacy in performance).\n"
            "- **Consumer Protection Act, 2019 — Chapter VI (Sections 84 & 86):** Product Liability holding manufacturer strictly liable for manufacturing defect, "
            "and seller / marketplace liable for selling or altering defective goods.\n"
            "- **Consumer Protection (E-Commerce) Rules, 2020 — Rule 4(4) & 4(5):** Mandatory Grievance Officer on platform (acknowledgment in 48h, resolution in 1 month)."
        )
        actions = (
            f"1. Issue a formal written notice to {company} and the seller quoting Rule 6(2) of E-Commerce Rules, 2020.\n"
            f"2. Explicitly note that company 'no return' disclaimers are statutorily void under Section 2(47)(viii).\n"
            f"3. Demand replacement or 100% refund to original payment source within 7 days."
        )
        tier1 = f"{company} Marketplace Grievance Officer / Nodal Officer (mandatory acknowledgment in 48h, resolution in 1 month under Rule 4)"
        tier2 = "National Consumer Helpline (NCH - 1915 / https://consumerhelpline.gov.in/) & Central Consumer Protection Authority (CCPA)"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    elif sub == "ecommerce_product_liability":
        factual_eval = (
            f"Regarding the hazardous / exploding {item_disp} purchased from {company}: Causing bodily injury, burns, or property damage "
            "triggers strict statutory product liability against the manufacturer, seller, and e-commerce platform under Chapter VI of CPA 2019."
        )
        primary_section = (
            "- **Consumer Protection Act, 2019 — Chapter VI (Sections 82 to 87)**: "
            "*Product Liability for Harm Caused by Defective Products*"
        )
        legal_test = (
            "- **Manufacturer Liability (Section 84):** Product manufacturer is strictly liable for manufacturing defects, design defects, deviation from specifications, "
            "failure to conform to express warranty, or failure to give adequate instructions/warnings.\n"
            "- **Seller / Marketplace Liability (Section 86):** Product seller or e-commerce platform is liable if it exercised control over manufacture/packaging, "
            "altered the product, failed to exercise reasonable care in inspection, or made independent express warranties."
        )
        supporting = "- **Consumer Protection Act, 2019 — Section 2(11):** Deficiency in service for supply of hazardous or non-conforming goods."
        actions = (
            f"1. Secure and preserve the damaged product, fragments, charred components, and medical treatment records.\n"
            f"2. Issue a legal notice to both the manufacturer and {company} claiming medical damages and property loss under Section 84 & 86.\n"
            f"3. File a product liability complaint before the District Consumer Commission via e-Jagriti."
        )
        tier1 = f"Manufacturer Legal Desk & {company} Grievance Officer"
        tier2 = "National Consumer Helpline (NCH - 1915) & Central Consumer Protection Authority (CCPA)"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    elif sub == "ecommerce_dark_patterns":
        factual_eval = (
            f"Regarding the deceptive charges or interface architecture on {company}: Employing pre-ticked opt-in checkboxes, "
            "hidden booking fees, or unilateral cancellation charges violates explicit dark pattern prohibitions."
        )
        primary_section = (
            "- **Consumer Protection (E-Commerce) Rules, 2020 — Rule 4(2) & Rule 4(11)**: "
            "*Prohibition of Unilateral Cancellation Fees and Dark Pattern Architectures*\n"
            "- **Consumer Protection Act, 2019 — Section 2(47)**: "
            "*Unfair Trade Practice: Deceptive Methods, Misleading Representations, and Hidden Fees*"
        )
        legal_test = (
            "- **Prohibition of Unilateral Cancellation Charges (Rule 4(2)):** No e-commerce entity shall impose cancellation fees on consumers entitled to cancel an order, unless equivalent charges are borne by the entity for seller-initiated cancellations.\n"
            "- **Prohibition of Dark Patterns (Rule 4(11) & CCPA Guidelines 2023):** Platforms are prohibited from using deceptive design architecture, pre-ticked opt-in checkboxes, auto-enrolled paid subscriptions, hidden fees, false urgency timers, or forced basket sneaks.\n"
            "- **Grievance Officer Redressal (Rule 4(4) & 4(5)):** Mandatory platform Grievance Officer must acknowledge grievance within 48 hours and resolve within 1 month."
        )
        supporting = "- **Consumer Protection Act, 2019 — Section 21 & Section 89:** CCPA penalties for deceptive trade practices."
        actions = (
            f"1. Capture screenshots of the deceptive checkout flow, pre-ticked checkboxes, and fee breakdown on {company}.\n"
            f"2. Demand full reversal of the unauthorized charge from {company} Grievance Officer.\n"
            f"3. Report dark pattern practice to CCPA via National Consumer Helpline."
        )
        tier1 = f"{company} Platform Grievance Officer"
        tier2 = "National Consumer Helpline (NCH - 1915) & CCPA"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - Consumer Commission"

    # --- 3. INSURANCE SUB-ISSUES ---
    elif sub == "insurance_delay_standard":
        factual_eval = (
            f"Regarding your delayed {item_disp} filed with {company}{amount_disp}: Withholding claim settlement beyond the statutory 30-day "
            "period without lawful justification entitles the policyholder to automatic statutory penal interest."
        )
        primary_section = (
            "- **IRDAI Regulations, 2024 — Regulation 11 & Schedule**: "
            "*Mandatory 30-Day Turnaround Time (TAT) for Insurance Claim Settlement*\n"
            "- **IRDAI Regulations, 2024 — Regulation 14 & Schedule**: "
            "*Penal Interest on Delayed Payment at Bank Rate + 2% per annum*"
        )
        legal_test = (
            "- **Mandatory 30-Day TAT (Regulation 11):** Insurers are statutorily required to settle or repudiate any health, life, or general insurance claim "
            "within thirty (30) days from the receipt of all necessary documents. Where a surveyor/investigation is appointed, the survey report must be submitted within 30 days, "
            "and the claim settled within 30 days thereafter.\n"
            "- **Statutory Penal Interest (Regulation 14):** If the insurer fails to pay claim proceeds within the prescribed 30-day timeline without lawful justification, "
            "it is legally liable to pay penal interest on the claim amount at a rate of **Bank Rate plus 2% per annum**, accruing from the 31st day until the actual date of payment.\n"
            "- **Prohibition of Forced Discharge Vouchers:** Forcing policyholders to sign unconditional discharge vouchers to release payments is voidable under established law."
        )
        supporting = (
            "- **Consumer Protection Act, 2019 — Section 2(11):** Unjustified delay or withholding of claim settlement constitutes actionable Deficiency in Service.\n"
            "- **Insurance Ombudsman Rules, 2017 — Rule 13(1)(b):** Grants jurisdiction to Insurance Ombudsman for delayed settlement of claims."
        )
        actions = (
            f"1. Issue a formal demand notice to {company} Grievance Redressal Officer quoting Regulation 11 & Regulation 14.\n"
            f"2. Calculate penal interest from Day 31 at prevailing Bank Rate + 2% per annum on the claim amount.\n"
            f"3. Escalate to IRDAI Bima Bharosa portal and the Insurance Ombudsman if unresolved within 14 days."
        )
        tier1 = f"{company} Grievance Redressal Officer (GRO) - mandatory internal resolution within 14 days"
        tier2 = "IRDAI Bima Bharosa Portal (https://bimabharosa.irdai.gov.in/) & Insurance Ombudsman (under Insurance Ombudsman Rules, 2017 for claims up to ₹30 Lakhs)"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    elif sub == "insurance_cashless_delay":
        factual_eval = (
            f"Regarding the cashless hospital authorization / discharge delay with {company}: Insurers and TPAs are mandated "
            "under IRDAI Master Circular 2024 to issue cashless approvals within 1 hour and discharge clearances within 3 hours."
        )
        primary_section = (
            "- **IRDAI Regulations, 2024 — Regulation 11 & IRDAI Master Circular, 2024**: "
            "*Mandatory 1-Hour Cashless Pre-Authorization and 3-Hour Hospital Discharge Clearance Mandate*"
        )
        legal_test = (
            "- **1-Hour Pre-Authorization:** Insurers/TPAs must communicate cashless approval decision to the network hospital within one (1) hour of request.\n"
            "- **3-Hour Discharge Clearance:** Final discharge authorization must be granted within three (3) hours of receiving the hospital discharge summary.\n"
            "- **No Patient Detention:** Network hospitals are prohibited from holding patients or forcing cash deposits during authorization wait times."
        )
        supporting = "- **Consumer Protection Act, 2019 — Section 2(11):** Harassment and physical detention due to TPA delays constitutes gross Deficiency in Service."
        actions = (
            f"1. Demand the hospital TPA desk log the exact transmission timestamp of the final bill to {company}.\n"
            f"2. Contact {company} emergency hospital helpdesk quoting policy and pre-auth reference.\n"
            f"3. Lodge immediate complaint on Bima Bharosa for violation of IRDAI Master Circular timelines."
        )
        tier1 = f"{company} Emergency Cashless Grievance Desk & Grievance Redressal Officer (GRO)"
        tier2 = "IRDAI Bima Bharosa Portal & Insurance Ombudsman"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - Consumer Commission"

    elif sub == "insurance_repudiation":
        factual_eval = (
            f"Regarding the claim repudiation by {company}{amount_disp}: Insurers bear the strict legal burden of proof to substantiate "
            "pre-existing disease exclusions with cogent medical evidence and must issue a reasoned order within 30 days."
        )
        primary_section = (
            "- **IRDAI Regulations, 2024 — Regulation 11 & Schedule**: "
            "*Claim Repudiation Standards, Written Reasoned Order, and 30-Day Window*\n"
            "- **Insurance Ombudsman Rules, 2017 — Rule 13(1)(a)**: "
            "*Insurance Ombudsman Jurisdiction over Wrongful Rejection or Repudiation of Claims*"
        )
        legal_test = (
            "- **Mandatory Written Reasoned Rejection (Regulation 11):** Insurers cannot reject claims arbitrarily or verbally. Rejection must be communicated with exact contractual grounds and evidence within 30 days.\n"
            "- **Strict Burden of Proof on Insurer:** Under settled Supreme Court jurisprudence (*Manmohan Nanda v. United India Insurance*), the insurer carries the burden of proving material non-disclosure or exclusion applicability. Pre-existing disease rejections require strict medical nexus proof.\n"
            "- **Ombudsman Remedy (Rule 13):** Complainants can escalate repudiation disputes to the Insurance Ombudsman without advocate fees for claims up to ₹30 Lakhs."
        )
        supporting = (
            "- **Consumer Protection Act, 2019 — Section 2(11):** Arbitrary repudiation constitutes actionable Deficiency in Service.\n"
            "- **IRDAI Regulations, 2024 — Regulation 10:** Mandatory Grievance Redressal Mechanism requiring GRO response within 14 days."
        )
        actions = (
            f"1. Demand a complete written copy of the repudiation letter, surveyor report, and medical opinion from {company}.\n"
            f"2. Obtain treating doctor's certificate refuting any nexus with pre-existing conditions.\n"
            f"3. File a complaint before the Insurance Ombudsman under Rule 13(1)(a) within 1 year of repudiation."
        )
        tier1 = f"{company} Grievance Redressal Officer (GRO) - mandatory review within 14 days"
        tier2 = "Insurance Ombudsman (under Insurance Ombudsman Rules, 2017) & IRDAI Bima Bharosa Portal"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    elif sub == "insurance_free_look":
        factual_eval = (
            f"Regarding cancellation of your policy with {company}: Policyholders possess an unconditional statutory right to cancel "
            "life or health insurance policies within the free-look period and receive a 100% premium refund."
        )
        primary_section = (
            "- **IRDAI Regulations, 2024 — Regulation 8 & 9**: "
            "*30-Day Mandatory Free-Look Cancellation Period and Premium Refund*"
        )
        legal_test = (
            "- **30-Day Window:** Policyholders purchasing insurance electronically or through distance marketing have a mandatory 30-day window (15 days for physical policies) "
            "from policy receipt to review terms and cancel.\n"
            "- **Full Premium Refund:** Insurer must refund the full premium paid, minus only proportionate risk premium for cover period and actual medical examination costs."
        )
        supporting = "- **Consumer Protection Act, 2019 — Section 2(46):** Protection against unfair contract terms and hidden exclusion clauses."
        actions = (
            f"1. Submit written policy cancellation request to {company} within 30 days of receiving the digital policy document.\n"
            f"2. Attach policy schedule and proof of date of receipt.\n"
            f"3. Demand 100% premium refund (less proportionate risk cover) to your bank account."
        )
        tier1 = f"{company} Grievance Redressal Officer (GRO) - resolution within 14 days"
        tier2 = "IRDAI Bima Bharosa Portal & Insurance Ombudsman"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - Consumer Commission"

    # --- 4. GENERAL / FALLBACK ---
    else:
        factual_eval = (
            f"Regarding your grievance concerning {item_disp} with {company}: Service providers and traders are statutorily "
            "required to maintain quality standards and honour representations under the Consumer Protection Act, 2019."
        )
        top_h = hits[0]
        top_statute = top_h.get("statute") or top_h.get("title", "Consumer Protection Act, 2019")
        top_sec = top_h.get("section") or "Section 2(11) (Deficiency in Service)"
        top_title = top_h.get("title", "Governing Statutory Provision")

        primary_section = f"- **{top_statute} — {top_sec}**: *{top_title}*"
        legal_test = (
            f"- **Statutory Test ({top_sec}):** The consumer must establish that the opposite party "
            "failed to maintain the statutory standard of service, violated express terms, or engaged in an act of "
            "omission or negligence resulting in harm, financial loss, or deprivation of service."
        )
        if len(hits) > 1 and hits[1].get("section"):
            h2 = hits[1]
            supporting = (
                f"- **{h2.get('statute', 'Consumer Protection Law')} — {h2.get('section')}:** {h2.get('title', 'Supporting Provision')}.\n"
                f"- **Consumer Protection Act, 2019 — Section 2(11) & Section 39:** Statutory remedies for deficiency in service including refund, replacement, and compensation."
            )
        else:
            supporting = "- **Consumer Protection Act, 2019 — Section 2(11) & Section 39:** Statutory remedies for deficiency in service."

        actions = (
            f"1. Issue a formal written legal notice to {company} specifying the defects/deficiencies and granting 15 days to resolve.\n"
            f"2. Preserve invoices, communications, and photographic evidence.\n"
            f"3. Escalate to the National Consumer Helpline (1915) or file via e-Jagriti."
        )
        tier1 = f"{company} Nodal Grievance Officer"
        tier2 = "National Consumer Helpline (NCH - 1915) or Relevant Sector Regulator"
        tier3 = "e-Jagriti Portal (https://e-jagriti.gov.in/) - District Consumer Disputes Redressal Commission"

    sources_summary = "\n".join(f"[{i+1}] {h.get('title')} ({h.get('statute')} - {h.get('section')})" for i, h in enumerate(hits[:4]))

    # Dynamic Executive Summary tailored to the sector and grievance
    if domain == "aviation":
        summary_text = (
            f"Under DGCA CAR passenger regulations and the Carriage by Air Act, 1972, {company} is subject to strict statutory liability. "
            f"For your {item_disp}, the carrier cannot evade liability through unilateral terms or internal disclaimers. "
            "You possess an enforceable statutory entitlement to immediate relief (statutory compensation up to ₹20,000 domestic / 1,288 SDRs international, "
            "100% full refund of fare and statutory taxes, or mandatory passenger amenities). "
            f"If {company} fails to resolve this grievance satisfactorily, binding recourse is available through the Ministry of Civil Aviation AirSewa portal and e-Jagriti."
        )
    elif domain == "ecommerce":
        summary_text = (
            f"Under the Consumer Protection (E-Commerce) Rules, 2020 and Consumer Protection Act, 2019, {company} and the seller operate under strict legal accountability. "
            f"Any refusal to take back or refund your {item_disp}—or reliance on blanket 'no return' invoice disclaimers—is legally void under Rule 6(2). "
            "Refusing to refund consideration within thirty (30) days is statutorily classified as a per se Unfair Trade Practice under Section 2(47)(viii). "
            "You have an established legal right to a 100% refund to your original payment method or replacement, enforceable via platform grievance channels, the National Consumer Helpline (1915), CCPA, and e-Jagriti."
        )
    elif domain == "insurance":
        summary_text = (
            f"Under IRDAI Regulations, 2024 and established insurance jurisprudence, {company} is statutorily bound by mandatory turnaround times. "
            f"For your {item_disp}, claim processing and settlement must be completed within 30 days of document submission. "
            "Unjustified delays legally entitle you to mandatory penal interest at Bank Rate plus 2% per annum under Regulation 14, while arbitrary repudiation places the strict burden of proof on the insurer. "
            "You can secure binding, fee-free redressal up to ₹30 Lakhs through the Insurance Ombudsman or approach the District Consumer Commission via e-Jagriti."
        )
    else:
        summary_text = (
            f"Under the Consumer Protection Act, 2019, {company} is legally required to maintain statutory service standards and honour contractual commitments. "
            f"The shortcomings concerning your {item_disp} establish actionable deficiency in service under Section 2(11) and unfair trade practices under Section 2(47). "
            "You are statutorily entitled under Section 39 to full refund, replacement, damages for harassment, and litigation costs through the 3-tier escalation framework."
        )

    # Strategic Suggestions & Proactive Steps Beyond Immediate Escalation
    suggestions_text = f"""- **Evidence Archival & Preservation:** Maintain an organized digital dossier containing order invoices, receipts, proof of payment/transaction IDs, unboxing videos/photos, airline boarding passes and baggage tags, or hospital discharge summaries. Export and archive all email threads, support ticket transcripts, and chat logs with {company} before histories expire or get purged.
- **Formal Pre-Litigation Legal Demand Notice:** Before initiating formal court proceedings, serve a formal 15-day written Legal Demand Notice to the registered corporate headquarters and Grievance Officer of {company} via Speed Post / Registered Post AD or formal email. Explicitly cite the statutory provisions, compute any applicable statutory penal interest or compensation, and state that failure to resolve within 15 days will result in filing a consumer complaint seeking punitive damages, mental agony compensation, and litigation costs.
- **Limitation Period Vigilance (Section 69, CPA 2019):** Be mindful of the statutory limitation period: any formal complaint before the District Consumer Disputes Redressal Commission must be filed within **two (2) years** from the date on which the cause of action arose (e.g. date of refusal, flight incident, or repudiation notice).
- **Self-Representation via e-Jagriti (No Advocate Required):** Under the Consumer Protection Act, 2019, consumers do not need to hire an advocate or lawyer. You can easily register, file your verified complaint, upload evidence, pay nominal online court fees, and track hearing dates directly from home via the online **e-Jagriti Portal** (https://e-jagriti.gov.in). Virtual hearings are available across most District and State Commissions.
- **Regulatory & Public Interest Alert to CCPA / Regulators:** If {company} routinely engages in this practice (such as widespread refusal of refunds, systemic dark patterns, or persistent claim delays), submit a formal representation to the **Central Consumer Protection Authority (CCPA)** at `cp-ccpa@nic.in` or the relevant sector regulator (DGCA Passenger Grievance Cell / IRDAI) to prompt regulatory inspections and class penalties under Section 18 of CPA 2019.
- **Automated Complaint Letter Generator:** You can click the **"Draft a complaint letter"** button in this application to auto-generate a structured, formal legal grievance letter pre-populated with your specific facts and statutory citations, ready to send to {company}."""

    answer = f"""### Case-Specific Legal Assessment & Statutory Analysis

**1. Factual Assessment & Complaint Merits:**
- **Identified Enterprise:** {company}
- **Disputed Item / Transaction:** {item_disp}{amount_disp}
- **Case Evaluation:** {factual_eval}

**2. Exact Governing Statutory Section for this Issue:**
{primary_section}

**3. Precise Legal Test & Statutory Criteria:**
{legal_test}

**4. Statutory Grounding from Verified Regulatory Authorities:**
> *"{top_excerpt_summary}"*
> — Verified Official Source: **{top_statute}** ([{top_title}]({top_hit.get('source_url', '')}))

**5. Immediate Action Plan for this Grievance:**
{actions}

**6. Secondary / Supporting Statutory Provisions:**
{supporting}

**7. 3-Tier Escalation Path for this Grievance:**
- **Tier 1 (Internal Grievance / Nodal Officer):** {tier1}
- **Tier 2 (Sector Regulator / Statutory Authority):** {tier2}
- **Tier 3 (Judicial Consumer Commission):** {tier3}

**8. Executive Legal Summary:**
{summary_text}

**9. Strategic Suggestions & Proactive Steps Beyond Immediate Escalation:**
{suggestions_text}

*Citations & Grounding:*
{sources_summary}
[S1] All remedies cited above are grounded strictly in the verified statutory authorities."""
    return answer


async def generate(
    prompt: str,
    hits: list[dict[str, Any]] | None = None,
    domain: str = "general",
    issue: str | None = None,
    user_message: str | None = None
) -> str:
    provider = os.getenv("LLM_PROVIDER", "offline").lower()
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL")
        if not key or not model:
            raise HTTPException(503, "LLM setup incomplete: configure OPENAI_API_KEY and OPENAI_MODEL.")
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
            )
            if r.status_code >= 400:
                raise HTTPException(502, "Configured LLM provider request failed.")
            return r.json()["choices"][0]["message"]["content"]

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL")
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        if not model:
            raise HTTPException(503, "LLM setup incomplete: configure OLLAMA_MODEL.")
        try:
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(
                    f"{base.rstrip('/')}/api/chat",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False
                    }
                )
                r.raise_for_status()
                return r.json()["message"]["content"]
        except httpx.HTTPError:
            raise HTTPException(502, "Configured local model is unavailable.")

    if provider in {"offline", "mock", "template", "rule-based"}:
        msg = user_message or prompt
        return synthesize_legal_response(msg, hits or [], domain, issue=issue)

    raise HTTPException(503, "No language model configured. Set LLM_PROVIDER=openai, ollama, or offline.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_configured": os.getenv("LLM_PROVIDER", "offline").lower() in {"openai", "ollama", "offline", "mock", "template"},
        "vector_store_exists": Path(os.getenv("CHROMA_PATH", "./vector_store")).exists()
    }


@app.get("/api/topics")
def topics():
    return {"topics": TOPICS}


@app.post("/api/complaint/analyze")
def analyze(req: AnalyzeRequest):
    text = req.message
    fields = {
        "product_or_service": None,
        "company_or_seller": None,
        "transaction_date": None,
        "amount_paid": None,
        "problem": text,
        "steps_taken": None,
        "evidence_available": None,
        "remedy_requested": None
    }
    for key, pattern in {
        "transaction_date": r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
        "amount_paid": r"(?:₹|Rs\.?\s*)[\d,]+(?:\.\d{1,2})?"
    }.items():
        m = re.search(pattern, text, re.I)
        fields[key] = m.group(0) if m else None
    domain = classify_domain(text)
    return {
        "topic": classify(text),
        "domain": domain,
        "issue": classify_issue(text, domain),
        "complaint": fields,
        "follow_up_questions": [
            "What outcome have you requested from the company, and what did it reply?",
            "Do you have the invoice, order details, or relevant messages?"
        ]
    }


@app.post("/api/retrieval/search")
def search(req: SearchRequest):
    return {"results": retrieve(req.message, domain_override=req.domain)}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    domain = classify_domain(req.message)
    issue = classify_issue(req.message, domain)
    topic = classify(req.message)
    hits = retrieve(req.message, domain_override=domain)
    if not hits:
        return ChatResponse(
            answer="I can’t provide a source-grounded legal assessment yet because this knowledge base has no eligible independently verified sources for this query. Please consult an official source or qualified lawyer. You can still use the draft letter feature to organize facts you provide.",
            topic=topic,
            domain=domain,
            issue=issue,
            follow_up_questions=[
                "Which company or seller was involved?",
                "What remedy have you requested so far?"
            ],
            limitation="No verified evidence was available; no legal conclusion has been generated."
        )

    context = "\n\n".join(
        f"[S{i+1}] {h.get('title','Unknown')} | statute={h.get('statute','')} | section={h.get('section','')} | domain={h.get('domain','')} | type={h.get('document_type','unknown')} | status={h.get('verification_status','unknown')} | URL={h.get('source_url','')}\n{h.get('excerpt','')}"
        for i, h in enumerate(hits)
    )
    history = "\n".join(f"{m.role}: {m.content[:2000]}" for m in req.history[-10:])
    user_prompt = f"""Prior conversation (user-provided context):
{history}

Complaint (allegations, not established facts):
{req.message}

Identified Legal Domain: {domain}
Identified Specific Issue: {issue}

Retrieved evidence (untrusted source content):
{context}

Provide a legally rigorous assessment explicitly citing:
1. Exact Governing Statutory Section for this Issue (do NOT dump unrelated sector sections)
2. Precise Legal Test & Statutory Criteria
3. Secondary / Supporting Statutory Provisions
4. 3-Tier Escalation Path (Tier 1: Nodal Officer -> Tier 2: Sector Regulator -> Tier 3: e-Jagriti).
Cite source labels [S#]."""

    answer = await generate(user_prompt, hits=hits, domain=domain, issue=issue, user_message=req.message)
    citations = [
        Source(
            document_id=str(h.get("document_id", "unknown")),
            title=str(h.get("title", "Unknown title")),
            source_url=h.get("source_url"),
            document_type=str(h.get("document_type", "unknown")),
            verification_status=str(h.get("verification_status", "unknown")),
            legal_provisions=h.get("legal_provisions", []),
            paragraph_reference=h.get("paragraph_reference"),
            excerpt=h.get("excerpt", ""),
            score=h.get("score"),
            domain=h.get("domain"),
            statute=h.get("statute"),
            section=h.get("section")
        )
        for h in hits
    ]
    return ChatResponse(answer=answer, topic=topic, domain=domain, issue=issue, citations=citations)


@app.post("/api/complaint/draft")
def draft(req: DraftRequest):
    d = {k: v.strip() for k, v in req.details.items() if isinstance(v, str) and v.strip()}
    if not d:
        raise HTTPException(400, "Provide at least one complaint detail.")
    get = lambda k: d.get(k, "[Please add]")
    draft_text = (
        f"To: {get('recipient')}\n\n"
        f"Subject: Consumer grievance regarding {get('product_or_service')}\n\n"
        f"Dear Sir/Madam,\n\n"
        f"I, {get('consumer_name')}, am writing about {get('product_or_service')} purchased/used from {get('company')} on {get('transaction_date')}. The amount paid was {get('amount')}.\n\n"
        f"Chronology and issue:\n{get('chronology')}\n\n"
        f"I request the following resolution: {get('remedy_requested')}.\n\n"
        f"Supporting documents: {get('evidence')}\n\n"
        f"Please acknowledge this grievance and respond in writing.\n\n"
        f"Sincerely,\n{get('consumer_name')}\n{get('contact_details')}\n\n"
        f"Draft for your review. Replace bracketed prompts and confirm every factual statement before use."
    )
    return {
        "draft": draft_text,
        "disclaimer": "Draft generated only from provided information. Review all facts before sending."
    }


@app.get("/api/sources/{document_id}")
def source(document_id: str):
    try:
        import chromadb
        client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "./vector_store"))
        collection = client.get_collection("consumer_law")
        result = collection.get(where={"document_id": document_id}, limit=1, include=["metadatas"])
        if not result.get("metadatas"):
            raise HTTPException(404, "Source metadata was not found in the current vector store.")
        return result["metadatas"][0]
    except ValueError:
        raise HTTPException(404, "Source metadata was not found in the current vector store.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, "Source registry is unavailable.")


def ensure_vector_store():
    db = Path(os.getenv("CHROMA_PATH", "./vector_store"))
    if not db.exists() or not (db / "chroma.sqlite3").exists():
        log.info("Vector store not found. Building verified legal frameworks...")
        try:
            from scripts.ingest_legal_frameworks import ingest
            ingest()
            log.info("Vector store built successfully.")
        except Exception as e:
            log.warning(f"Could not auto-ingest vector store: {e}")


@app.on_event("startup")
def on_startup():
    ensure_vector_store()


# Mount frontend production build if available
frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists() and (frontend_dist / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

