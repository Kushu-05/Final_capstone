import re
from typing import Dict, Any, List
from backend.app.llm.engine import llm_engine

class AspectRootCauseAgent:
    """
    Performs Aspect-Based Sentiment Analysis (ABSA) and Root-Cause Diagnostics
    on customer reviews and complaints.
    """
    def __init__(self):
        pass

    def analyze(self, reviews: List[Dict[str, Any]], query: str = "") -> Dict[str, Any]:
        if not reviews:
            return {
                "aspects": [],
                "primary_root_causes": [],
                "contributing_factors": [],
                "evidence_quotes": []
            }

        aspect_counts = {}
        quotes = []
        root_causes = []

        for r in reviews:
            text = r.get("text", "")
            lower_t = text.lower()
            quotes.append({
                "review_id": r.get("review_id", ""),
                "restaurant": r.get("restaurant_name", ""),
                "stars": r.get("stars", 1),
                "text": text[:140] + ("..." if len(text) > 140 else "")
            })

            # Aspect discovery
            if any(w in lower_t for w in ["taste", "flavor", "salty", "bland", "seasoning", "sauce", "tasteless", "burnt", "overcooked", "inedible"]):
                aspect_counts["Food Taste & Recipe Execution"] = aspect_counts.get("Food Taste & Recipe Execution", 0) + 1
            if any(w in lower_t for w in ["cold", "undercooked", "lukewarm"]):
                aspect_counts["Food Temperature Control"] = aspect_counts.get("Food Temperature Control", 0) + 1
            if any(w in lower_t for w in ["wait", "delay", "minute", "hour"]):
                aspect_counts["Kitchen Ticket & Seating Pacing"] = aspect_counts.get("Kitchen Ticket & Seating Pacing", 0) + 1
            if any(w in lower_t for w in ["rude", "attitude", "ignore", "hostess", "server"]):
                aspect_counts["Hospitality & Staff Demeanor"] = aspect_counts.get("Hospitality & Staff Demeanor", 0) + 1
            clean_t = re.sub(r"\b(never|don'?t|can'?t|won'?t)?\s*sick\s+of\b", " ", lower_t)
            if any(w in clean_t for w in ["poisoning", "staple", "hair", "got sick", "dirty", "vomit"]):
                aspect_counts["Sanitation & Food Safety"] = aspect_counts.get("Sanitation & Food Safety", 0) + 1
            if any(w in lower_t for w in ["allergy", "gluten", "wrong", "missing"]):
                aspect_counts["Order Accuracy & Allergen Handling"] = aspect_counts.get("Order Accuracy & Allergen Handling", 0) + 1
            if any(w in lower_t for w in ["surcharge", "overpriced", "fee", "bill"]):
                aspect_counts["Transparent Billing & Pricing"] = aspect_counts.get("Transparent Billing & Pricing", 0) + 1
            if any(w in lower_t for w in ["loud", "music", "ac", "air condition"]):
                aspect_counts["Dining Room Environment & Acoustics"] = aspect_counts.get("Dining Room Environment & Acoustics", 0) + 1

        # Synthesize Root Causes dynamically
        sorted_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)
        
        for aspect, count in sorted_aspects:
            if aspect == "Food Taste & Recipe Execution":
                root_causes.append({
                    "aspect": aspect,
                    "frequency": count,
                    "root_cause": "Inconsistent kitchen recipe execution, over-reduction or scorching of sauces, and station cooks skipping required line tasting spoons prior to service.",
                    "operational_area": "Back of House (BOH) Cook Line & Recipe Standardization",
                    "severity": "High"
                })
            elif aspect == "Food Temperature Control":
                root_causes.append({
                    "aspect": aspect,
                    "frequency": count,
                    "root_cause": "Line cook queue delay paired with delayed runner pickup at the pass window. Food is sitting exposed before server delivery.",
                    "operational_area": "Back of House (BOH) & Expediting",
                    "severity": "High"
                })
            elif aspect == "Kitchen Ticket & Seating Pacing":
                root_causes.append({
                    "aspect": aspect,
                    "frequency": count,
                    "root_cause": "Concurrent wave seating at front desk overwhelming fry and grill stations without pacing throttle on tickets.",
                    "operational_area": "Front of House (FOH) Host Station & Kitchen Line",
                    "severity": "High"
                })
            elif aspect == "Hospitality & Staff Demeanor":
                root_causes.append({
                    "aspect": aspect,
                    "frequency": count,
                    "root_cause": "High server table loads during rush hours causing stress and burnout; absence of structured de-escalation protocols.",
                    "operational_area": "Front of House (FOH) Floor Management",
                    "severity": "Medium"
                })
            elif aspect == "Sanitation & Food Safety":
                root_causes.append({
                    "aspect": aspect,
                    "frequency": count,
                    "root_cause": "Critical lapse in prep table temperature logging and cross-contamination sanitization between raw and ready-to-eat stations.",
                    "operational_area": "Kitchen Quality Assurance (QA) & HACCP",
                    "severity": "Critical"
                })
            elif aspect == "Order Accuracy & Allergen Handling":
                root_causes.append({
                    "aspect": aspect,
                    "frequency": count,
                    "root_cause": "Verbal modification breakdowns between waitstaff POS entry and cook tickets; lack of distinct allergen-flagged plates.",
                    "operational_area": "Point of Sale (POS) & Line Expeditor",
                    "severity": "High"
                })
            elif aspect == "Transparent Billing & Pricing":
                root_causes.append({
                    "aspect": aspect,
                    "frequency": count,
                    "root_cause": "Automatic surcharge line items not proactively communicated by waitstaff before presenting the guest folio.",
                    "operational_area": "Front of House / Accounting",
                    "severity": "Medium"
                })
            elif aspect == "Dining Room Environment & Acoustics":
                root_causes.append({
                    "aspect": aspect,
                    "frequency": count,
                    "root_cause": "Sound reflection on hard surfaces without acoustic dampening; delayed HVAC maintenance scheduling.",
                    "operational_area": "Facilities & Ambience",
                    "severity": "Low"
                })

        return {
            "top_aspects": [{"aspect": a, "occurrences": c} for a, c in sorted_aspects],
            "observed_facts": [
                f"Customer reviews report {c} occurrence(s) relating directly to '{a}'."
                for a, c in sorted_aspects
            ],
            "inferred_root_causes": root_causes,
            "root_causes": root_causes,
            "evidence_quotes": quotes[:5],
            "summary": f"Diagnosed {len(sorted_aspects)} distinct operational friction aspects across {len(reviews)} retrieved review cases."
        }

aspect_root_cause_agent = AspectRootCauseAgent()
