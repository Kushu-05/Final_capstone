import re
from typing import Dict, Any, List
from backend.app.llm.engine import llm_engine

CLASSIFICATION_TAXONOMY = {
    "Food Quality": {
        "keywords": ["cold", "undercooked", "raw", "burnt", "salty", "tasteless", "bland", "stale", "rubbery", "overcooked", "mushy", "spoiled", "sour", "dry", "charred", "taste", "tasted", "flavor", "flavorless", "flavors", "seasoning", "seasoned", "sauce", "food", "inedible", "greasy", "undercooked"],
        "subcategories": ["Temperature Deficiency", "Doneness & Texture", "Flavor & Seasoning", "Freshness"]
    },
    "Staff & Service": {
        "keywords": ["rude", "attitude", "ignored", "server", "waiter", "waitress", "hostess", "manager", "disrespectful", "argument", "unfriendly", "inattentive", "slow service"],
        "subcategories": ["Staff Demeanor", "Front of House Inattentiveness", "Management De-escalation", "Staff Knowledge"]
    },
    "Wait Time": {
        "keywords": ["waited", "minutes", "hour", "hours", "delay", "line", "queue", "slow", "forever", "took so long", "waiting", "backlog"],
        "subcategories": ["Seating Queue", "Kitchen Ticket Pacing", "Beverage Delivery", "Check / Bill Delay"]
    },
    "Cleanliness & Hygiene": {
        "keywords": ["dirty", "filthy", "restroom", "toilet", "hair", "staple", "poisoning", "sick", "vomit", "vomiting", "nausea", "bug", "roach", "fly", "flies", "sticky", "sanitation", "soap"],
        "subcategories": ["Foreign Object Contamination", "Foodborne Illness / Poisoning", "Restroom Sanitation", "Dining Area Cleanliness"]
    },
    "Order Accuracy": {
        "keywords": ["wrong", "missing", "forgot", "allergy", "allergic", "peanut", "gluten", "celiac", "dairy", "substituted", "incorrect", "side missing"],
        "subcategories": ["Allergen Mishandling", "Missing Order Items", "Wrong Entrée Delivered", "Modification Errors"]
    },
    "Pricing & Value": {
        "keywords": ["overpriced", "expensive", "surcharge", "fee", "gratuity", "double-charge", "portion", "rip off", "scam", "bill", "receipt"],
        "subcategories": ["Deceptive Surcharges", "Shrinkflation / Portion Size", "Billing Errors", "Value Discrepancy"]
    },
    "Ambience & Facilities": {
        "keywords": ["loud", "music", "noise", "screaming", "ac", "air condition", "hot", "freezing", "dark", "lighting", "seating", "wobbly", "drafty"],
        "subcategories": ["Acoustics & Loud Music", "HVAC Temperature Control", "Physical Furniture & Seating", "Pest Management"]
    }
}

class ComplaintClassifierAgent:
    """
    Classifies restaurant customer complaints into operational domains,
    subcategories, sentiment polarity, and urgency rating.
    """
    def __init__(self):
        pass

    def classify(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        tokens = re.findall(r"\b\w+\b", text_lower)
        
        category_scores: Dict[str, float] = {}
        matched_tags: Dict[str, List[str]] = {}

        for cat, data in CLASSIFICATION_TAXONOMY.items():
            matches = [kw for kw in data["keywords"] if kw in text_lower]
            matched_tags[cat] = matches
            score = len(matches) * 1.5
            
            # Boost for critical food safety tokens
            if cat == "Cleanliness & Hygiene" and any(k in matches for k in ["poisoning", "staple", "sick", "vomit", "hair"]):
                score += 5.0
            # Boost for allergy
            if cat == "Order Accuracy" and any(k in matches for k in ["allergy", "allergic", "celiac", "gluten"]):
                score += 4.0

            category_scores[cat] = score

        # Determine winner
        best_cat = max(category_scores, key=category_scores.get)
        max_score = category_scores[best_cat]

        # Calculate confidence
        total_score = sum(category_scores.values())
        if total_score > 0:
            confidence = min(0.98, max(0.55, max_score / (total_score + 0.1)))
        else:
            best_cat = "Staff & Service"
            confidence = 0.50

        # Subcategory determination
        subcats = CLASSIFICATION_TAXONOMY[best_cat]["subcategories"]
        sub_cat = subcats[0]
        if "poisoning" in text_lower or "sick" in text_lower:
            sub_cat = "Foodborne Illness / Poisoning"
        elif "staple" in text_lower or "hair" in text_lower:
            sub_cat = "Foreign Object Contamination"
        elif "allergy" in text_lower or "celiac" in text_lower:
            sub_cat = "Allergen Mishandling"
        elif "cold" in text_lower:
            sub_cat = "Temperature Deficiency"
        elif any(w in text_lower for w in ["taste", "flavor", "salty", "bland", "seasoning", "sauce"]):
            sub_cat = "Flavor & Seasoning"
        elif "minutes" in text_lower or "hour" in text_lower:
            sub_cat = "Kitchen Ticket Pacing"
        elif "surcharge" in text_lower or "fee" in text_lower:
            sub_cat = "Deceptive Surcharges"
        elif "attitude" in text_lower or "rude" in text_lower:
            sub_cat = "Staff Demeanor"

        # Sentiment analysis
        negative_words = ["terrible", "awful", "horrible", "bad", "cold", "slow", "rude", "disgusting", "poisoning", "worst", "never", "hate", "inedible", "gross", "foul"]
        positive_words = ["great", "good", "delicious", "excellent", "wonderful", "amazing", "best", "loved", "friendly", "fast", "fresh"]
        
        neg_count = sum(1 for w in tokens if w in negative_words)
        pos_count = sum(1 for w in tokens if w in positive_words)
        
        sentiment_score = (pos_count - neg_count) / max(1, pos_count + neg_count)
        if sentiment_score <= -0.4:
            sentiment = "Very Negative"
        elif sentiment_score < 0.0:
            sentiment = "Negative"
        elif sentiment_score == 0.0:
            sentiment = "Neutral"
        else:
            sentiment = "Positive"

        # Urgency scoring
        if any(w in text_lower for w in ["poisoning", "staple", "hospital", "urgent", "allergy", "celiac", "vomit"]):
            urgency = "Critical"
        elif any(w in text_lower for w in ["cold", "undercooked", "rude", "waited 50", "over an hour", "double-charge"]):
            urgency = "High"
        elif any(w in text_lower for w in ["slow", "noisy", "dim", "lukewarm", "expensive"]):
            urgency = "Medium"
        else:
            urgency = "Low"

        return {
            "primary_category": best_cat,
            "sub_category": sub_cat,
            "confidence": round(confidence, 3),
            "sentiment": sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "urgency": urgency,
            "matched_keywords": matched_tags.get(best_cat, [])
        }

classifier_agent = ComplaintClassifierAgent()
