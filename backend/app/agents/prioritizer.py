import re
from typing import Dict, Any, List

class ComplaintPrioritizerAgent:
    """
    Computes mathematical prioritization scores for restaurant complaints based on:
    - Clinical/Safety Severity (foodborne illness, contamination, allergens)
    - Operational Urgency
    - Frequency of recurring incidents
    - Star rating penalty
    - Recency / Escalation Risk
    """
    def __init__(self):
        pass

    def prioritize(self, complaint: Dict[str, Any], frequency_count: int = 1) -> Dict[str, Any]:
        text = complaint.get("text", "").lower()
        stars = complaint.get("stars", 1)
        urgency_label = complaint.get("urgency", "Medium")
        category = complaint.get("category", "")

        # 1. Base Severity Assessment (0 - 100)
        # Exclude idiomatic 'sick of' and verify true clinical/foodborne illness or foreign objects
        text_no_idioms = re.sub(r"\b(never|don'?t|can'?t|won'?t)?\s*sick\s+of\b", " ", text)
        is_hazard = any(w in text_no_idioms for w in [
            "poisoning", "food poisoning", "staple", "broken glass", "glass in",
            "hospital", "urgent care", "vomit", "vomited", "threw up", "got sick", "violent illness"
        ])
        is_allergen = any(w in text for w in ["allergy", "allergic", "celiac", "peanuts", "nuts", "anaphylaxis"])
        
        if is_hazard or is_allergen:
            severity_score = 95.0
            critical_flag = "Critical Health & Safety Violation"
        elif any(w in text for w in ["raw chicken", "undercooked pork", "roach", "maggot", "foul smell"]):
            severity_score = 85.0
            critical_flag = "Severe Food Hygiene Defect"
        elif any(w in text for w in ["cold food", "rude", "screamed", "swore", "charged extra", "double-charge"]):
            severity_score = 65.0
            critical_flag = None
        elif any(w in text for w in ["waited 50", "over an hour", "forgot order", "missing"]):
            severity_score = 55.0
            critical_flag = None
        else:
            severity_score = 35.0
            critical_flag = None

        # 2. Urgency Score (0 - 100)
        urgency_map = {"Critical": 100.0, "High": 75.0, "Medium": 50.0, "Low": 25.0}
        urgency_score = urgency_map.get(urgency_label, 50.0)
        if is_hazard or is_allergen:
            urgency_score = 100.0

        # 3. Frequency Boost (0 - 100)
        # Recurring complaints in same category scale up urgency
        frequency_score = min(100.0, frequency_count * 18.0)

        # 4. Rating Penalty (0 - 100)
        # 1 star -> 100, 2 stars -> 75, 3 stars -> 50, 4 stars -> 25, 5 stars -> 0
        rating_penalty = max(0.0, (5.0 - float(stars)) * 25.0)

        # 5. Composite Prioritization Formula
        # Weights: Severity (35%), Urgency (25%), Rating Penalty (20%), Frequency (20%)
        composite_score = (
            (severity_score * 0.35) +
            (urgency_score * 0.25) +
            (rating_penalty * 0.20) +
            (frequency_score * 0.20)
        )

        # Force P0 if critical health/safety
        if is_hazard or is_allergen:
            composite_score = max(composite_score, 92.0)

        composite_score = round(min(100.0, composite_score), 1)

        # Tier assignment and SLA
        if composite_score >= 85.0:
            tier = "P0 - Critical"
            sla = "Immediate action (< 1 hour): Direct manager contact & safety hold"
            action_color = "red"
        elif composite_score >= 65.0:
            tier = "P1 - High"
            sla = "Within 4 hours: Shift leader investigation & guest compensation"
            action_color = "orange"
        elif composite_score >= 40.0:
            tier = "P2 - Medium"
            sla = "Within 24 hours: Operational coaching & station check"
            action_color = "yellow"
        else:
            tier = "P3 - Low"
            sla = "Weekly review: General service monitoring"
            action_color = "blue"

        return {
            "priority_score": composite_score,
            "priority_tier": tier,
            "sla": sla,
            "action_color": action_color,
            "critical_hazard_flag": critical_flag,
            "score_breakdown": {
                "severity_score": round(severity_score, 1),
                "urgency_score": round(urgency_score, 1),
                "frequency_score": round(frequency_score, 1),
                "rating_penalty": round(rating_penalty, 1)
            },
            "rationale": f"Classified as {tier} based on severity ({severity_score}/100) and star rating ({stars}/5)." + 
                         (f" ALERT: {critical_flag} triggered." if critical_flag else "")
        }

prioritizer_agent = ComplaintPrioritizerAgent()
