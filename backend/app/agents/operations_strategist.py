from typing import Dict, Any, List
from backend.app.llm.engine import llm_engine

class OperationsStrategistAgent:
    """
    Formulates actionable operational improvement recommendations, SOPs,
    preventative maintenance schedules, and staff training plans based on diagnosed root causes.
    """
    def __init__(self):
        pass

    def recommend(self, root_cause_data: Dict[str, Any], query: str = "", restaurant_name: str = "") -> Dict[str, Any]:
        root_causes = root_cause_data.get("root_causes", [])
        top_aspects = root_cause_data.get("top_aspects", [])

        immediate_actions = []
        preventative_sops = []
        staff_coaching = []
        kpis = []

        if not root_causes:
            immediate_actions.append("Conduct a 15-minute operational check-in across all line stations and service staff.")
            preventative_sops.append("Standardize shift handover logbook entries for consistent service monitoring.")
            staff_coaching.append("Deliver quarterly refresher training on guest satisfaction and active listening.")
            kpis.append({"metric": "Overall Star Rating", "target": "Maintain >= 4.2 / 5.0"})
        else:
            for rc in root_causes[:3]:
                aspect = rc["aspect"]
                if "Taste" in aspect or "Recipe" in aspect or "Seasoning" in aspect:
                    immediate_actions.append(
                        "Mandate head chef station tasting spoons across all sauce reductions, stocks, and hot entrees prior to service kickoff."
                    )
                    preventative_sops.append(
                        "SOP-KIT-105: Standardized laminated recipe card specifications with exact gram-scale seasonings at every prep and sauté station."
                    )
                    staff_coaching.append(
                        "Line cook sauce reduction and seasoning calibration training; blind-taste testing during pre-shift line meetings."
                    )
                    kpis.append({"metric": "Food Taste & Flavor Satisfaction", "target": ">= 95% Positive Dish Feedback"})

                elif "Temperature" in aspect or "Food Quality" in aspect:
                    immediate_actions.append(
                        "Calibrate pass-window warming stations immediately and enforce a strict 90-second runner pickup rule for all hot entrees."
                    )
                    preventative_sops.append(
                        "SOP-FQ-101: Install digital temperature probing before plating; any dish under 145°F at the window must be remade immediately."
                    )
                    staff_coaching.append(
                        "Chef-led training on plate assembly pacing and communication between saute line and expo window."
                    )
                    kpis.append({"metric": "Food Temperature Incident Complaints", "target": "< 1% of total orders"})

                elif "Ticket" in aspect or "Wait Time" in aspect:
                    immediate_actions.append(
                        "Institute a 10-table seating wave limit at the host stand to prevent sudden surges on kitchen ticket printers."
                    )
                    preventative_sops.append(
                        "SOP-OPS-202: Deploy automatic kitchen display system (KDS) visual pacing timers (green < 12m, yellow 12-18m, red > 18m)."
                    )
                    staff_coaching.append(
                        "Front-of-house host team training on realistic quote times and proactive guest beverage delivery during waits."
                    )
                    kpis.append({"metric": "Average Kitchen Ticket Time", "target": "< 16 minutes during peak dinner"})

                elif "Hospitality" in aspect or "Staff" in aspect:
                    immediate_actions.append(
                        "Cap maximum table sections to 4 tables per server during peak Friday/Saturday services to prevent server burnout."
                    )
                    preventative_sops.append(
                        "SOP-FOH-305: Implement the '2-minute / 2-bite' table touch standard following meal delivery."
                    )
                    staff_coaching.append(
                        "Interactive hospitality workshop focusing on service recovery, de-escalation, and non-defensive apology techniques."
                    )
                    kpis.append({"metric": "FOH Staff Hospitality Rating", "target": ">= 4.6 / 5.0 CSAT"})

                elif "Sanitation" in aspect or "Safety" in aspect:
                    immediate_actions.append(
                        "IMMEDIATE MANDATE: Quarantine flagged ingredient batch, sanitize all prep surfaces, and conduct mandatory safety briefing."
                    )
                    preventative_sops.append(
                        "SOP-HACCP-401: Enforce two-tier temperature logging every 2 hours and hourly restroom sanitization sign-off sheets."
                    )
                    staff_coaching.append(
                        "Mandatory ServSafe food safety re-certification for all kitchen and prep team members."
                    )
                    kpis.append({"metric": "Health & Safety Non-Compliance Reports", "target": "Zero Tolerance (0 incidents)"})

                elif "Order Accuracy" in aspect or "Allergen" in aspect:
                    immediate_actions.append(
                        "Require double-check expo verification between kitchen tickets and plated dishes before any tray leaves the pass."
                    )
                    preventative_sops.append(
                        "SOP-POS-502: Use distinctive colored plates and bright red allergy flags for any order with dietary modifications."
                    )
                    staff_coaching.append(
                        "Server allergy protocol drill: repeat all modifications back to guests at the table and to the expo chef."
                    )
                    kpis.append({"metric": "Order Modification Accuracy Rate", "target": ">= 99.2%"})

                elif "Billing" in aspect:
                    immediate_actions.append(
                        "Instruct servers to verbally point out house policy surcharges or auto-gratuities before presenting payment terminals."
                    )
                    preventative_sops.append(
                        "SOP-ADM-601: Clear, bold print disclosure on physical menus and digital order screens regarding all fees."
                    )
                    staff_coaching.append("Brief waitstaff on clear, polite explanations when guests ask about service charge line items.")
                    kpis.append({"metric": "Billing Dispute Frequency", "target": "< 0.5% of receipts"})

        return {
            "restaurant_target": restaurant_name or "General Operations",
            "strategic_summary": f"Targeted operational strategy addressing {len(immediate_actions)} actionable interventions to stabilize service and elevate satisfaction.",
            "immediate_actions": immediate_actions,
            "preventative_sops": preventative_sops,
            "staff_coaching": staff_coaching,
            "success_kpis": kpis,
            "estimated_implementation_effort": "Low to Moderate (Floor procedural updates, no capital kitchen expenditure required)"
        }

operations_strategist_agent = OperationsStrategistAgent()
