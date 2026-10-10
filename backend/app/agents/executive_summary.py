import re
import json
from typing import Dict, Any, List, Optional
from backend.app.llm.engine import llm_engine
from backend.app.config import config

class ExecutiveSummaryAgent:
    """
    Executive Review Synthesis & Answer Agent:
    Synthesizes customer review evidence to directly answer the user's operational question.
    Generates:
    1. A cohesive, high-impact one-paragraph answer explaining the reasons behind the feedback.
    2. 2-4 distinct bullet-point operational root causes or contributing factors found in the reviews.
    3. Handles clean records when 0 reviews/complaints are present.
    """
    def __init__(self):
        pass

    def generate(
        self,
        query: str,
        reviews: List[Dict[str, Any]],
        target_restaurant: str = "All Restaurants",
        aspect: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes reviews to answer the query in one paragraph with key reasons.
        """
        clean_target = target_restaurant or "All Restaurants"
        if clean_target.lower() == "all":
            clean_target = "All Restaurants"

        # Case 1: Zero reviews / Clean Record
        if not reviews:
            aspect_phrase = f"regarding {aspect.lower()}" if (aspect and aspect.lower() != "general operations") else ""
            if clean_target != "All Restaurants":
                summary = (
                    f"No customer complaints {aspect_phrase} were found for {clean_target} in the historical dataset. "
                    f"Recorded customer feedback for this location does not show recurring negative reports for this specific inquiry."
                )
                reasons = [
                    f"0 matching complaint records found for {clean_target}",
                    "No recurring negative feedback incidents recorded for this criteria"
                ]
            else:
                summary = (
                    f"No customer complaints {aspect_phrase} were identified across the indexed reviews for query \"{query}\". "
                    f"Please try broadening your search terms or selecting specific categories to inspect historical feedback."
                )
                reasons = [
                    "0 matching complaint records found across indexed reviews",
                    "No recurring negative incidents identified for the current filter criteria"
                ]

            return {
                "summary": summary,
                "reasons": reasons,
                "review_count": 0,
                "aspect": aspect or "No Reviews Found"
            }

        # Case 2: Reviews exist - Attempt LLM synthesis
        active_provider = config.get_active_provider()
        if active_provider != "local":
            llm_res = self._synthesize_via_llm(query, reviews, clean_target, aspect)
            if llm_res and llm_res.get("summary") and llm_res.get("reasons"):
                return llm_res

        # Fallback / Offline Rule-Based Dynamic Synthesizer
        return self._dynamic_fallback_synthesize(query, reviews, clean_target, aspect)

    def _synthesize_via_llm(
        self,
        query: str,
        reviews: List[Dict[str, Any]],
        target_restaurant: str,
        aspect: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        lq = query.lower()
        is_complaint_inquiry = any(w in lq for w in [
            "complain", "complaint", "complaints", "bad", "poor", "slow", "rude", "issue", "problem",
            "low", "rating", "ratings", "drop", "down", "negative", "dissatisfaction", "criticism", "sick", "poison", "delay", "cold"
        ])

        # Prioritize complaint reviews (stars <= 3.0) if inquiry asks about issues/complaints
        if is_complaint_inquiry:
            complaint_reviews = [r for r in reviews if float(r.get("stars", 3.0)) <= 3.0]
            other_reviews = [r for r in reviews if float(r.get("stars", 3.0)) > 3.0]
            ordered_reviews = complaint_reviews + other_reviews
        else:
            ordered_reviews = reviews

        # Format top 4 reviews as concise text evidence for rapid sub-second generation
        evidence_snippets = []
        for i, r in enumerate(ordered_reviews[:4]):
            b_name = r.get("restaurant_name") or target_restaurant
            stars = r.get("stars", 1)
            txt = r.get("text", "").replace("\n", " ").strip()
            if len(txt) > 180:
                txt = txt[:177] + "..."
            evidence_snippets.append(f"Review {i+1} ({b_name}, {stars}★): \"{txt}\"")
        evidence_text = "\n".join(evidence_snippets)

        complaint_guidance = ""
        if is_complaint_inquiry:
            complaint_guidance = (
                "CRITICAL: The manager is specifically inquiring about complaints and negative feedback. "
                "Your summary must directly address and explain the operational complaints, breakdowns, and issues reported. "
                "Do NOT provide positive reviews, praise, or commendations when answering a complaint inquiry.\n"
            )

        system_prompt = (
            "You are an Executive Restaurant Operations Analyst. "
            "Your task is to answer the manager's question by synthesizing the provided customer reviews.\n"
            f"{complaint_guidance}"
            "Requirements:\n"
            "1. 'summary': EXACTLY ONE cohesive, high-impact paragraph (2 to 4 sentences) directly explaining "
            "the operational root causes and reasons behind customer feedback.\n"
            "2. 'reasons': A JSON array of 2 to 3 concise bullet-point reasons (each 6-12 words) explaining the primary causes.\n"
            "3. Ground all statements strictly in the provided reviews without inventing details.\n"
            "Return valid JSON only with keys: 'summary' (string) and 'reasons' (array of strings)."
        )

        prompt = (
            f"Manager's Question: \"{query}\"\n"
            f"Target Restaurant: {target_restaurant}\n"
            f"Operational Aspect: {aspect or 'General Operations'}\n\n"
            f"Customer Review Evidence:\n{evidence_text}\n\n"
            "Synthesize this into a 1-paragraph answer and list of reasons:"
        )

        try:
            raw_resp = llm_engine.generate(prompt=prompt, system_prompt=system_prompt, json_mode=True)
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_resp.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            data = json.loads(cleaned)

            summary = data.get("summary", "").strip()
            reasons = data.get("reasons", [])
            if isinstance(reasons, list) and len(reasons) > 0 and summary:
                # Ensure summary is one paragraph (no extra newlines)
                summary = " ".join([p.strip() for p in summary.split("\n") if p.strip()])
                return {
                    "summary": summary,
                    "reasons": [str(r).strip() for r in reasons if str(r).strip()],
                    "review_count": len(reviews),
                    "aspect": aspect or "Synthesized Operations"
                }
        except Exception as e:
            # Fall back to dynamic offline synthesis
            pass

        return None

    def _dynamic_fallback_synthesize(
        self,
        query: str,
        reviews: List[Dict[str, Any]],
        target_restaurant: str,
        aspect: Optional[str]
    ) -> Dict[str, Any]:
        """
        High-quality NLP rule-based synthesis that parses text patterns, low-star complaints,
        and hospitality aspects to produce a natural 1-paragraph executive answer and key reasons.
        """
        lower_q = query.lower()
        num_reviews = len(reviews)
        target_name_str = target_restaurant if (target_restaurant and target_restaurant != "All Restaurants") else "analyzed locations"

        # Analyze review texts for operational patterns
        all_text = " ".join([r.get("text", "").lower() for r in reviews])
        star_ratings = [r.get("stars", 3) for r in reviews]
        avg_stars = sum(star_ratings) / len(star_ratings) if star_ratings else 3.0
        low_star_count = sum(1 for s in star_ratings if s <= 2)

        # Collect distinct pain points
        pain_points = []
        reasons = []

        if any(w in all_text for w in ["slow", "waited", "wait", "minutes", "hour", "delay", "line", "forever"]):
            pain_points.append("excessive wait times and slow ticket pacing")
            reasons.append("Prolonged waiting times for seating, order fulfillment, and ticket processing")

        if any(w in all_text for w in ["cold", "undercooked", "raw", "burnt", "lukewarm", "taste", "salty", "bland", "terrible", "bad food", "mediocre"]):
            pain_points.append("inconsistent food preparation and temperature control")
            reasons.append("Food quality inconsistencies, including improper serving temperatures and seasoning lapses")

        if any(w in all_text for w in ["rude", "attitude", "ignore", "ignored", "server", "waiter", "hostess", "terrible service", "sucks"]):
            pain_points.append("inattentive or dismissive customer service")
            reasons.append("Staff service breakdowns, inattentive waitstaff, and lack of proactive table check-ins")

        if any(w in all_text for w in ["expensive", "overpriced", "surcharge", "bill", "price", "charge", "cost"]):
            pain_points.append("customer perception of overpriced menu items")
            reasons.append("Perception of high pricing and unfavorable price-to-portion value")

        if any(w in all_text for w in ["wrong", "missing", "forgot", "order"]):
            pain_points.append("kitchen order accuracy errors")
            reasons.append("Order inaccuracy and missing requested modifications between POS and the line")

        if not pain_points:
            pain_points.append("general operational and service execution issues")
            reasons.append("Operational inconsistencies affecting overall guest dining experience")

        # Identify genuine complaint reviews (stars <= 3 or text containing critique keywords)
        crit_keywords = [
            "slow", "wait", "waited", "delay", "rude", "cold", "raw", "bad", "terrible", "awful",
            "overpriced", "wrong", "disappoint", "poor", "sucks", "hate", "issue", "problem",
            "complaint", "mistake", "inedible", "undercooked", "stale"
        ]
        complaint_candidates = [
            r for r in reviews 
            if float(r.get("stars", 3)) <= 3.0 or any(w in r.get("text", "").lower() for w in crit_keywords)
        ]
        
        is_complaint_query = any(w in lower_q for w in [
            "complain", "complaint", "complaints", "bad", "poor", "slow", "rude", "issue", "problem",
            "low", "rating", "drop", "down", "negative", "dissatisfaction", "criticism"
        ])

        # Determine specific topic intent of manager's question
        is_asking_staff = any(w in lower_q for w in ["staff", "waiter", "server", "hostess", "attitude", "rude", "demeanor", "employee", "behavior"])
        is_asking_food = any(w in lower_q for w in ["food", "taste", "cold", "quality", "dish", "meal", "undercooked", "raw", "salty", "flavor"])
        is_asking_service = any(w in lower_q for w in ["service", "slow", "wait", "waited", "waiting", "delay", "late", "time", "speed", "minutes", "hour"])
        is_asking_pricing = any(w in lower_q for w in ["price", "pricing", "expensive", "overpriced", "cost", "bill", "fee", "charge"])
        is_asking_hygiene = any(w in lower_q for w in ["sick", "poison", "ill", "hygiene", "dirty", "clean", "vomit", "hair", "sanitation"])

        # Prioritize relevant reviews and select most pertinent verbatim quote
        topic_keywords = []
        if is_asking_staff:
            topic_keywords.extend(["rude", "attitude", "staff", "server", "waiter", "service", "ignore"])
        elif is_asking_food:
            topic_keywords.extend(["food", "cold", "undercooked", "raw", "taste", "bland", "salty", "terrible", "bad"])
        elif is_asking_service:
            topic_keywords.extend(["slow", "wait", "waited", "minutes", "hour", "delay", "line", "forever"])
        elif is_asking_pricing:
            topic_keywords.extend(["expensive", "overpriced", "price", "bill", "charge", "portion"])
        elif is_asking_hygiene:
            topic_keywords.extend(["sick", "poisoning", "ill", "vomit", "dirty", "hair", "stomach"])

        def _candidate_score(r):
            txt = r.get("text", "").lower()
            stars = float(r.get("stars", 3.0))
            topic_matches = sum(1 for kw in topic_keywords if kw in txt)
            # More topic matches and lower star rating gets highest priority
            return (topic_matches * 10) - stars

        if complaint_candidates:
            critical_review = max(complaint_candidates, key=_candidate_score)
            sample_quote = critical_review.get("text", "").strip()
            if len(sample_quote) > 110:
                sample_quote = sample_quote[:107] + "..."
            has_genuine_complaints = True
        else:
            critical_review = reviews[0] if reviews else {}
            sample_quote = critical_review.get("text", "").strip()
            if len(sample_quote) > 110:
                sample_quote = sample_quote[:107] + "..."
            has_genuine_complaints = False

        # If user explicitly asked about complaints, but all retrieved reviews are purely positive
        if is_complaint_query and not has_genuine_complaints and avg_stars >= 4.0:
            summary = (
                f"Across historical customer reviews for {target_name_str}, no recurring negative complaints were identified. "
                f"Guest sentiment on this topic is overwhelmingly positive with an average customer rating of {avg_stars:.1f} stars, "
                f"with reviewers commending operations: \"{sample_quote}\". "
                f"Standard service delivery and quality benchmarks are currently being maintained."
            )
            reasons = [
                f"No recurring complaint patterns identified across retrieved feedback for {target_name_str}",
                f"Customer sentiment remains positive with average rating of {avg_stars:.1f} stars",
                "Operations and staff execution meet customer expectations"
            ]
            return {
                "summary": summary,
                "reasons": reasons,
                "review_count": num_reviews,
                "aspect": aspect or "Clean Operations"
            }

        # Dynamically reorder pain points and reasons according to query intent
        ordered_pain_points = []
        ordered_reasons = []

        if is_asking_staff:
            ordered_pain_points.append("inattentive or dismissive customer service")
            ordered_reasons.append("Staff service breakdowns, inattentive waitstaff, and lack of proactive table check-ins")
        elif is_asking_food:
            ordered_pain_points.append("inconsistent food preparation and temperature control")
            ordered_reasons.append("Food quality inconsistencies, including improper serving temperatures and seasoning lapses")
        elif is_asking_service:
            ordered_pain_points.append("excessive wait times and slow ticket pacing")
            ordered_reasons.append("Prolonged waiting times for seating, order fulfillment, and ticket processing")
        elif is_asking_pricing:
            ordered_pain_points.append("customer perception of overpriced menu items")
            ordered_reasons.append("Perception of high pricing and unfavorable price-to-portion value")
        elif is_asking_hygiene:
            ordered_pain_points.append("food safety and hygiene concerns")
            ordered_reasons.append("Sanitation lapses and health risk incidents reported by guests")

        for pp in pain_points:
            if pp not in ordered_pain_points:
                ordered_pain_points.append(pp)
        for r in reasons:
            if r not in ordered_reasons:
                ordered_reasons.append(r)

        pain_desc = (
            f"{ordered_pain_points[0]} as well as {ordered_pain_points[1]}" 
            if len(ordered_pain_points) > 1 else (ordered_pain_points[0] if ordered_pain_points else "operational execution issues")
        )

        if is_asking_staff:
            summary = (
                f"Based on customer reviews for {target_name_str}, staff-related complaints focus on "
                f"{pain_desc}. Customers express dissatisfaction with front-of-house interactions, "
                f"highlighting specific experiences like \"{sample_quote}\". "
                f"Addressing hospitality coaching and active manager table check-ins will directly resolve these recurring grievances."
            )
        elif is_asking_food:
            summary = (
                f"An analysis of customer reviews for {target_name_str} reveals that food quality complaints center on "
                f"{pain_desc}. Customers frequently report issues with plate execution and temperature control, "
                f"citing specific feedback like \"{sample_quote}\". Strengthening kitchen station quality checks and line tasting spoons will directly "
                f"prevent these customer grievances."
            )
        elif is_asking_service:
            summary = (
                f"Customer reviews for {target_name_str} indicate that service delays stem from bottlenecks in "
                f"{pain_desc}. Guests repeatedly report waiting longer than expected "
                f"for initial table greetings and dish delivery, with negative reviews emphasizing \"{sample_quote}\". "
                f"Addressing front-of-house table pacing and kitchen pass expediting is essential to resolving these recurring complaints."
            )
        elif is_asking_pricing:
            summary = (
                f"Customer feedback for {target_name_str} shows that pricing dissatisfaction centers on "
                f"{pain_desc}. Reviewers express concerns regarding price-to-portion value and unexpected charges, "
                f"noting experiences such as \"{sample_quote}\". Reviewing menu pricing transparency and portion consistency will mitigate these complaints."
            )
        elif is_complaint_query or any(w in lower_q for w in ["low", "poor", "rating", "rate", "stars", "bad", "complain", "down", "drop"]):
            summary = (
                f"Based on historical customer reviews for {target_name_str}, customer complaints are primarily driven by "
                f"{pain_desc}. Customers specifically express dissatisfaction during their visits, "
                f"frequently highlighting experiences such as \"{sample_quote}\". These operational friction points directly "
                f"undermine guest satisfaction and contribute to an average customer rating of {avg_stars:.1f} stars across the evaluated reviews."
            )
        else:
            summary = (
                f"Based on customer reviews for {target_name_str}, feedback regarding {aspect or 'operations'} highlights "
                f"{pain_desc} as the primary operational challenges. Guests report friction during service, "
                f"reflected in verbatims such as \"{sample_quote}\". Addressing these root causes will improve overall customer satisfaction and prevent negative sentiment."
            )

        # Ensure we have at least 2 distinct reasons
        if len(ordered_reasons) < 2:
            ordered_reasons.append("Pacing gaps between kitchen expediting and server table delivery")

        return {
            "summary": summary,
            "reasons": ordered_reasons[:3],
            "review_count": num_reviews,
            "aspect": aspect or "Operational Analysis"
        }

executive_summary_agent = ExecutiveSummaryAgent()
