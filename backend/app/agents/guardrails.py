import re
from typing import Dict, Any, Tuple

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"disregard all prior instructions",
    r"reveal (the )?(system )?prompt",
    r"you are now in DAN mode",
    r"system prompt overrides",
    r"jailbreak",
    r"drop database",
    r"drop table",
    r"delete from",
    r"<script[\s\S]*?>[\s\S]*?<\/script>",
]

RESTAURANT_DOMAIN_KEYWORDS = [
    "food", "service", "wait", "time", "order", "staff", "chef", "server", "waiter", "waitress",
    "table", "restaurant", "menu", "pasta", "burger", "pizza", "bar", "kitchen", "hygiene",
    "clean", "dirty", "sick", "poisoning", "rating", "review", "complaint", "drink", "cocktail",
    "manager", "customer", "price", "bill", "check", "surcharge", "tip", "experience", "dish",
    "atmosphere", "ambience", "noise", "loud", "dining", "allergy", "hair", "steak", "cold",
    "ramen", "bbq", "taco", "chinese", "italian", "wine", "seafood", "hostess", "seating", "delay"
]

class InputValidationGuardrails:
    """
    Input validation guardrails:
    1. Length and structure boundaries
    2. Prompt injection and jailbreak attempt filtration
    3. Off-topic domain boundary evaluation
    4. Input sanitization
    """
    def __init__(self, min_len: int = 3, max_len: int = 1200):
        self.min_len = min_len
        self.max_len = max_len

    def validate_and_sanitize(self, query: str) -> Tuple[bool, str, Dict[str, Any]]:
        raw = query.strip()
        report = {
            "passed": True,
            "length_check": True,
            "injection_check": True,
            "domain_relevance_check": True,
            "flags": []
        }

        # 1. Length check
        if len(raw) < self.min_len:
            report["passed"] = False
            report["length_check"] = False
            report["flags"].append(f"Query is too short (min {self.min_len} characters required).")
            return False, raw, report

        if len(raw) > self.max_len:
            report["passed"] = False
            report["length_check"] = False
            report["flags"].append(f"Query exceeds maximum allowed length of {self.max_len} characters.")
            return False, raw[:self.max_len], report

        # 2. Prompt injection detection
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, raw, re.IGNORECASE):
                report["passed"] = False
                report["injection_check"] = False
                report["flags"].append("Potential prompt injection or prohibited security directive detected.")
                return False, raw, report

        # 3. Domain relevance check
        # Check if the query contains at least one restaurant/hospitality related token or generic operational query
        tokens = re.findall(r"\b\w+\b", raw.lower())
        domain_match = any(token in RESTAURANT_DOMAIN_KEYWORDS for token in tokens)
        
        # If very generic query like "what are complaints?" or "summary"
        general_operational_terms = ["complaint", "similar", "issue", "trend", "problem", "solve", "recommend", "action"]
        op_match = any(term in tokens for term in general_operational_terms)

        if not domain_match and not op_match and len(tokens) > 3:
            report["domain_relevance_check"] = False
            report["flags"].append("Notice: Query may be outside the restaurant operations and customer review domain.")
            # Note: We flag as advisory rather than hard rejection for query flexibility

        # 4. Sanitization: remove malicious HTML tags or script blocks
        sanitized = re.sub(r"<[^>]*>", "", raw)
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", sanitized)

        return True, sanitized, report

guardrails = InputValidationGuardrails()
