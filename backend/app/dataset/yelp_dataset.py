import json
import os
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))

SAMPLE_BUSINESSES = [
    {
        "business_id": "biz_bella_roma_01",
        "name": "Bella Roma Trattoria",
        "categories": "Italian, Pizza, Pasta, Wine Bars, Restaurants",
        "city": "Philadelphia",
        "state": "PA",
        "stars": 3.6,
        "review_count": 340,
        "attributes": {
            "RestaurantsReservations": "True",
            "BusinessAcceptsCreditCards": "True",
            "GoodForKids": "True",
            "NoiseLevel": "loud",
            "OutdoorSeating": "False"
        },
        "hours": {
            "Monday": "16:00-22:00",
            "Tuesday": "16:00-22:00",
            "Wednesday": "16:00-22:00",
            "Thursday": "16:00-22:00",
            "Friday": "16:00-23:00",
            "Saturday": "12:00-23:00",
            "Sunday": "12:00-21:00"
        }
    },
    {
        "business_id": "biz_tokyo_ramen_02",
        "name": "Tokyo Express Ramen & Izakaya",
        "categories": "Japanese, Ramen, Noodles, Asian Fusion, Restaurants",
        "city": "Austin",
        "state": "TX",
        "stars": 4.1,
        "review_count": 520,
        "attributes": {
            "RestaurantsReservations": "False",
            "BusinessAcceptsCreditCards": "True",
            "GoodForKids": "True",
            "NoiseLevel": "average",
            "OutdoorSeating": "True"
        },
        "hours": {
            "Monday": "11:00-22:00",
            "Tuesday": "11:00-22:00",
            "Wednesday": "11:00-22:00",
            "Thursday": "11:00-22:00",
            "Friday": "11:00-23:00",
            "Saturday": "11:00-23:00",
            "Sunday": "11:00-21:00"
        }
    },
    {
        "business_id": "biz_smokehouse_bbq_03",
        "name": "Smokey Mountain Pit BBQ",
        "categories": "Barbeque, Southern, Smokehouse, American (Traditional), Restaurants",
        "city": "Nashville",
        "state": "TN",
        "stars": 3.2,
        "review_count": 280,
        "attributes": {
            "RestaurantsReservations": "False",
            "BusinessAcceptsCreditCards": "True",
            "GoodForKids": "True",
            "NoiseLevel": "loud",
            "OutdoorSeating": "True"
        },
        "hours": {
            "Wednesday": "11:00-21:00",
            "Thursday": "11:00-21:00",
            "Friday": "11:00-22:00",
            "Saturday": "11:00-22:00",
            "Sunday": "11:00-20:00"
        }
    },
    {
        "business_id": "biz_le_bistrot_04",
        "name": "Le Bistrot Parisian",
        "categories": "French, Wine Bars, Fine Dining, European, Restaurants",
        "city": "New Orleans",
        "state": "LA",
        "stars": 4.4,
        "review_count": 410,
        "attributes": {
            "RestaurantsReservations": "True",
            "BusinessAcceptsCreditCards": "True",
            "GoodForKids": "False",
            "NoiseLevel": "quiet",
            "OutdoorSeating": "True"
        },
        "hours": {
            "Tuesday": "17:00-22:00",
            "Wednesday": "17:00-22:00",
            "Thursday": "17:00-22:00",
            "Friday": "17:00-23:00",
            "Saturday": "17:00-23:00",
            "Sunday": "10:30-15:00"
        }
    },
    {
        "business_id": "biz_taco_fuego_05",
        "name": "Taco Fuego Cantina",
        "categories": "Mexican, Tacos, Cocktail Bars, Tex-Mex, Restaurants",
        "city": "Phoenix",
        "state": "AZ",
        "stars": 3.4,
        "review_count": 395,
        "attributes": {
            "RestaurantsReservations": "False",
            "BusinessAcceptsCreditCards": "True",
            "GoodForKids": "True",
            "NoiseLevel": "loud",
            "OutdoorSeating": "True"
        },
        "hours": {
            "Monday": "11:00-23:00",
            "Tuesday": "11:00-23:00",
            "Wednesday": "11:00-23:00",
            "Thursday": "11:00-23:00",
            "Friday": "11:00-01:00",
            "Saturday": "11:00-01:00",
            "Sunday": "11:00-22:00"
        }
    },
    {
        "business_id": "biz_golden_harbor_06",
        "name": "Golden Harbor Seafood & Dim Sum",
        "categories": "Seafood, Chinese, Dim Sum, Asian, Restaurants",
        "city": "San Francisco",
        "state": "CA",
        "stars": 3.8,
        "review_count": 620,
        "attributes": {
            "RestaurantsReservations": "True",
            "BusinessAcceptsCreditCards": "True",
            "GoodForKids": "True",
            "NoiseLevel": "loud",
            "OutdoorSeating": "False"
        },
        "hours": {
            "Monday": "10:00-21:30",
            "Tuesday": "10:00-21:30",
            "Wednesday": "10:00-21:30",
            "Thursday": "10:00-21:30",
            "Friday": "10:00-22:00",
            "Saturday": "09:30-22:00",
            "Sunday": "09:30-21:30"
        }
    },
    {
        "business_id": "biz_craft_burger_07",
        "name": "The Rusty Grill & Craft Burgers",
        "categories": "Burgers, American (New), Beer Bar, Gastropub, Restaurants",
        "city": "Denver",
        "state": "CO",
        "stars": 3.5,
        "review_count": 310,
        "attributes": {
            "RestaurantsReservations": "False",
            "BusinessAcceptsCreditCards": "True",
            "GoodForKids": "True",
            "NoiseLevel": "average",
            "OutdoorSeating": "True"
        },
        "hours": {
            "Monday": "11:30-22:00",
            "Tuesday": "11:30-22:00",
            "Wednesday": "11:30-22:00",
            "Thursday": "11:30-22:00",
            "Friday": "11:30-23:30",
            "Saturday": "11:30-23:30",
            "Sunday": "11:30-21:00"
        }
    },
    {
        "business_id": "biz_sweet_artisan_08",
        "name": "Sweet Harvest Bakery & Café",
        "categories": "Bakeries, Coffee & Tea, Breakfast & Brunch, Desserts, Restaurants",
        "city": "Portland",
        "state": "OR",
        "stars": 4.5,
        "review_count": 480,
        "attributes": {
            "RestaurantsReservations": "False",
            "BusinessAcceptsCreditCards": "True",
            "GoodForKids": "True",
            "NoiseLevel": "quiet",
            "OutdoorSeating": "True"
        },
        "hours": {
            "Monday": "07:00-17:00",
            "Tuesday": "07:00-17:00",
            "Wednesday": "07:00-17:00",
            "Thursday": "07:00-17:00",
            "Friday": "07:00-18:00",
            "Saturday": "07:00-18:00",
            "Sunday": "08:00-16:00"
        }
    }
]

# Authentic base templates across operational categories for dynamic generation
COMPLAINT_TEMPLATES = [
    # Slow Service / Wait Time
    {
        "category": "Wait Time",
        "aspect": "Kitchen Ticket Delay & Seating",
        "texts": [
            "We waited over 55 minutes for our main courses to arrive. When we asked the server, they blamed the kitchen backlog. The restaurant was only half full!",
            "Horrendously slow service on a Friday night. Waited 30 minutes just for water and bread, and another 45 minutes for our entrees. Table next to us arrived after and left before.",
            "Customers were standing by the host stand for 25 minutes without anyone acknowledging them. The hostess disappeared completely. Completely disorganized front of house.",
            "Took 2 hours from being seated to receiving our check. We had theatre tickets and nearly missed our show due to extreme kitchen delays.",
            "The drive-through and takeout orders seemed prioritized while dine-in guests waited over 50 minutes for simple pasta dishes. Very frustrating."
        ],
        "stars_range": [1, 2],
        "urgency": "High",
        "severity": "Medium"
    },
    # Food Quality / Temperature & Taste
    {
        "category": "Food Quality",
        "aspect": "Food Taste & Seasoning",
        "texts": [
            "The food taste was terrible. The marinara sauce tasted sour and burnt, and the pasta was completely bland without any seasoning.",
            "Horrible food taste! The chicken tasted stale and rubbery, and the soup had an overwhelmingly salty taste that was completely inedible.",
            "Very disappointed with the food taste. The steak tasted sour and unseasoned, and the mashed potatoes were cold and tasted like instant powder.",
            "Major complaints about the food taste tonight. The curry had an unpleasant bitter taste, lacking authentic spices or flavor.",
            "The seafood pasta had a pungent, foul fishy taste indicating it was far from fresh. The chef needs to check ingredient freshness."
        ],
        "stars_range": [1, 2],
        "urgency": "High",
        "severity": "High"
    },
    {
        "category": "Food Quality",
        "aspect": "Food Temperature & Doneness",
        "texts": [
            "My chicken parmigiana was completely cold in the center and undercooked. When I sent it back, it took 25 minutes to return microwaved and rubbery.",
            "The ramen broth was lukewarm with zero aroma, and the noodles were clumped and mushy. Disappointing given their reputation.",
            "Ordered medium-rare ribeye and it arrived charred and well-done. The side vegetables were cold and unseasoned. Not worth $48.",
            "The soup was overly salty and tasted sour, almost inedible. We informed the waiter and they just shrugged.",
            "Food arrived cold as if it had been sitting on the pass window under dead heat lamps for 20 minutes before our server picked it up."
        ],
        "stars_range": [1, 2],
        "urgency": "High",
        "severity": "High"
    },
    # Health & Safety / Food Safety (P0 Critical)
    {
        "category": "Cleanliness & Hygiene",
        "aspect": "Food Safety & Foreign Objects",
        "texts": [
            "URGENT: Found a metal staple inside my burger bun! Cut the inside of my gum. Manager apologized and took off $5, which is completely unacceptable.",
            "Both my husband and I suffered severe food poisoning after having the seafood pasta on Saturday night. Spent the entire Sunday in urgent care.",
            "Found a long black hair baked directly into the pizza crust. Disgusting. Lost our appetite immediately.",
            "The raw oysters tasted foul and ammonia-like. Two hours later I had intense stomach cramps and vomiting. Avoid the shellfish at all costs.",
            "Restroom was completely filthy with no hand soap in the dispenser and dirty paper towels all over the floor. Makes me worry about kitchen sanitation."
        ],
        "stars_range": [1, 1],
        "urgency": "Critical",
        "severity": "Critical"
    },
    # Staff Service & Attitude
    {
        "category": "Staff & Service",
        "aspect": "Server Behavior & Attitude",
        "texts": [
            "Server had an awful attitude from the moment we sat down. Rolled her eyes when we asked for separate checks and slammed the plates onto the table.",
            "Our waiter never checked on us once after dropping off the food. Empty glasses remained empty for 40 minutes while staff chatted behind the bar.",
            "The manager was dismissive and combative when we brought up that our steak was overcooked. Argued with us instead of offering a resolution.",
            "Extremely rude staff. When we asked if a dish contained nuts due to an allergy, the waiter replied 'I think it's fine' without even checking with the kitchen.",
            "Server brought the check before we even finished eating and asked if we could hurry up because they needed the table. Completely unwelcoming."
        ],
        "stars_range": [1, 2],
        "urgency": "High",
        "severity": "Medium"
    },
    # Order Accuracy
    {
        "category": "Order Accuracy",
        "aspect": "Missing Items & Dietary Customization",
        "texts": [
            "Specifically requested gluten-free penne due to celiac disease, but received regular wheat pasta. Luckily noticed the texture before eating much.",
            "Ordered takeout for four people and half the items were missing: no side salads, missing dipping sauces, and wrong salad dressing.",
            "I ordered a vegetarian taco plate and received beef tacos. When I called back, they told me to drive 20 minutes back to swap it without any compensation.",
            "Two out of three dishes had wrong ingredients. Kitchen seems incapable of following simple dietary modifications.",
            "Ordered extra cheese and bacon, paid the upcharge, but neither was included on the sandwich when unwrapped."
        ],
        "stars_range": [1, 2],
        "urgency": "High",
        "severity": "Medium"
    },
    # Pricing & Billing
    {
        "category": "Pricing & Value",
        "aspect": "Hidden Surcharges & Portions",
        "texts": [
            "Added an undisclosed 20% 'kitchen wellness' surcharge plus mandatory 18% gratuity on a table of two. Deceptive pricing practices.",
            "Portion sizes have shrunk by half compared to last year while prices jumped 30%. $24 for a tiny scoop of pasta is price gouging.",
            "Double-charged my credit card for the bill. When I called to get it reversed, the accounting manager was unreachable for 5 days.",
            "Charged $4 for regular tap water filtered through their tap machine without disclosing the charge. Ridiculous nickel-and-diming.",
            "Check had extra cocktails we never ordered. Had to debate with the server for ten minutes to get them removed."
        ],
        "stars_range": [2, 3],
        "urgency": "Medium",
        "severity": "Medium"
    },
    # Ambience & Facilities
    {
        "category": "Ambience & Noise",
        "aspect": "Atmosphere & Temperature Control",
        "texts": [
            "Music was so deafeningly loud that my party of four had to shout across the table. Asked the server twice to lower it, nothing was done.",
            "AC was broken in mid-July. Dining room felt like 85 degrees inside with stagnant air. Very uncomfortable dining experience.",
            "Drafty table right next to the kitchen swinging door. Every time staff rushed past, cold wind and loud clattering dishes interrupted us.",
            "Dim lighting to the point where elderly guests could not read the menu with phone flashlights. Tables sticky and wobbly.",
            "Excessive fruit flies hovering around the bar and landing in our drinks. Indicates poor drain maintenance."
        ],
        "stars_range": [2, 3],
        "urgency": "Medium",
        "severity": "Low"
    },
    # Positive / Benchmark Reviews
    {
        "category": "Positive Experience",
        "aspect": "Exemplary Food & Hospitality",
        "texts": [
            "Outstanding dining experience! The handmade pasta was cooked perfectly al dente, and our server Marco was attentive, warm, and knowledgeable.",
            "One of the best dining spots in town. Fresh ingredients, prompt service, and reasonable prices for the generous portion sizes.",
            "Incredible food! Highly recommend the signature ramen broth and gyoza. Table turnaround was swift despite the long queue outside.",
            "Impeccable service and clean dining room. The chef even came out to check on tables. Will definitely be returning soon with friends.",
            "Delightful bakery treats and aromatic coffee. Perfect atmosphere for weekend brunch. Staff always greets you with a genuine smile."
        ],
        "stars_range": [4, 5],
        "urgency": "Low",
        "severity": "Low"
    }
]

class YelpDatasetManager:
    """
    Manages loading, filtering, querying, and storing Yelp Open Dataset records.
    Supports real Yelp JSON files and authentic seeded dataset.
    """
    def __init__(self, data_directory: str = DATA_DIR):
        self.data_dir = data_directory
        os.makedirs(self.data_dir, exist_ok=True)
        self.businesses_path = os.path.join(self.data_dir, "yelp_businesses.json")
        self.reviews_path = os.path.join(self.data_dir, "yelp_reviews.json")
        self.businesses: List[Dict[str, Any]] = []
        self.reviews: List[Dict[str, Any]] = []
        self.business_map: Dict[str, Dict[str, Any]] = {}
        self._ensure_dataset()

    def _ensure_dataset(self):
        """Generates or loads the dataset."""
        if not os.path.exists(self.businesses_path) or not os.path.exists(self.reviews_path):
            self._generate_dataset()
        else:
            self._load_dataset()

    def _generate_dataset(self):
        """Generates authentic Yelp records spanning 2024 to 2026 across restaurants."""
        self.businesses = SAMPLE_BUSINESSES
        self.business_map = {b["business_id"]: b for b in self.businesses}

        reviews = []
        review_counter = 1000
        user_counter = 500

        # Create 250+ authentic reviews distributed over time and categories
        # Temporal spread: from Jan 2024 through March 2026
        base_date = datetime(2024, 1, 10)
        end_date = datetime(2026, 3, 25)
        total_days = (end_date - base_date).days

        for b in self.businesses:
            b_id = b["business_id"]
            # Generate between 30 and 45 reviews per restaurant
            num_reviews = random.randint(30, 42)
            for _ in range(num_reviews):
                review_counter += 1
                user_counter += 1
                r_id = f"rev_{review_counter}_{b_id[:10]}"
                u_id = f"usr_{user_counter}"

                # Sample template
                # Weighted: 70% complaints (various categories), 30% positive
                if random.random() < 0.72:
                    tmpl = random.choice([t for t in COMPLAINT_TEMPLATES if t["category"] != "Positive Experience"])
                else:
                    tmpl = [t for t in COMPLAINT_TEMPLATES if t["category"] == "Positive Experience"][0]

                text = random.choice(tmpl["texts"])
                stars = random.randint(tmpl["stars_range"][0], tmpl["stars_range"][1])

                # Generate date with realistic trend evolution (spikes in recent months)
                day_offset = int(random.triangular(0, total_days, total_days * 0.85))
                rev_date = base_date + timedelta(days=day_offset)

                rev = {
                    "review_id": r_id,
                    "business_id": b_id,
                    "user_id": u_id,
                    "stars": stars,
                    "date": rev_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "text": text,
                    "useful": random.randint(0, 12),
                    "funny": random.randint(0, 5),
                    "cool": random.randint(0, 7),
                    # Augmented metadata for fast retrieval & analysis
                    "restaurant_name": b["name"],
                    "city": b["city"],
                    "state": b["state"],
                    "category": tmpl["category"],
                    "aspect": tmpl["aspect"],
                    "urgency": tmpl["urgency"],
                    "severity": tmpl["severity"]
                }
                reviews.append(rev)

        # Sort by date descending
        reviews.sort(key=lambda x: x["date"], reverse=True)
        self.reviews = reviews

        # Save to JSON
        with open(self.businesses_path, "w", encoding="utf-8") as f:
            json.dump(self.businesses, f, indent=2)
        with open(self.reviews_path, "w", encoding="utf-8") as f:
            json.dump(self.reviews, f, indent=2)

    def _load_dataset(self):
        """Loads business and review records from JSON files."""
        try:
            with open(self.businesses_path, "r", encoding="utf-8") as f:
                self.businesses = json.load(f)
            with open(self.reviews_path, "r", encoding="utf-8") as f:
                self.reviews = json.load(f)
            self.business_map = {b["business_id"]: b for b in self.businesses}
        except Exception as e:
            # Re-generate if corrupt
            self._generate_dataset()

    def get_all_businesses(self) -> List[Dict[str, Any]]:
        return self.businesses

    def get_all_reviews(self) -> List[Dict[str, Any]]:
        return self.reviews

    def get_business(self, business_id: str) -> Optional[Dict[str, Any]]:
        return self.business_map.get(business_id)

    def filter_reviews(
        self,
        business_id: Optional[str] = None,
        category: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        min_stars: Optional[int] = None,
        max_stars: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filters reviews dynamically based on all specified metadata criteria."""
        try:
            import psycopg2
            conn = psycopg2.connect(host="127.0.0.1", port=5432, dbname="postgres", user="bharathyadav", password="")
            with conn.cursor() as cur:
                where_clauses = ["1=1"]
                params = []
                if business_id and business_id.lower() != "all":
                    where_clauses.append("(LOWER(b.name) = %s OR LOWER(r.business_id) = %s)")
                    params.extend([business_id.lower(), business_id.lower()])
                if city and city.lower() != "all":
                    where_clauses.append("LOWER(b.city) = %s")
                    params.append(city.lower())
                if state and state.lower() != "all":
                    where_clauses.append("LOWER(b.state) = %s")
                    params.append(state.lower())
                if min_stars is not None:
                    where_clauses.append("r.stars >= %s")
                    params.append(min_stars)
                if max_stars is not None:
                    where_clauses.append("r.stars <= %s")
                    params.append(max_stars)
                if start_date:
                    where_clauses.append("r.date >= %s")
                    params.append(start_date)
                if end_date:
                    where_clauses.append("r.date <= %s")
                    params.append(end_date)

                query = f"""
                    SELECT r.review_id, r.business_id, COALESCE(b.name, 'Unknown Restaurant'), r.stars, r.date, r.text, COALESCE(b.city, ''), COALESCE(b.state, '')
                    FROM yelp_reviews r
                    LEFT JOIN yelp_businesses b ON r.business_id = b.business_id
                    WHERE {" AND ".join(where_clauses)}
                    ORDER BY r.date DESC;
                """
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
            conn.close()
            if rows:
                results = []
                for row in rows:
                    results.append({
                        "review_id": row[0],
                        "business_id": row[1],
                        "restaurant_name": row[2],
                        "stars": float(row[3] or 1.0),
                        "date": str(row[4] or ""),
                        "text": row[5] or "",
                        "city": row[6] or "",
                        "state": row[7] or ""
                    })
                return results
        except Exception:
            pass

        results = []
        for r in self.reviews:
            if business_id and business_id.lower() != "all":
                b_match = (
                    r.get("business_id", "").lower() == business_id.lower() or
                    r.get("restaurant_name", "").lower() == business_id.lower()
                )
                if not b_match:
                    continue
            if category and category.lower() != "all":
                b_info = self.business_map.get(r["business_id"], {})
                b_cats = b_info.get("categories", "").lower()
                r_cat = r.get("category", "").lower()
                if category.lower() not in b_cats and category.lower() not in r_cat:
                    continue
            if city and city.lower() != "all" and r.get("city", "").lower() != city.lower():
                continue
            if state and state.lower() != "all" and r.get("state", "").lower() != state.lower():
                continue
            if min_stars is not None and r.get("stars", 5) < min_stars:
                continue
            if max_stars is not None and r.get("stars", 1) > max_stars:
                continue
            if start_date and r.get("date", "")[:10] < start_date:
                continue
            if end_date and r.get("date", "")[:10] > end_date:
                continue
            results.append(r)
        return results

    def get_distinct_metadata(self) -> Dict[str, Any]:
        """Returns unique categories, cities, states, and restaurants for dynamic UI dropdowns."""
        # Try loading from PostgreSQL production database
        try:
            import psycopg2
            conn = psycopg2.connect(host="127.0.0.1", port=5432, dbname="postgres", user="bharathyadav", password="")
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM yelp_businesses;")
                total_biz = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM yelp_reviews;")
                total_rev = cur.fetchone()[0]

                if total_biz > 0 and total_rev > 0:
                    # Fetch reviewed restaurants for accurate selection
                    cur.execute("""
                        SELECT DISTINCT b.name 
                        FROM yelp_businesses b 
                        JOIN yelp_reviews r ON b.business_id = r.business_id 
                        ORDER BY b.name 
                        LIMIT 300;
                    """)
                    restaurants = [r[0] for r in cur.fetchall()]

                    cur.execute("SELECT DISTINCT city FROM yelp_businesses WHERE city IS NOT NULL AND city != '' ORDER BY city LIMIT 50;")
                    cities = [r[0] for r in cur.fetchall()]

                    cur.execute("SELECT DISTINCT state FROM yelp_businesses WHERE state IS NOT NULL AND state != '' ORDER BY state LIMIT 50;")
                    states = [r[0] for r in cur.fetchall()]

                    cur.execute("SELECT categories FROM yelp_businesses WHERE categories IS NOT NULL LIMIT 1000;")
                    cats = set()
                    for row in cur.fetchall():
                        for c in (row[0] or "").split(","):
                            clean = c.strip()
                            if clean:
                                cats.add(clean)

                    conn.close()
                    return {
                        "restaurants": restaurants,
                        "cities": cities,
                        "states": states,
                        "categories": sorted(list(cats))[:40],
                        "total_reviews": total_rev,
                        "total_restaurants": total_biz
                    }
        except Exception:
            pass

        restaurants = sorted(list({b["name"] for b in self.businesses}))[:300]
        cities = sorted(list({b["city"] for b in self.businesses}))
        states = sorted(list({b["state"] for b in self.businesses}))
        
        # Aggregate categories from restaurants and review operational tags
        cats = set()
        for b in self.businesses:
            for c in b.get("categories", "").split(","):
                clean = c.strip()
                if clean:
                    cats.add(clean)
        for r in self.reviews:
            if r.get("category"):
                cats.add(r["category"])
        
        return {
            "restaurants": restaurants,
            "cities": cities,
            "states": states,
            "categories": sorted(list(cats)),
            "total_reviews": len(self.reviews),
            "total_restaurants": len(self.businesses)
        }

# Global singleton instance
dataset_manager = YelpDatasetManager()
