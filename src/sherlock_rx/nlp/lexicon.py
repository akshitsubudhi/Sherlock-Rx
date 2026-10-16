"""Closed-class domain lexicons: metrics, breakdown dimensions, baselines, therapy areas,
region and state aliases. Values are canonical ids the dialogue layer understands."""

from __future__ import annotations

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "net_sales": ("sales", "net sales", "secondary sales", "secondaries", "revenue", "turnover",
                  "business", "value", "performance", "numbers", "offtake"),
    "units": ("units", "volume", "volumes", "strips", "unit sales", "quantity"),
    "target": ("target", "targets", "plan", "budget", "aop"),
    "achievement": ("achievement", "target achievement", "achievement percentage", "ach"),
    "growth": ("growth", "yoy growth", "growth rate"),
    "market_share": ("market share", "share", "volume share"),
    "market_units": ("market size", "market", "category size", "molecule market"),
    "prescriptions": ("prescriptions", "prescription", "rx", "rxs", "scripts", "rx count"),
    "calls": ("calls", "doctor calls", "call average", "coverage", "detailing", "visits"),
    "vacancy_rate": ("vacancy", "vacancies", "vacancy rate", "field force", "mr strength",
                     "manpower", "reps", "mrs", "headcount"),
    "stockout_days": ("stockout", "stockouts", "stock out", "stock outs", "stock-out",
                      "stock-outs", "out of stock", "availability", "supply", "shortage",
                      "stock"),
    "active_retailers": ("retailers", "distribution", "retail reach", "reach", "chemists",
                         "outlets", "stockists", "retail coverage"),
}

DIMENSION_ALIASES: dict[str, tuple[str, ...]] = {
    "state": ("by state", "state wise", "statewise", "state-wise", "across states",
              "each state", "per state", "states"),
    "territory": ("by territory", "territory wise", "territory-wise", "by hq", "hq wise",
                  "hq-wise", "headquarter wise", "territories", "by town", "hqs"),
    "region": ("by region", "region wise", "region-wise", "zone wise", "zone-wise", "by zone",
               "across regions", "all regions", "regions", "zones", "each region"),
    "sku": ("by sku", "sku wise", "sku-wise", "by pack", "skus", "by strength", "pack wise"),
    "month": ("by month", "monthly", "month wise", "month-wise", "month on month", "mom",
              "every month"),
    "brand": ("by brand", "brand wise", "brand-wise", "all brands", "brands", "each brand",
              "portfolio"),
}

BASELINE_ALIASES: dict[str, tuple[str, ...]] = {
    "target": ("vs target", "versus target", "against target", "vs plan", "against plan",
               "compared to target", "vs budget", "to target"),
    "last_year": ("vs last year", "versus last year", "yoy", "year on year", "vs ly", "vs py",
                  "compared to last year", "over last year", "than last year", "y-o-y",
                  "vs same period last year"),
    "previous_period": ("vs last quarter", "vs previous quarter", "qoq", "quarter on quarter",
                        "vs last month", "vs previous month", "sequentially"),
}

THERAPY_ALIASES: dict[str, tuple[str, ...]] = {
    "Cardiology": ("cardiology", "cardiac", "cardio", "heart", "hypertension"),
    "Diabetology": ("diabetology", "diabetes", "anti-diabetic", "antidiabetic", "diabetic"),
    "Respiratory": ("respiratory", "asthma", "allergy", "respi"),
    "Gastroenterology": ("gastroenterology", "gastro", "gi", "acidity", "antacid"),
    "Pain": ("pain", "analgesic", "analgesics", "pain management"),
    "Neurology": ("neurology", "neuro", "neuropathy", "vitamins"),
    "Anti-infectives": ("anti-infectives", "anti-infective", "antibiotic", "antibiotics",
                        "anti infectives"),
    "Dermatology": ("dermatology", "derma", "skin", "antifungal", "anti-fungal"),
}

REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "North": ("north", "north zone", "northern", "north india", "northern region"),
    "South": ("south", "south zone", "southern", "south india", "southern region"),
    "East": ("east", "east zone", "eastern", "east india", "eastern region"),
    "West": ("west", "west zone", "western", "west india", "western region"),
    "Central": ("central", "central zone", "central india", "central region"),
}

# State codes are only matched when written in upper case ("UP", not "up").
STATE_ALIASES: dict[str, tuple[str, ...]] = {
    "Odisha": ("orissa",),
    "West Bengal": ("bengal", "wb"),
    "Uttar Pradesh": ("u.p.", "u p"),
    "Tamil Nadu": ("tamilnadu", "t.n."),
    "Andhra Pradesh": ("andhra",),
    "Madhya Pradesh": ("m.p.",),
    "Delhi": ("new delhi", "ncr", "delhi ncr"),
    "Chhattisgarh": ("chattisgarh", "chhatisgarh"),
}
