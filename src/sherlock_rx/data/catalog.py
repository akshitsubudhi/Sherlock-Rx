"""Master data for the synthetic Indian pharma market.

Brands and competitors are fictional; molecules, therapy areas, Indian states and the
April-March fiscal year are real so the data behaves like a domestic formulations business.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sku:
    sku_id: str
    name: str
    base_ptr: float  # price to retailer, INR per strip/unit
    mix: float  # share of brand units


@dataclass(frozen=True)
class Brand:
    brand_id: str
    name: str
    molecule: str
    therapy: str
    national_value_cr: float  # approximate annual secondary sales, INR crore
    share: float  # typical volume share of its molecule market
    growth: float  # underlying annual market growth for the molecule
    skus: tuple[Sku, ...]
    competitors: tuple[str, ...]
    aliases: tuple[str, ...] = field(default=())


BRANDS: tuple[Brand, ...] = (
    Brand("B01", "Cardivex", "Telmisartan", "Cardiology", 180, 0.22, 0.09,
          (Sku("B01-40", "Cardivex 40", 92.0, 0.62), Sku("B01-80", "Cardivex 80", 148.0, 0.38)),
          ("Telmikind", "Hypotel", "Sartanex"), ("cardivx", "cardi vex")),
    Brand("B02", "Glucoryl", "Metformin + Glimepiride", "Diabetology", 240, 0.18, 0.11,
          (Sku("B02-M1", "Glucoryl M1", 78.0, 0.55), Sku("B02-M2", "Glucoryl M2", 104.0, 0.45)),
          ("Glimiday", "Diabefort", "Metglim"), ("glucoril", "gluco ryl")),
    Brand("B03", "Respiron", "Montelukast + Levocetirizine", "Respiratory", 130, 0.20, 0.08,
          (Sku("B03-T", "Respiron Tab", 115.0, 0.70), Sku("B03-S", "Respiron Syrup", 86.0, 0.30)),
          ("Montair-L", "Breathex", "Allerkast"), ("respiran",)),
    Brand("B04", "Gastrazole", "Pantoprazole", "Gastroenterology", 160, 0.16, 0.07,
          (Sku("B04-40", "Gastrazole 40", 64.0, 0.75), Sku("B04-DSR", "Gastrazole DSR", 118.0, 0.25)),
          ("Pantocid-X", "Pepsofix", "Acidguard"), ("gastrozole", "gastra")),
    Brand("B05", "Painex", "Aceclofenac + Paracetamol", "Pain", 110, 0.14, 0.06,
          (Sku("B05-P", "Painex P", 58.0, 0.65), Sku("B05-SP", "Painex SP", 82.0, 0.35)),
          ("Zerodol-Z", "Aceflam", "Dolorex"), ("pain ex", "painx")),
    Brand("B06", "Neurovita", "Methylcobalamin", "Neurology", 90, 0.19, 0.10,
          (Sku("B06-1500", "Neurovita 1500", 132.0, 0.60), Sku("B06-OD", "Neurovita OD", 165.0, 0.40)),
          ("Nervicob", "Cobaforte", "Myelin-Plus"), ("neuro vita",)),
    Brand("B07", "Cefobact", "Cefixime", "Anti-infectives", 140, 0.17, 0.07,
          (Sku("B07-200", "Cefobact 200", 98.0, 0.68), Sku("B07-DS", "Cefobact Dry Syrup", 72.0, 0.32)),
          ("Taxim-OZ", "Cefinext", "Zifibact"), ("cefobac", "cefo bact")),
    Brand("B08", "Dermaclear", "Luliconazole", "Dermatology", 70, 0.21, 0.12,
          (Sku("B08-C", "Dermaclear Cream", 145.0, 0.80), Sku("B08-L", "Dermaclear Lotion", 168.0, 0.20)),
          ("Lulifin", "Fungiclear", "Skinazole"), ("derma clear",)),
)

# Zone -> states. Zones are the "regions" sales leadership reviews.
REGIONS: dict[str, tuple[str, ...]] = {
    "North": ("Delhi", "Punjab", "Haryana", "Uttar Pradesh", "Rajasthan"),
    "South": ("Tamil Nadu", "Karnataka", "Kerala", "Andhra Pradesh", "Telangana"),
    "East": ("West Bengal", "Odisha", "Bihar", "Assam", "Jharkhand"),
    "West": ("Maharashtra", "Gujarat", "Goa"),
    "Central": ("Madhya Pradesh", "Chhattisgarh"),
}

STATE_CODES: dict[str, str] = {
    "Delhi": "DL", "Punjab": "PB", "Haryana": "HR", "Uttar Pradesh": "UP", "Rajasthan": "RJ",
    "Tamil Nadu": "TN", "Karnataka": "KA", "Kerala": "KL", "Andhra Pradesh": "AP",
    "Telangana": "TG", "West Bengal": "WB", "Odisha": "OD", "Bihar": "BR", "Assam": "AS",
    "Jharkhand": "JH", "Maharashtra": "MH", "Gujarat": "GJ", "Goa": "GA",
    "Madhya Pradesh": "MP", "Chhattisgarh": "CG",
}

# Relative market size of a state (drives territory count and territory size).
STATE_WEIGHT: dict[str, float] = {
    "Delhi": 1.0, "Punjab": 0.8, "Haryana": 0.7, "Uttar Pradesh": 1.6, "Rajasthan": 1.0,
    "Tamil Nadu": 1.3, "Karnataka": 1.2, "Kerala": 1.0, "Andhra Pradesh": 1.0, "Telangana": 0.9,
    "West Bengal": 1.2, "Odisha": 0.6, "Bihar": 0.9, "Assam": 0.5, "Jharkhand": 0.5,
    "Maharashtra": 1.8, "Gujarat": 1.2, "Goa": 0.3, "Madhya Pradesh": 0.9, "Chhattisgarh": 0.5,
}

# Headquarter towns used to name the first territories of each state.
HQ_TOWNS: dict[str, tuple[str, ...]] = {
    "Delhi": ("Central Delhi", "South Delhi", "East Delhi", "West Delhi", "North Delhi"),
    "Punjab": ("Ludhiana", "Amritsar", "Jalandhar", "Patiala"),
    "Haryana": ("Gurugram", "Faridabad", "Ambala", "Hisar"),
    "Uttar Pradesh": ("Lucknow", "Kanpur", "Varanasi", "Agra", "Meerut", "Prayagraj", "Gorakhpur"),
    "Rajasthan": ("Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner"),
    "Tamil Nadu": ("Chennai", "Coimbatore", "Madurai", "Trichy", "Salem"),
    "Karnataka": ("Bengaluru", "Mysuru", "Hubballi", "Mangaluru", "Belagavi"),
    "Kerala": ("Kochi", "Thiruvananthapuram", "Kozhikode", "Thrissur"),
    "Andhra Pradesh": ("Vijayawada", "Visakhapatnam", "Guntur", "Tirupati"),
    "Telangana": ("Hyderabad", "Warangal", "Karimnagar", "Nizamabad"),
    "West Bengal": ("Kolkata", "Howrah", "Siliguri", "Durgapur", "Asansol"),
    "Odisha": ("Bhubaneswar", "Cuttack", "Rourkela"),
    "Bihar": ("Patna", "Gaya", "Muzaffarpur", "Bhagalpur"),
    "Assam": ("Guwahati", "Dibrugarh", "Silchar"),
    "Jharkhand": ("Ranchi", "Jamshedpur", "Dhanbad"),
    "Maharashtra": ("Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad", "Thane", "Kolhapur"),
    "Gujarat": ("Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"),
    "Goa": ("Panaji", "Margao"),
    "Madhya Pradesh": ("Indore", "Bhopal", "Jabalpur", "Gwalior"),
    "Chhattisgarh": ("Raipur", "Bilaspur", "Durg"),
}

# Monthly seasonality multipliers per therapy, indexed by calendar month 1..12.
SEASONALITY: dict[str, tuple[float, ...]] = {
    "Respiratory":      (1.18, 1.08, 0.95, 0.85, 0.82, 0.92, 1.05, 1.08, 1.05, 1.00, 1.08, 1.20),
    "Anti-infectives":  (0.95, 0.90, 0.88, 0.90, 0.95, 1.08, 1.20, 1.22, 1.15, 1.00, 0.92, 0.90),
    "Dermatology":      (0.85, 0.88, 0.98, 1.10, 1.15, 1.12, 1.10, 1.05, 1.00, 0.95, 0.90, 0.88),
}
FLAT_SEASON: tuple[float, ...] = (1.0,) * 12


def brand_by_name(name: str) -> Brand:
    for b in BRANDS:
        if b.name.lower() == name.lower():
            return b
    raise KeyError(name)


def region_of_state(state: str) -> str:
    for region, states in REGIONS.items():
        if state in states:
            return region
    raise KeyError(state)
