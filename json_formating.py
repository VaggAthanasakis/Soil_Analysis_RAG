import json

# Original JSON dictionary as a Python dict
data = {
    "Soil humidity": "6.09 % null",
    "SP": "59.62 % null",
    "Phenomenon Special Weight": "1.27 g/cm3 null",
    "PWP": "14.90 % null",
    "FC": "32.79 % null",
    "Water Available": "17.89 % null",
    "Sand": "8.34 % null",
    "Silt": "27.37 % null",
    "Clay": "64.29 % bouyoucos clay Clay heavy ground",
    "Evaluation": "Clayey",
    "pH": "7.85 null CORRESPORTY PASTS",
    "Elect. Conductance": "281.00 µs/cm KORESMOU WATER",
    "Organic Matter": "2.54 % liquid oxidation",
    "Caco3": "12.50 % KB Volumetrically",
    "Active CaCo3": "2.68 % KB C2O4 (NH4) 2",
    "Total Nitrogen (N)": "0.15 % Kjeldahl",
    "Nitrate Nitrogen (NO3-N)": "4.72 mg/kg 1N kcl",
    "Phosphorus (P)": "4.35 mg/kg olsen",
    "potassium (K)": "207.94 mg/kg NH4AOC, pH 7",
    "sodium (Na)": "25.67 mg/kg NH4AOC, PH 7",
    "calcium (Ca)": "7583.55 mg/kg NH4AOC, pH 7",
    "magnesium (Mg)": "780.85 mg/kg NH4AOC, pH 7",
    "iron (Fe)": "6.92 mg/kg DTPA",
    "zinc (Zn)": "0.29 mg/kg DTPA",
    "Manganese (Mn)": "1.25 mg/kg DTPA",
    "Copper (Cu)": "1.88 mg/kg DTPA",
    "Boron (B)": "0.93 mg/kg azomethin",
    "C.E.C": "null null computing",
    "Relationship C/N": "null null null",
    "E.S.P": "null null null",
    "S.A.R": "null"
}

def parse_value(field_value):
    # Check if the field_value is None or string 'null'
    if field_value is None or field_value.strip().lower() == "null":
        return {"value": None, "unit": None, "method": None}
    
    # Split the string into tokens based on whitespace
    tokens = field_value.split()
    
    # If no tokens, return all None
    if not tokens:
        return {"value": None, "unit": None, "method": None}
    
    # First token is the value.
    value = tokens[0]
    
    # Second token if exists is the unit, otherwise None.
    unit = tokens[1] if len(tokens) > 1 else None
    
    # Remaining tokens are joined as method if they exist.
    method = " ".join(tokens[2:]) if len(tokens) > 2 else None
    
    return {"value": value, "unit": unit, "method": method}

# Build the new JSON structure
new_data = {key: parse_value(value) for key, value in data.items()}

# Pretty print the new JSON
print(json.dumps(new_data, indent=4))
