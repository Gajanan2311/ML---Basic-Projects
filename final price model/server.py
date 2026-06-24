from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import math

app = Flask(__name__)
CORS(app)

# ── Load artifacts ────────────────────────────────────────────────────────────
model    = joblib.load('final_model.pkl')
scaler   = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl')
thresh   = joblib.load('thresholds.pkl')
lookup   = pd.read_csv('demand_supply_lookup.csv')

print("✅ All artifacts loaded")
print(f"   Demand -> Q25:{thresh['low_demand_value']:.1f} Q75:{thresh['high_demand_value']:.1f}")
print(f"   Supply -> Q25:{thresh['low_supply_value']:.1f} Q75:{thresh['high_supply_value']:.1f}")

# ── Colombo places & travel time table ───────────────────────────────────────
PLACES = [
    "Colombo Fort Railway Station",
    "Bandaranaike Memorial Hall (Fort)",
    "Pettah Market",
    "Gangaramaya Temple",
    "Galle Face Green",
    "Independence Square",
    "National Museum of Colombo",
    "Viharamahadevi Park",
    "Lotus Tower",
    "Jami Ul-Alfar Mosque",
    "Dutch Hospital Shopping Precinct",
    "Kollupitiya (Liberty Plaza)",
    "Bambalapitiya",
    "Wellawatte",
    "Dehiwala Zoo",
    "Mount Lavinia Beach",
    "Nugegoda Town",
    "Rajagiriya",
    "Borella (General Hospital)",
    "Maradana",
]

# km offsets from Colombo Fort (south = negative, east = positive)
_COORDS = {
    "Colombo Fort Railway Station":         (0.0,  0.0),
    "Bandaranaike Memorial Hall (Fort)":    (0.2,  0.1),
    "Pettah Market":                        (0.4,  0.3),
    "Gangaramaya Temple":                   (-1.2, 0.6),
    "Galle Face Green":                     (-1.6, 0.0),
    "Independence Square":                  (-3.2, 1.0),
    "National Museum of Colombo":           (-3.0, 0.8),
    "Viharamahadevi Park":                  (-2.8, 0.7),
    "Lotus Tower":                          (-1.9, 0.6),
    "Jami Ul-Alfar Mosque":                 (0.3,  0.4),
    "Dutch Hospital Shopping Precinct":     (0.1,  0.0),
    "Kollupitiya (Liberty Plaza)":          (-2.6, 0.3),
    "Bambalapitiya":                        (-4.2, 0.4),
    "Wellawatte":                           (-5.8, 0.3),
    "Dehiwala Zoo":                         (-7.2, 0.5),
    "Mount Lavinia Beach":                  (-9.5, 0.3),
    "Nugegoda Town":                        (-5.2, 2.8),
    "Rajagiriya":                           (-2.6, 3.8),
    "Borella (General Hospital)":           (-2.1, 1.6),
    "Maradana":                             (-0.6, 1.3),
}

# Location category mapping (used as feature for the model)
_LOCATION_CAT = {
    "Colombo Fort Railway Station":         "Urban",
    "Bandaranaike Memorial Hall (Fort)":    "Urban",
    "Pettah Market":                        "Urban",
    "Gangaramaya Temple":                   "Urban",
    "Galle Face Green":                     "Urban",
    "Independence Square":                  "Urban",
    "National Museum of Colombo":           "Urban",
    "Viharamahadevi Park":                  "Urban",
    "Lotus Tower":                          "Urban",
    "Jami Ul-Alfar Mosque":                 "Urban",
    "Dutch Hospital Shopping Precinct":     "Urban",
    "Kollupitiya (Liberty Plaza)":          "Urban",
    "Bambalapitiya":                        "Suburban",
    "Wellawatte":                           "Suburban",
    "Dehiwala Zoo":                         "Suburban",
    "Mount Lavinia Beach":                  "Suburban",
    "Nugegoda Town":                        "Suburban",
    "Rajagiriya":                           "Suburban",
    "Borella (General Hospital)":           "Urban",
    "Maradana":                             "Urban",
}

def _travel_time(a, b):
    ax, ay = _COORDS[a]
    bx, by = _COORDS[b]
    dist   = math.sqrt((ax-bx)**2 + (ay-by)**2)
    base   = (dist / 22) * 60          # 22 kmph avg Colombo traffic
    penalty = 5 if dist < 1.5 else 8 if dist < 4 else 12
    return max(5, round(base + penalty))

# Pre-build full route table
ROUTE_TABLE = {}
for a in PLACES:
    for b in PLACES:
        if a != b:
            ROUTE_TABLE[f"{a}|{b}"] = _travel_time(a, b)

print(f"✅ Route table built: {len(ROUTE_TABLE)} routes across {len(PLACES)} Colombo locations")


# ── Pricing helpers ───────────────────────────────────────────────────────────
def demand_mult(riders, t):
    hdv, ldv = t['high_demand_value'], t['low_demand_value']
    raw = riders/hdv if riders > hdv else riders/ldv if riders < ldv else 1.0
    return float(np.clip(raw, t['DEMAND_FLOOR'], t['DEMAND_CAP']))

def supply_mult(drivers, t):
    hsv, lsv = t['high_supply_value'], t['low_supply_value']
    raw = lsv/drivers if drivers < lsv else hsv/drivers if drivers > hsv else 1.0
    return float(np.clip(raw, t['SUPPLY_FLOOR'], t['SUPPLY_CAP']))


# ── Route 1: list of places ───────────────────────────────────────────────────
@app.route('/api/places', methods=['GET'])
def get_places():
    return jsonify({'places': PLACES})


# ── Route 2: travel time + demand/supply defaults for a route ─────────────────
@app.route('/api/route', methods=['GET'])
def get_route():
    origin = request.args.get('origin', '')
    dest   = request.args.get('dest', '')
    time   = request.args.get('time', 'Morning')

    if origin == dest:
        return jsonify({'error': 'Origin and destination cannot be the same'}), 400
    if f"{origin}|{dest}" not in ROUTE_TABLE:
        return jsonify({'error': 'Unknown location'}), 400

    duration = ROUTE_TABLE[f"{origin}|{dest}"]
    location = _LOCATION_CAT.get(origin, 'Urban')

    # Safe filtering: convert to string to prevent mismatch issues
    row = lookup[
        (lookup['Time_of_Booking'].astype(str).str.strip() == time) &
        (lookup['Location_Category'].astype(str).str.strip() == location)
    ]
    
    # Safe fallback if row is empty
    if not row.empty:
        avg_riders  = int(round(row['avg_riders'].values[0]))
        avg_drivers = int(round(row['avg_drivers'].values[0]))
    else:
        # Default fallbacks if data isn't in your CSV
        avg_riders  = 60
        avg_drivers = 30

    return jsonify({
        'origin'      : origin,
        'destination' : dest,
        'duration_min': duration,
        'location_cat': location,
        'avg_riders'  : avg_riders,
        'avg_drivers' : avg_drivers,
        'time'        : time,
    })



# ── Route 3: predict fare ─────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        body    = request.get_json()
        origin  = body['origin']
        dest    = body['destination']
        time    = body['time']
        vehicle = body['vehicle']
        loyalty = body.get('loyalty',    'Regular')
        ratings = float(body.get('ratings',    4.0))
        past    = int(body.get('past_rides',   20))
        riders  = float(body['riders'])
        drivers = float(body['drivers'])

        # Duration from route table
        key = f"{origin}|{dest}"
        if key not in ROUTE_TABLE:
            return jsonify({'error': f'Unknown route: {key}'}), 400
        duration = ROUTE_TABLE[key]

        # Encode categoricals
        time_enc    = int(encoders['Time_of_Booking'].transform([time])[0])
        vehicle_enc = int(encoders['Vehicle_Type'].transform([vehicle])[0])
        loyalty_enc = int(encoders['Customer_Loyalty_Status'].transform([loyalty])[0])

        # Feature row — exact same order as features_goal1 in notebook
        features = pd.DataFrame([[
            duration, riders, drivers,
            time_enc, vehicle_enc, loyalty_enc,
            past, ratings
        ]], columns=[
            'Expected_Ride_Duration', 'Number_of_Riders', 'Number_of_Drivers',
            'Time_of_Booking_enc', 'Vehicle_Type_enc', 'Customer_Loyalty_Status_enc',
            'Number_of_Past_Rides', 'Average_Ratings'
        ])

        scaled     = scaler.transform(features)
        base_fare  = max(float(model.predict(scaled)[0]), 0)

        d_mult     = demand_mult(riders, thresh)
        s_mult     = supply_mult(drivers, thresh)
        combined   = d_mult * s_mult
        final_fare = base_fare * combined
        pct_change = (combined - 1.0) * 100

        return jsonify({
            'origin'        : origin,
            'destination'   : dest,
            'duration_min'  : duration,
            'base_fare'     : round(base_fare,  2),
            'final_fare'    : round(final_fare, 2),
            'demand_mult'   : round(d_mult,     3),
            'supply_mult'   : round(s_mult,     3),
            'combined_mult' : round(combined,   3),
            'pct_change'    : round(pct_change, 1),
            'thresholds'    : {
                'demand_q25': round(thresh['low_demand_value'],  1),
                'demand_q75': round(thresh['high_demand_value'], 1),
                'supply_q25': round(thresh['low_supply_value'],  1),
                'supply_q75': round(thresh['high_supply_value'], 1),
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ── Route 4: health check ─────────────────────────────────────────────────────
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'places': len(PLACES), 'routes': len(ROUTE_TABLE)})


from flask import send_from_directory

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)