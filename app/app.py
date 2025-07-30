import os
import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# -------------------- Load Model and Data --------------------

# Paths
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'car_data.csv')

# Load model
model = pickle.load(open(model_path, 'rb'))

# Load dataset
df = pd.read_csv(data_path)

# Extract brand and model
df['brand'] = df['name'].apply(lambda x: x.split()[0])
df['model'] = df['name'].apply(lambda x: ' '.join(x.split()[1:2]))

# Clean numeric features
def extract_number(x):
    try: return float(str(x).split()[0])
    except: return None

def clean_torque(x):
    try: return float(str(x).split('Nm')[0].strip().split()[0])
    except: return None

df['mileage'] = df['mileage'].apply(extract_number)
df['engine'] = df['engine'].apply(extract_number)
df['max_power'] = df['max_power'].apply(extract_number)
df['torque'] = df['torque'].apply(clean_torque)
df['car_age'] = 2025 - df['year']
df.dropna(inplace=True)

# -------------------- Routes --------------------

@app.route('/')
def home():
    brands = sorted(df['brand'].unique())
    return render_template('index.html', brands=brands)

@app.route('/get_models', methods=['POST'])
def get_models():
    brand = request.form.get('brand')
    models = sorted(df[df['brand'] == brand]['model'].unique()) if brand else []
    return jsonify(models)

@app.route('/get_options', methods=['POST'])
def get_options():
    brand = request.form.get('brand')
    model = request.form.get('model')

    if not brand or not model:
        return jsonify({})

    filtered = df[(df['brand'] == brand) & (df['model'] == model)]

    details = filtered.to_dict(orient='records')

    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    options = {
        'km_driven': sorted([convert(x) for x in filtered['km_driven'].unique()]),
        'fuel': sorted(filtered['fuel'].unique()),
        'seller_type': sorted(filtered['seller_type'].unique()),
        'transmission': sorted(filtered['transmission'].unique()),
        'owner': sorted(filtered['owner'].unique()),
        'mileage': sorted([convert(x) for x in filtered['mileage'].unique()]),
        'engine': sorted([convert(x) for x in filtered['engine'].unique()]),
        'max_power': sorted([convert(x) for x in filtered['max_power'].unique()]),
        'torque': sorted([convert(x) for x in filtered['torque'].unique()]),
        'seats': sorted([convert(x) for x in filtered['seats'].unique()]),
        'car_age': sorted([convert(x) for x in filtered['car_age'].unique()]),
        'details': [{k: convert(v) for k, v in rec.items()} for rec in details]
    }
    return jsonify(options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form

        km_driven = float(form.get('km_driven', 0))
        fuel = form.get('fuel', 'Petrol')
        seller_type = form.get('seller_type', 'Dealer')
        transmission = form.get('transmission', 'Manual')
        owner = form.get('owner', 'First Owner')
        mileage = float(form.get('mileage', 0))
        engine = float(form.get('engine', 0))
        max_power = float(form.get('max_power', 0))
        torque = float(form.get('torque', 0))
        seats = int(form.get('seats', 5))
        car_age = int(form.get('car_age', 5))

        enc = {
            'fuel': {'Petrol': 2, 'Diesel': 1, 'CNG': 0, 'LPG': 3, 'Electric': 4},
            'seller_type': {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2},
            'transmission': {'Manual': 1, 'Automatic': 0},
            'owner': {
                'First Owner': 0,
                'Second Owner': 1,
                'Third Owner': 2,
                'Fourth & Above Owner': 3,
                'Test Drive Car': 4
            }
        }

        features = [[
            km_driven,
            enc['fuel'].get(fuel, 2),
            enc['seller_type'].get(seller_type, 0),
            enc['transmission'].get(transmission, 1),
            enc['owner'].get(owner, 0),
            mileage,
            engine,
            max_power,
            torque,
            seats,
            car_age
        ]]

        predicted_price = model.predict(features)[0]
        price = f"₹ {int(round(predicted_price, 2)):,}"

        brands = sorted(df['brand'].unique())
        return render_template('index.html',
                               brands=brands,
                               prediction_text=price,
                               car_info={
                                   'km_driven': km_driven,
                                   'fuel': fuel,
                                   'seller_type': seller_type,
                                   'transmission': transmission,
                                   'owner': owner,
                                   'mileage': mileage,
                                   'engine': engine,
                                   'max_power': max_power,
                                   'torque': torque,
                                   'seats': seats,
                                   'car_age': car_age
                               })

    except Exception as e:
        return render_template('index.html',
                               brands=sorted(df['brand'].unique()),
                               prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
