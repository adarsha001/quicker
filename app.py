from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd
from num2words import num2words

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and car dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Read the cleaned CSV file into a DataFrame
car = pd.read_csv('Clean_Car.csv')  # This will fetch data from the CSV file

@app.route('/api')
def hello_world():
    return jsonify({"message": "Hello from Flask!"})

@app.route('/')
def index():
    # Fetch unique values for dropdowns from the DataFrame
    companies = sorted(car['company'].unique())
    years = sorted([int(year) for year in car['year'].unique()], reverse=True)  # Convert int64 to int

    return jsonify({
        'companies': companies,
        'years': years
    })

@app.route('/get_cars', methods=['POST'])
def get_cars():
    """Returns car models and fuel types based on the selected company."""
    try:
        data = request.get_json()
        company = data.get("company")

        if not company:
            return jsonify({'error': 'Company is required'}), 400

        # Filter cars by company
        filtered_cars = car[car['company'] == company]

        # Get unique car models for the selected company
        car_models = sorted(filtered_cars['name'].unique().tolist())

        # Get unique fuel types for the selected company
        fuel_types = sorted(filtered_cars['fuel_type'].unique().tolist())

        return jsonify({
            'car_models': car_models,
            'fuel_types': fuel_types
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received Data:", data)
        
        # Validate required fields
        required_fields = ["company", "car_model", "year", "fuel_type", "kilo_driven"]
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Convert input values
        company = data['company']
        car_model = data['car_model']
        year = int(data['year'])
        fuel_type = data['fuel_type']
        kms_driven = int(data['kilo_driven'])

        # Ensure the car model exists for the selected company
        if car_model not in car[car['company'] == company]['name'].values:
            return jsonify({'error': 'Invalid car model for the selected company'}), 400

        # Ensure fuel type exists for the selected company
        if fuel_type not in car[car['company'] == company]['fuel_type'].values:
            return jsonify({'error': 'Invalid fuel type for the selected company'}), 400

        # Prepare data
        input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Predict price
        prediction = model.predict(input_data)
        predicted_price = int(round(prediction[0]))  # Ensure integer value

        # Convert numeric price to words (Indian format)
        price_in_words = f"Rupees {num2words(predicted_price, lang='en_IN').capitalize()} Only"

        return jsonify({'predicted_price': predicted_price, 'price_in_words': price_in_words})

    except ValueError as ve:
        return jsonify({'error': f'Value error: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
