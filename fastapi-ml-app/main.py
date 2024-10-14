import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import traceback
import holidays
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import urllib.parse
# Load environment variables from a .env file
load_dotenv()
# Create the FastAPI app
app = FastAPI()

# Enable CORS for all origins (for testing purposes, you can restrict this later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust it as per your need
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
GOOGLE_MAP_API_KEY = os.getenv("GOOGLE_MAP_API_KEY")

# Load the pre-trained machine learning model from a .pkl file
try:
    model = joblib.load('CC_model.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define the input data schema
class InputData(BaseModel):
    NO_OF_ITEMS: list
    EXPECTED_DELIVERY_DAYS: list
    SHIPMENT_DATE: list
    EXPECTED_DATE: list
    DISTANCE_KM: list
    NUM_WEEKEND: list
    NUM_PH: list
    TYPE_OF_ITEM_EXPRESS_DOCUMENTS: list
    TYPE_OF_ITEM_OTHERS: list
    TYPE_OF_ITEM_PACKAGES_AND_PARCELS: list

# Define the feature names that the model was trained on
FEATURE_NAMES = [
    'NO_OF_ITEMS',
    'EXPECTED_DELIVERY_DAYS',
    'DISTANCE_KM',
    'NUM_WEEKENDS',  # This replaces NUM_WEEKEND
    'NUM_HOLIDAYS',  # This replaces NUM_PH
    'TYPE_OF_ITEM_EXPRESS DOCUMENTS',
    'TYPE_OF_ITEM_OTHERS',
    'TYPE_OF_ITEM_PACKAGES AND PARCELS'
]



def count_weekends_and_holidays(start_date, end_date):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    years = range(start_date.year, end_date.year + 1)
    holiday_dates = set()
    
    for year in years:
        malaysia_holidays = holidays.Malaysia(years=[year])
        holiday_dates.update(malaysia_holidays.keys())
    
    print(f"Holiday Dates from {start_date.year} to {end_date.year}: {sorted(holiday_dates)}")
    num_weekends = 0
    num_holidays = 0
    current_date = start_date
    
    while current_date <= end_date:
        if current_date.weekday() >= 5:  
            num_weekends += 1
        if current_date.date() in holiday_dates: 
            num_holidays += 1
        current_date += timedelta(days=1)

    return num_weekends, num_holidays

@app.get("/distance/")
def get_distance_matrix(origin: str, destination: str):
    try:
        encoded_origin = urllib.parse.quote(origin)
        encoded_destination = urllib.parse.quote(destination)
        encoded_api_key = urllib.parse.quote(GOOGLE_MAP_API_KEY)

        url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={encoded_origin}&destinations={encoded_destination}&key={encoded_api_key}"
        
        response = requests.get(url)

        print("Response from Distance Matrix API:", response.status_code, response.text)

        if response.status_code == 200:
            return response.json()  # Return the JSON response from the Distance Matrix API
        else:
            raise HTTPException(status_code=500, detail="Error fetching data from Google API")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
# API endpoint to make predictions
@app.post('/predict')
async def predict(input_data: InputData):
    try:
        # Log the input data for debugging
        print(f"Received input data: {input_data}")
        num_weekends, num_holidays = count_weekends_and_holidays(input_data.SHIPMENT_DATE[0], input_data.EXPECTED_DATE[0])

        # Prepare the input data as a dictionary
        input_dict = {
            'NO_OF_ITEMS': input_data.NO_OF_ITEMS,
            'EXPECTED_DELIVERY_DAYS': input_data.EXPECTED_DELIVERY_DAYS,
            'DISTANCE_KM': input_data.DISTANCE_KM,
            'NUM_WEEKENDS': num_weekends,  # Renamed
            'NUM_HOLIDAYS': num_holidays,       # Renamed
            'TYPE_OF_ITEM_EXPRESS DOCUMENTS': input_data.TYPE_OF_ITEM_EXPRESS_DOCUMENTS,
            'TYPE_OF_ITEM_OTHERS': input_data.TYPE_OF_ITEM_OTHERS,
            'TYPE_OF_ITEM_PACKAGES AND PARCELS': input_data.TYPE_OF_ITEM_PACKAGES_AND_PARCELS
        }

        # Log the prepared dictionary
        print(f"Prepared input dict: {input_dict}")

        # Convert the input dictionary to a DataFrame with the correct feature names
        input_df = pd.DataFrame(input_dict)

        # Log the DataFrame for debugging
        print(f"Input DataFrame: \n{input_df}")

        # Make the prediction using the model
        prediction = model.predict(input_df)

        # Log the prediction
        print(f"Prediction: {prediction}")

        # Return the prediction
        return {'prediction': prediction.tolist()}
    
    except Exception as e:
        # Log the full stack trace
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
