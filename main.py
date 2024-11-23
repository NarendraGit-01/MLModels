
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 'KWH TOTAL SQFT', 'ELECTRICITY ACCOUNTS',
#        'RENTER-OCCUPIED HOUSING PERCENTAGE', 'RENTER-OCCUPIED HOUSING UNITS',
#        'OCCUPIED UNITS PERCENTAGE', 'AVERAGE BUILDING AGE',
#        'AVERAGE HOUSESIZE', 'AVERAGE STORIES', 'TOTAL POPULATION',
#        'ZERO KWH ACCOUNTS'

class model_input(BaseModel):

    KWH_TOTAL_SQFT: float
    ELECTRICITY_ACCOUNTS: float
    RENTER_OCCUPIED_HOUSING_PERCENTAGE: float
    RENTER_OCCUPIED_HOUSING_UNITS: float
    OCCUPIED_UNITS_PERCENTAGE: float
    AVERAGE_BUILDING_AGE: float
    AVERAGE_HOUSESIZE: float
    AVERAGE_STORIES: float
    TOTAL_POPULATION: float
    ZERO_KWH_ACCOUNTS: float

Energy_model=pickle.load(open('Enegrgy_Consumption_LinearReg.sav','rb'))

@app.post('/Enegrgy_prediction')
def Energy_pred(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    kwh_total_sqft = input_dictionary['KWH_TOTAL_SQFT']
    electricity_accounts = input_dictionary['ELECTRICITY_ACCOUNTS']
    renter_occupied_housing_percentage = input_dictionary['RENTER_OCCUPIED_HOUSING_PERCENTAGE']
    renter_occupied_housing_units = input_dictionary['RENTER_OCCUPIED_HOUSING_UNITS']
    occupied_units_percentage = input_dictionary['OCCUPIED_UNITS_PERCENTAGE']
    average_building_age = input_dictionary['AVERAGE_BUILDING_AGE']
    average_housesize = input_dictionary['AVERAGE_HOUSESIZE']
    average_stories = input_dictionary['AVERAGE_STORIES']
    total_population = input_dictionary['TOTAL_POPULATION']
    zero_kwh_accounts = input_dictionary['ZERO_KWH_ACCOUNTS']

input_list  = [kwh_total_sqft, electricity_accounts, renter_occupied_housing_percentage, renter_occupied_housing_units, occupied_units_percentage, average_building_age, average_housesize, average_stories, total_population, zero_kwh_accounts]

prediction = Energy_model.predict([input_list])

