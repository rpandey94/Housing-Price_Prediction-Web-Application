# Housing price prediction web application

import os
import joblib
import locale
import numpy as np
import pandas as pd
import streamlit as st


# Reading data from csv

file_path = os.path.join("datasets", "housing")

def fetch_data(file_path=file_path):
    file_path = os.path.join(file_path, "housing.csv")
    return pd.read_csv(file_path)


# Fetching housing data
df = fetch_data()

# Collecting data
ocean_proximity = st.sidebar.selectbox(
    'What type of location do you want?',
    ('NEAR OCEAN', 'NEAR BAY', 'ISLAND', 'INLAND', '<1H OCEAN'))
total_rooms = st.sidebar.number_input('Total Rooms (within a block)', 1000)
total_bedrooms = st.sidebar.number_input('Total Bedrooms (within a block)', 500)
housing_median_age = st.sidebar.number_input('Median House Age In Years (within a block)', 20)
population = st.sidebar.number_input('Number of people (within a block)', 8000)
households = st.sidebar.number_input('Total number of households (within a block)', 200)
median_income = st.sidebar.number_input('Income Of Households (within a block) (measured in tens of thousands of US Dollars)', 0.4999, 15.0001)


# filter on ocean proximity
proximity_map = df[df["ocean_proximity"] == ocean_proximity]

# Adding a title to the web page
st.write("""
        # California Housing Price Prediction Web Application

        ## Housing Blocks on Map

	     """)

# Data on Geo Map
map_data = proximity_map[["latitude", "longitude"]]
st.map(map_data)
# Show Raw data and column description
if st.checkbox('Show Raw Data And Column Description'):
    st.write("""

         ## The California Housing Price Dataset

         """)
    st.write(df)
    st.write(""" 


## Column Descriptions

1. longitude: A measure of how far west a house is; a higher value is farther west

2. latitude: A measure of how far north a house is; a higher value is farther north

3. housing_median_age: Median age of a house within a block; a lower number is a newer building

4. total_rooms: Total number of rooms within a block

5. total_bedrooms: Total number of bedrooms within a block

6. population: Total number of people residing within a block

7. households: Total number of households, a group of people residing within a home unit, for a block

8. median_income: Median income for households within a block of houses (measured in tens of thousands of US Dollars)

9. median_house_value: Median house value for households within a block (measured in US Dollars)

10. ocean_proximity: Location of the house w.r.t ocean/sea """)

# Making a map to display location of block for which median price will be predicted
st.write(""" ## Make Prediction """)
label = list(zip(proximity_map["latitude"], proximity_map["longitude"]))
location = st.selectbox('Select From Available Geo Coordinates Of Blocks For Predicting House Price: ', label, index=500)
latitude = location[0]
longitude = location[1]
lst = [location[0]]
lst2 = [location[1]]
lst = [float(i) for i in lst]
lst2 = [float(i) for i in lst2]
geo_map = pd.DataFrame(list(zip(lst, lst2)), columns =['latitude', 'longitude'])
st.map(geo_map, zoom=5)
st.write('Map is showing the location of the block for which prediction will be generated', geo_map)

# Making predictions

my_model_loaded = joblib.load("Saved_Models/my_model.pkl")


if (ocean_proximity == "<1H OCEAN"):
    ocean_proximity_list = [1, 0, 0, 0, 0]
elif (ocean_proximity == "INLAND"):
    ocean_proximity_list = [0, 1, 0, 0, 0]
elif (ocean_proximity == "NEAR OCEAN"):
    ocean_proximity_list = [0, 0, 1, 0, 0]
elif (ocean_proximity == "NEAR BAY"):
    ocean_proximity_list = [0, 0, 0, 1, 0]
elif (ocean_proximity == "ISLAND"):
    ocean_proximity_list = [0, 0, 0, 0, 1]
else:
    ocean_proximity_list = [0, 0, 0, 0, 0]

rooms_per_household = total_rooms / households
population_per_household = population / households

features = [longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population,
            households, median_income, rooms_per_household, population_per_household] + ocean_proximity_list

features = np.array(features)
features = features.reshape(1, -1)
prediction = my_model_loaded.predict(features)
value = prediction[0]
locale.setlocale( locale.LC_ALL, '' )

if st.button('Predict House Price'):

    st.write("House price is " + str(locale.currency( value, grouping=True )))

st.write(""" ### © Rishav Pandey 2020 """)