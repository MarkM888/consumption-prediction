import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests

def welcome_page():
    # Customize page background and text colors
    page_bg_img = '''
    <style>
    body {
        background-color: black;
        color: white;
    }
    h1, h2, h3 {
        color: red;
    }
    p {
        color: black;
    }
    .st-b2 {
        color: black;
    }
    .css-1cpxqw2 a, .css-1cpxqw2, .css-1d391kg {
        color: black !important;
    }
    .st-cp {
        color: black !important;
    }
    .css-qrbaxs, .css-qrbaxs * {
        color: black !important;
    }
    .stButton button {
        background-color: red;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Display an image on the welcome page
    # st.image('/mnt/c/Users/Usuario/Downloads/smart meter.jpg', use_column_width=True)

    # Display the content
    st.markdown('<h1>Project Overview: Consumption Prediction</h1>', unsafe_allow_html=True)

    st.markdown('<h2>Objective</h2>', unsafe_allow_html=True)
    st.markdown('<p>The "Consumption Prediction" project is focused on accurately forecasting household energy consumption to optimize energy usage and reduce associated costs. Utilizing advanced machine learning techniques, the project aims to predict total energy consumption by analyzing various factors such as weather conditions, demographic data, and appliance usage.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Dataset: IDEAL Household Energy</h2>', unsafe_allow_html=True)
    st.markdown('<p>The dataset for this project was collected from 255 homes in the UK over a two-year period. It provides comprehensive details on energy usage, environmental conditions, and demographic factors.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Comprehensive Data Capture</h2>', unsafe_allow_html=True)
    st.markdown('<p>The dataset includes the following variables:</p>', unsafe_allow_html=True)
    st.markdown('<ul><li><strong>Electricity and Gas Consumption</strong>: Detailed records of energy usage.</li><li><strong>Environmental Factors</strong>: Temperature and humidity data.</li><li><strong>Demographic Data</strong>: Household size, income levels, and energy awareness.</li><li><strong>Appliance Usage</strong>: Information on the usage patterns of different household appliances.</li></ul>', unsafe_allow_html=True)

    st.markdown('<h2>Longitudinal Approach</h2>', unsafe_allow_html=True)
    st.markdown('<p>Data collection spanned from August 2016 to June 2018, offering a multi-year, seasonal snapshot of household energy consumption patterns.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Data Preprocessing and Model Development</h2>', unsafe_allow_html=True)
    st.markdown('<p>The initial dataset was 200GB in size but was efficiently reduced to 100KB through preprocessing. The objective is to leverage this refined dataset to develop a robust machine learning model capable of predicting household energy consumption in kilowatts based on the input variables.</p>', unsafe_allow_html=True)

    st.markdown('<h2>Project Goals</h2>', unsafe_allow_html=True)
    st.markdown('<ul><li><strong>Accurate Predictions</strong>: Develop a reliable model to forecast energy consumption.</li><li><strong>Cost Optimization</strong>: Enable households to reduce energy costs through better consumption predictions.</li><li><strong>Energy Efficiency</strong>: Support initiatives aimed at optimizing energy usage and enhancing overall efficiency.</li></ul>', unsafe_allow_html=True)

    st.markdown('<p>This project will provide valuable insights and tools for energy management, benefiting both consumers and energy providers.</p>', unsafe_allow_html=True)

def prediction_page():
    st.markdown('<h1>House Consumption Predictor</h1>', unsafe_allow_html=True)

    cities = ['London', 'Birmingham', 'Manchester', 'Leeds-Bradford',
              'Glasgow', 'Southampton-Portsmouth','Liverpool', 'Newcastle',
              'Nottingham','Sheffield']

    # Sidebar inputs
    st.sidebar.header('User Inputs')
    num_rooms = st.sidebar.number_input('Number of Rooms', min_value=1, value=1)
    location = st.sidebar.selectbox('Location', cities)
    working_status = st.sidebar.selectbox('Working Status', ['Onsite', 'Work from Home'])
    hometype = st.sidebar.selectbox('Type of House', ['Flat', 'House'])

    # Convert working status to binary
    working_status_code = 1 if working_status == 'Work from Home' else 0

    # Get latitude and longitude for the selected city
    city_lat_long = {
        'London': [51.5074,0.1278],
        'Birmingham': [52.4862, 1.8904],
        'Manchester': [53.4839, 2.2446],
        'Leeds-Bradford': [53.8008,1.5491],
        'Glasgow': [55.8642, 4.2518],
        'Southampton-Portsmouth': [50.9097, 1.4044],
        'Liverpool': [53.4084, 2.9916],
        'Newcastle': [54.9783, 1.6178],
        'Nottingham': [52.9548, 1.1581],
        'Sheffield': [53.3811, 1.4701]
    }

    #Query the weather API to get the temperature for the next 7 days for the selected city
    lat = city_lat_long[location][0]
    long = city_lat_long[location][1]

    # Weather API
    weather_url = f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={long}&daily=temperature_2m_min,temperature_2m_max'

    w_response = requests.get(weather_url).json()['daily']
    dates = w_response['time']
    temps_min = w_response['temperature_2m_min']
    temps_max = w_response['temperature_2m_max']
    temps_avg = [(temps_min[i] + temps_max[i])/2 for i in range(7)]

    # Display weather data
    weather_df = pd.DataFrame({
        'Date': dates,
        'Min Temp (°C)': temps_min,
        'Max Temp (°C)': temps_max,
        'Avg Temp (°C)': temps_avg
    })

    st.markdown('<h2>7-Day Weather Forecast</h2>', unsafe_allow_html=True)
    st.dataframe(weather_df)


    # Prepare features for prediction
    features = {
        'room_count': str(num_rooms),
        'workstatus': str(working_status_code),
        'temp_mean': temps_avg[0],
        'temp_min': temps_min[0],
        'temp_max': temps_max[0],
        'total_area': num_rooms * 100
    }

    # Add hometype feature
    hometype_code = 'house_or_bungalow' if hometype == 'House' else 'flat'
    features['hometype'] = hometype_code

    # Prediction Model Public API
    url = 'https://api-w2mh3no3sa-ew.a.run.app/predict'

    # Define the Number of Residents with a slider
    num_people = st.slider('Number of People', min_value=1, max_value=10, value=1)
    features['residents'] = str(num_people)

    # Create post request to the local API endpoint
    response = requests.post(url, json=features).json()
    prediction = response['prediction']

    # Display the prediction
    st.write(f'### Electricity Prediction: {prediction:.2f} kWh')

    # Generate daily predictions for the next 7 days
    daily_predictions = []

    for i in range(len(dates)):
        # Update features with updated temps
        features['temp_mean'] = str(temps_avg[i])
        features['temp_min'] = str(temps_min[i])
        features['temp_max'] = str(temps_max[i])

        # Daily Prediction Value
        predictions = requests.post(url, json=features).json()['prediction']
        daily_predictions.append((dates[i], predictions))

    # Create a dataframe for daily predictions
    predictions_df = pd.DataFrame(daily_predictions, columns=['Date', 'Predicted Consumption (kWh)'])

    # Plot the predictions with gradient colors
    st.markdown('<h2>Predicted Consumption Over Time</h2>', unsafe_allow_html=True)

    chart_data = predictions_df.set_index('Date').reset_index()
    bar_chart = alt.Chart(chart_data).mark_bar().encode(
        x='Date:T',
        y='Predicted Consumption (kWh):Q',
        color=alt.Color('Predicted Consumption (kWh):Q', scale=alt.Scale(scheme='viridis'))
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(bar_chart, use_container_width=True)

if __name__ == '__main__':
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Welcome", "Prediction"])
    if page == "Welcome":
        welcome_page()
    else:
        prediction_page()
