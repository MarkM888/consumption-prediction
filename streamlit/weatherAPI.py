import requests
import pandas as pd

# Define the API key and base URL
api_key = '***REMOVED***'
base_url = "http://api.weatherapi.com/v1"

# Function to get weather data from WeatherAPI
def get_weather(city, api_key):
    forecast_url = f"{base_url}/forecast.json?key={api_key}&q={city}&days=10&aqi=no&alerts=no"
    response = requests.get(forecast_url)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    else:
        return response.json()

# Function to extract and organize weather data into a DataFrame
def process_weather_data(weather_data):
    dates = []
    min_temp = []
    max_temp = []
    avg_temp = []

    for day in weather_data['forecast']['forecastday']:
        dates.append(day['date'])
        min_temp.append(day['day']['mintemp_c'])
        max_temp.append(day['day']['maxtemp_c'])
        avg_temp.append(day['day']['avgtemp_c'])

    df_temp_final = pd.DataFrame({
        'min_temp': min_temp,
        'max_temp': max_temp,
        'avg_temp': avg_temp
    }, index=dates)

    return df_temp_final

# Combined function
def get_city_weather(city):
    weather_data = get_weather(city, api_key)
