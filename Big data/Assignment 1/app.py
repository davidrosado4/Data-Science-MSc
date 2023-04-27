import requests
import pandas as pd
import pymongo
import datetime
import json

# Read the csv
cities = pd.read_csv('Top100-US.csv', sep=';')

# Get the url of the weatherapi
url = "https://weatherapi-com.p.rapidapi.com/current.json"

myclient = pymongo.MongoClient("mongodb://mongo:27017/mydb")

# Use database named Big-dataUB
mydb = myclient["Big-dataUB"]

# Use collection named "city_weather"
mycol = mydb["city_weather"]

print("Creating the database...")

for i in range(len(cities['Zip'])):
    zip_code = json.dumps(int(cities['Zip'].values[i]))
    city_name = cities['City'].values[i]
    querystring = {"q": zip_code}

    headers = {
        "X-RapidAPI-Key": "5713db1449msh8eb178be599f1e8p15c209jsn0194ce1bc670",
        "X-RapidAPI-Host": "weatherapi-com.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    # Dictionary as a document
    date_now = datetime.datetime.now()
    date_now_string = date_now.strftime("%m/%d/%Y, %H:%M:%S")
    dict = {"zip": zip_code, "city": city_name, "created_at": date_now_string, "weather": response.text}
    # Insert a document to the collection
    x = mycol.insert_one(dict)
print("Finish!")