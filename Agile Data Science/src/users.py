from oauth2client.service_account import ServiceAccountCredentials
import gspread
import json

scopes = [
'https://www.googleapis.com/auth/spreadsheets',
'https://www.googleapis.com/auth/drive'
]

credentials = ServiceAccountCredentials.from_json_keyfile_name("movierecommender-405816-41309bde9020.json", scopes) #access the json key you downloaded earlier 
gs = gspread.authorize(credentials) # authenticate the JSON key with gspread
sheet = gs.open("data") #open sheet
users_worksheet = sheet.worksheet("users_sheet") #open sheet
movies_worksheet = sheet.worksheet("movies")
feedback_worksheet = sheet.worksheet("feedback")


users = users_worksheet.col_values(1)
passwords = users_worksheet.col_values(2)

print(users)
print(passwords)


