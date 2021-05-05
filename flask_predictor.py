"""
Programmers: Kyle Pavey and Adam Lee
Class: 322-01, Spring 2021
Final Project
5/5/21
"""

import requests # lib to make http requests
import json # lib to help with parsing JSON objects

# url = "https://interview-flask-app.herokuapp.com/predict?level=Junior&lang=Java&tweets=yes&phd=yes"
url = "http://cpsc322-final.herokuapp.com/predict?track_duration=long&artist_popularity=5&genre=latin&available_markets=small&danceability=2&acousticness=2&tempo=3"



# make a GET request to get the search results back
# https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods
response = requests.get(url=url)

# first thing... check the response status code 
status_code = response.status_code
print("status code:", status_code)

if status_code == 200:
    # success! grab the message body
    json_object = json.loads(response.text)
    print(json_object)
