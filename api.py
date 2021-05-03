"""
Programmers: Kyle Pavey and Adam Lee
Class: 322-01, Spring 2021
Final Project
4/21/21
"""
import os
import urllib.request 
import json
import pandas as pd
import requests
import importlib
import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils
import mysklearn.mypytable
importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable 

client_id = ''
client_secret = ''

AUTH_URL = 'https://accounts.spotify.com/api/token'

auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': client_id,
    'client_secret': client_secret,
})

auth_response_data = auth_response.json()

access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

BASE_URL = 'https://api.spotify.com/v1/'

track_quantity = 5
track_table = []
characters = 'abcdefghijklmnopqrstuvwxyz'

for i in range(track_quantity):
    random_search_key, offset = myutils.getRandomSearch(characters)
    # print(random_search_key, offset)
    search_json_response = requests.get(BASE_URL + "search{}{}{}{}".format("?q="+str(random_search_key), "&offset="+str(offset), "&limit=1", "&type=track"), headers=headers)
    search_json_object = search_json_response.json()

    # Information about the track
    track_id = search_json_object["tracks"]["items"][0]["id"]
    track_name = search_json_object["tracks"]["items"][0]["name"]
    album_id = search_json_object["tracks"]["items"][0]["album"]["id"]
    artist_id = search_json_object["tracks"]["items"][0]["artists"][0]["id"]
    artist_name = search_json_object["tracks"]["items"][0]["artists"][0]["name"]
    track_popularity = search_json_object["tracks"]["items"][0]["popularity"]
    track_duration = search_json_object["tracks"]["items"][0]["duration_ms"]

    # Information about the artist
    artist_obj = requests.get(BASE_URL + "artists/{}/".format(artist_id), headers=headers)
    artist_obj = artist_obj.json()
    artist_popularity = artist_obj["popularity"]
    genres = artist_obj["genres"]

    # Information about the album
    # album_obj = requests.get(BASE_URL + "albums/{}/".format(album_id), headers=headers)
    # album_obj = album_obj.json()
    # # print(album_obj)
    # # print("")
    # album_genres = album_obj["genres"]

    # Information about the audio features
    audio_features = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)
    audio_features_json = audio_features.json()
    danceability = audio_features_json["danceability"]
    acousticness = audio_features_json["acousticness"]
    tempo = audio_features_json["tempo"]

    track_table.append([track_id, track_name, track_popularity, track_duration, artist_name, artist_popularity, genres, danceability, acousticness, tempo])


column_names = ["track_id", "track_name", "track_popularity", "track_duration", "artist_name", "artist_popularity", "genres", "danceability", "acousticness", "tempo"]
track_pytable = MyPyTable(column_names, track_table)
music_fname = os.path.join("input_data", "music-data.csv")
track_pytable.save_to_file(music_fname)
# print(track_pytable.data)


    

    
