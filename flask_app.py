"""
Programmers: Kyle Pavey and Adam Lee
Class: 322-01, Spring 2021
Final Project
5/5/21
"""

import os
import pickle 
from flask import Flask, jsonify, request 

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "<h1>Welcome to my App</h1>", 200

@app.route("/predict", methods=["GET"])
def predict():
    track_duration = request.args.get("track_duration", "")
    artist_popularity = request.args.get("artist_popularity", "")
    genre = request.args.get("genres", "")
    available_markets = request.args.get("available_markets", "")
    danceability = request.args.get("danceability", "")
    acousticness = request.args.get("acousticness", "")
    tempo = request.args.get("tempo", "")

    prediction = predict_music_well([track_duration, artist_popularity, genre, available_markets, danceability, acousticness, tempo])

    if prediction is not None:
        result = {"prediction": prediction}
        return jsonify(result), 200
    else: 
        # failure!!
        return "Error making prediction", 400


def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                return tdidt_predict(header, value_list[2], instance)
    else: 
        return tree[1]

def predict_music_well(instance):
    infile = open("tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()
    print("header:", header)
    print("tree:", tree)

    try: 
        return tdidt_predict(header, tree, instance)
    except:
        return None


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)