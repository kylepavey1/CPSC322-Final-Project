import os
import pickle 
from flask import Flask, jsonify, request 

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return "<h1>Welcome to my App</h1>", 200

@app.route("/predict", methods=["GET"])
def predict():
    # goal is to extract the 4 attribute values from query string
    # use the request.args dictionary
    track_duration = request.args.get("track_duration", "")
    artist_popularity = request.args.get("artist_popularity", "")
    genre = request.args.get("genres", "")
    available_markets = request.args.get("available_markets", "")
    danceability = request.args.get("danceability", "")
    acousticness = request.args.get("acousticness", "")
    tempo = request.args.get("tempo", "")

    print("level:", track_duration, artist_popularity, genre, danceability)
    # task: extract the remaining 3 args

    # get a prediction for this unseen instance via the tree
    # return the prediction as a JSON response

    # prediction = predict_interviews_well([level, lang, tweets, phd])
    # if anything goes wrong, predict_interviews_well() is going to return None
    # if prediction is not None:
    #     result = {"prediction": prediction}
    #     return jsonify(result), 200
    # else: 
    #     # failure!!
    #     return "Error making prediction", 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)