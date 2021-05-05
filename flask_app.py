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
    # task: extract the remaining 3 args

    # get a prediction for this unseen instance via the tree
    # return the prediction as a JSON response
    prediction = predict_music_well([track_duration, artist_popularity, genre, available_markets, danceability, acousticness, tempo])
    # if anything goes wrong, predict_interviews_well() is going to return None
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
        # now I need to find which "edge" to follow recursively
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match!! recurse!!
                return tdidt_predict(header, value_list[2], instance)
    else: # "Leaf"
        return tree[1] # leaf class label

def predict_music_well(instance):
    # 1. we need to a tree (and its header)
    # we need to save a trained model (fit()) to a file
    # so we can load that file into memory in another python
    # process as a python object (predict())
    # import pickle and "load" the header and interview tree 
    # as Python objects we can use for step 2
    print("predicting well")
    infile = open("tree.p", "rb")
    header, tree = pickle.load(infile)
    infile.close()
    print("header:", header)
    print("tree:", tree)

    # 2. use the tree to make a prediction
    try: 
        return tdidt_predict(header, tree, instance) # recursive function
    except:
        return None


if __name__ == "__main__":
    # app.run(debug=False)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)