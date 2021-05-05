import pickle # standard python library

# "pickle" an object (AKA object serialization)
# save a Python object to a binary file

# "unpickle" an object (AKA object de-serialization)
# load a Python object from a binary file (back into memory)

# for your project, pickle an instance MyRandomForestClassifier, MyDecisionTreeClassifier
# for demo use header and interview_tree below

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
header = ["track_duration", "artist_popularity", "genres", "available_markets", "danceability", "acousticness", "tempo"]

music_tree = \
['Attribute', 'available_markets', 
    ['Value', 'small', 
        ['Leaf', 'Above Average', 0, 74]], 
    ['Value', 'large', 
        ['Attribute', 'genres', 
            ['Value', 'pop', 
                ['Attribute', 'tempo', 
                    ['Value', '2', 
                        ['Leaf', 'Above Average', 0, 224]], 
                    ['Value', '3', 
                        ['Leaf', 'Above Average', 0, 54]], 
                    ['Value', '1', 
                        ['Leaf', 'Above Average', 1, 279]]]], 
            ['Value', 'country', 
                ['Leaf', 'Above Average', 0, 85]], 
            ['Value', 'r&b', 
                ['Leaf', 'Above Average', 0, 30]], 
            ['Value', 'latin', 
                ['Leaf', 'Above Average', 0, 35]], 
            ['Value', 'hip hop', 
                ['Attribute', 'tempo', 
                    ['Value', '2', 
                        ['Attribute', 'acousticness', 
                            ['Value', '1', 
                                ['Leaf', 'Above Average', 0, 101]], 
                            ['Value', '2', 
                                ['Leaf', 'Above Average', 0, 15]], 
                            ['Value', '3', 
                                ['Leaf', 'Above Average', 3, 119]]]], 
                    ['Value', '3', 
                        ['Leaf', 'Above Average', 0, 45]], 
                    ['Value', '1', 
                        ['Leaf', 'Average', 1, 165]]]], 
            ['Value', 'rock', 
                ['Attribute', 'acousticness', 
                    ['Value', '1', 
                        ['Leaf', 'Above Average', 0, 68]], 
                    ['Value', '2', 
                        ['Leaf', 'Above Average', 0, 10]], 
                    ['Value', '3', 
                        ['Leaf', 'Above Average', 3, 81]]]], 
            ['Value', 'rap', 
                ['Attribute', 'danceability', 
                    ['Value', '3', 
                        ['Leaf', 'Above Average', 0, 125]], 
                    ['Value', '2', 
                        ['Leaf', 'Above Average', 0, 35]], 
                    ['Value', '1', 
                        ['Leaf', 'Above Average', 2, 162]]]], 
            ['Value', 'metal', 
                ['Leaf', 'Above Average', 0, 31]], 
            ['Value', 'classical', 
                ['Leaf', 'Low', 3, 883]], 
            ['Value', 'indie', 
                ['Leaf', 'Above Average', 0, 12]]]], 
    ['Value', 'medium', 
        ['Leaf', 'Low', 0, 156]]]

packaged_object = [header, music_tree]
# pickle packaged_object
outfile = open("tree.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()