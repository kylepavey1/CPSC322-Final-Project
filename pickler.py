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
header = ["level", "lang", "tweets", "phd"]
interview_tree = \
["Attribute", "level", 
    ["Value", "Senior", 
        ["Attribute", "tweets", 
            ["Value", "yes", 
                ["Leaf", "True", 2, 5]
            ],
            ["Value", "no", 
                ["Leaf", "False", 3, 5]
            ]
        ]
    ],
    ["Value", "Mid", 
        ["Leaf", "True", 4, 14]
    ],
    ["Value", "Junior", 
        ["Attribute", "phd", 
            ["Value", "yes", 
                ["Leaf", "False", 2, 5]
            ],
            ["Value", "no", 
                ["Leaf", "True", 3, 5]
            ]
        ]
    ]
]

music_tree = \
['Attribute', 'att3', 
    ['Value', 'small', 
        ['Leaf', 'Above Average', 0, 74]], 
    ['Value', 'large', 
        ['Attribute', 'att2', 
            ['Value', 'pop', 
                ['Attribute', 'att6', 
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
                ['Attribute', 'att6', 
                    ['Value', '2', 
                        ['Attribute', 'att5', 
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
                ['Attribute', 'att5', 
                    ['Value', '1', 
                        ['Leaf', 'Above Average', 0, 68]], 
                    ['Value', '2', 
                        ['Leaf', 'Above Average', 0, 10]], 
                    ['Value', '3', 
                        ['Leaf', 'Above Average', 3, 81]]]], 
            ['Value', 'rap', 
                ['Attribute', 'att4', 
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

packaged_object = [header, interview_tree]
# pickle packaged_object
outfile = open("tree.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()