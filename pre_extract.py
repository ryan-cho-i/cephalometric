import json 
import numpy as np

def detect_index (file_path) :
        
    with open(file_path, "r") as json_file:       
        json_data = json.load(json_file)
        list_keys = list(json_data.keys())

        index_focused = []

        i = 0
        for i in range(len(json_data.keys())) : 
            if len(json_data[list_keys[i]]) > 2 :
                index_focused.append(i)
        
        j = 0
        for j in range(len(index_focused)) :   
            print(list_keys[index_focused[j]], index_focused[j])

    return index_focused

def judge_word (word) :

    if 'a' in word : 
        return 1
    elif 'i' in word : 
        return 1
    elif 'e' in word : 
        return 1
    elif 'o' in word : 
        return 1
    elif 'u' in word : 
        return 1
    elif 'P' in word : 
        return 1
    elif 'M' in word : 
        return 1
    elif 'R' in word : 
        return 1

<<<<<<< HEAD
def extract (file_path, target) :
=======
def extracte (file_path, target) :
>>>>>>> e45329e9a9f7e9da5b669037c4a395b5315d8463

    with open(file_path, "r") as json_file:       
        json_data = json.load(json_file)        

    return json_data [target]

def replace_word_with_number (coord) :

    coord_replaced = []
    coord_replaced_index = []
    i = 0
    for i in range(len(coord)) : 
        if judge_word(coord[i]) :
            coord_replaced.append(coord[i])
            coord_replaced_index.append(i)
        
    return coord, coord_replaced, coord_replaced_index 

def extractor (file_path, target) :

    with open(file_path, "r") as json_file:       
        json_data = json.load(json_file)
        coord, word, index = replace_word_with_number (json_data [target])

        i = 0
        for i in range(len(word)) :
<<<<<<< HEAD
            coord [index[i]] = extract (file_path, word[i])
=======
            coord [index[i]] = extracte (file_path, word[i])
>>>>>>> e45329e9a9f7e9da5b669037c4a395b5315d8463

    return coord 









