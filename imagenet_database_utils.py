#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:29:53 2018
@author: guy.hacohen
"""

import csv
#import json
import ast
import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
import scipy.io as sio
#import numpy as np
#from keras_squeezenet import SqueezeNet
#from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
#import keras
from nltk.corpus import wordnet as wn
import pickle
#import urllib.request
#IMAGENET_WORDS = '/cs/labs/daphna/guy.hacohen/imagenet/words.txt'
#IMAGENET_LABELS_ID = '/cs/labs/daphna/guy.hacohen/imagenet/class_id_to_label.txt'
#IMAGENET_DATA_DIR = '/cs/dataset/ImageNet/'

def synset_to_wnid(synset):
    return synset.pos() + str(synset.offset()).zfill(8) 

def wnid_to_synset(wnid):
    pos = wnid[0]
    offset = int(wnid[1:])
    return wn.synset_from_pos_and_offset(pos, offset)

def label_to_synset(data_dict):
    wnid = data_dict["wnid"]
    return wnid_to_synset(wnid)

def wnid_children(wnid_file, metadata):
    """
    gets wnid and returns all the wnids which are children of the wnid.
    children is in the sense of every returned wnid "is a" member of 
    the synset represented by the given wnid
    """
    with open(wnid_file, "r+") as file:
        wnids = file.readlines()
    ## labels in the imagenet ar 1...1000, and i use 0...999
    wnids = [x.strip() for x in wnids]
    return [data_dict for data_dict in metadata if data_dict["wnid"] in wnids]


def cloests_words_in_wordnet(word_list, seed_word, similar_tresh):
    seed_synset = wn.synset(seed_word+".n.01")
    synsets_list = [wn.synset(word+".n.01") for word in word_list]
    similarities = [synset.lch_similarity(seed_synset) for synset in synsets_list]
    res = [idx for idx, similar in enumerate(similarities) if similar >= similar_tresh]
    return res
    
def create_wnid_to_label_dict(words_path):

    fieldnames = ("wnid","label")
    with open(words_path, 'r') as csvfile:
        reader = csv.DictReader( csvfile, fieldnames, delimiter ="\t")
        wnid_to_label = {}
        label_to_wnid = {}
        for row in reader:
            wnid_to_label[row["wnid"]] = row["label"]
            if row["label"] not in label_to_wnid:
                label_to_wnid[row["label"]] = []
            label_to_wnid[row["label"]].append(row["wnid"])
        return wnid_to_label, label_to_wnid

def create_class_id_dict(class_id_pth):
    with open(class_id_pth, 'r') as class_id_file:
        s = class_id_file.read()
        return ast.literal_eval(s)

def create_id_to_wnid_dict(label_to_wnid, id_to_label):
    res = {}
    for idd, label in id_to_label.items():
        res[idd] = label_to_wnid[label]
    return res
    
#wnid_to_label, label_to_wnid = create_wnid_to_label_dict(IMAGENET_WORDS)
#id_to_label = create_class_id_dict(IMAGENET_LABELS_ID)
#id_to_wnid = create_id_to_wnid_dict(label_to_wnid, id_to_label)


#def create_image_list(id_to_label_dict, label_to_wnid, database_path, outPath):
#    lines = []
#    for label in id_to_label_dict.values():
#        wnid_list = label_to_wnid[label]
#        for wnid in wnid_list:
#            images_dir = os.path.join(database_path, wnid)
#            for file_name in os.listdir(images_dir):
#                image = os.path.join(images_dir, file_name)
#                if os.path.isfile(image):
#                    pass
            
#def preprocess_database(database_dir, new_database_dir, preprocess_func, 
#                        id_to_wnid):
#    file_list = []
#    counter = 0
#    for idd, wnid_list in id_to_wnid.items():
#        for wnid in wnid_list:
#            wnid_old_dir = os.path.join(database_dir, wnid)
#            if not os.path.exists(wnid_old_dir):
#                continue
#            wnid_new_dir = os.path.join(new_database_dir, wnid)
#            if not os.path.exists(wnid_new_dir):
#                os.mkdir(wnid_new_dir)
#            for image_name in os.listdir(wnid_old_dir):
#                
#                image_old_path = os.path.join(wnid_old_dir, image_name)
#                image_new_path = os.path.join(wnid_new_dir, image_name)
#                im = plt.imread(image_old_path)
#                im = preprocess_func(im)
#                plt.imsave(image_new_path, im)
#                cur_line = str(counter) + "\t" + str(idd) + "\t" + image_new_path
#                file_list.append(cur_line)
#                counter += 1
#                if counter % 10000 == 0:
#                    print("done", counter, "images")
#    return file_list
    
def pad_int_into_str(num, wanted_length=5, pad_with="0"):
    res = str(num)
    while len(res) < wanted_length:
        res = pad_with + res
    return res
    
    
def preprocess_validation(database_dir, new_database_dir):
    images_in_base = os.listdir(database_dir)
    counter = 0
    for image_name in images_in_base:
        image_old_path = os.path.join(database_dir, image_name)
        image_new_path = os.path.join(new_database_dir, image_name)
        img = image.load_img(image_old_path, target_size=(224, 224))
#            im = preprocess_func(im)
        img.save(image_new_path)
#            cur_line = str(counter) + "\t" + str(idd) + "\t" + image_new_path
#            file_list.append(cur_line)
        counter += 1
        if counter % 10000 == 0:
            print("done", counter, "images")

def preprocess_database(database_dir, new_database_dir, metadata):
#    file_list = []
    counter = 0
#    n_classes = len(metadata)
    for class_data in metadata:
        wnid = class_data['wnid']
        wnid_old_dir = os.path.join(database_dir, wnid)
        wnid_new_dir = os.path.join(new_database_dir, wnid)
        if not os.path.exists(wnid_new_dir):
            os.mkdir(wnid_new_dir)
        class_images = os.listdir(wnid_old_dir)
        assert(len(class_images) == class_data['num_images'])
        for image_name in class_images:
            image_old_path = os.path.join(wnid_old_dir, image_name)
            image_new_path = os.path.join(wnid_new_dir, image_name)
            img = image.load_img(image_old_path, target_size=(224, 224))
#            im = preprocess_func(im)
            img.save(image_new_path)
#            cur_line = str(counter) + "\t" + str(idd) + "\t" + image_new_path
#            file_list.append(cur_line)
            counter += 1
            if counter % 10000 == 0:
                print("done", counter, "images")
    print("done", counter, "images")
    return
#    return file_list


def resize_img(img, new_size=(256, 256)):
    pil_img = PIL.Image.fromarray(img)
    pil_img = pil_img.resize(new_size, PIL.Image.ANTIALIAS)
    return np.array(pil_img)

def parse_metadata(metadata_path):
    """
    parsing the metadata file to to a 1000 entries array,
    each containing dict of specific class, with keys:
    "id" - the number of class (between 0 and 999)
    "wnid" - the wordnet id of the class. used to find the images of the class
    "label" - the label of the class, in words
    "discription" - the definition in english of the label, as presented in wordnet
    "num_images" - how many images for train are there
    """
    metadata = sio.loadmat(metadata_path)
    synsets = metadata['synsets']
    data = []
    n_classes = 1000
    for class_id in range(n_classes):
        class_dict = {}
        class_dict["id"] = class_id
        class_dict["wnid"] = synsets[class_id][0][1][0]
        class_dict["label"] = synsets[class_id][0][2][0]
        class_dict["discription"] = synsets[class_id][0][3][0]
        class_dict["num_images"] = synsets[class_id][0][7][0][0]
        data.append(class_dict)
    return data

def parse_validation_gt(validation_gt_path):
    with open(validation_gt_path, "r+") as val_file:
        validation_y = val_file.readlines()
    ## labels in the imagenet ar 1...1000, and i use 0...999
    validation_y = [int(x.strip())-1 for x in validation_y] 
    return validation_y

def count_cs_dataset(dataset_path, metadata):
    count = 0
    for idx, wnid in enumerate([clss["wnid"] for clss in metadata]):
#        print("counting class:", idx)
        path = os.path.join(dataset_path, wnid)
        try:
            count += len(os.listdir(path))
        except:
            print("wnid", wnid, "is not in the dataset")
            print("label:", metadata[idx]['label'])
#        count += len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    print(count)
    return count
    
def main():
    metadata = parse_metadata(r"/cs/labs/daphna/guy.hacohen/imagenet/ILSVRC2012_devkit_t12/data/meta.mat")
    validation_y = parse_validation_gt(r"/mnt/local/guy.hacohen/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt")
    sorted_val = sorted(validation_y)
#    print([sorted_val.index(i) for i in range(1000)])
    #count_cs_dataset('/mnt/local/guy.hacohen/ILSVRC2012_img_train', metadata)
#    preprocess_database('/mnt/local/guy.hacohen/ILSVRC2012_img_train',
#                        '/mnt/local/guy.hacohen/ILSVRC2012_img_train_preprocessed',
#                        metadata)
#    preprocess_validation("/mnt/local/guy.hacohen/ILSVRC2012_img_val",
#                          "/mnt/local/guy.hacohen/ILSVRC2012_img_val_preprocessed")
#    dog_wnid = "n02084071"
#    children_dog = wnid_children(r"/mnt/local/guy.hacohen/ILSVRC2012_devkit_t12/data/dog_children.txt",
#                                metadata)
#    dog_labels = [data_dict["id"] for data_dict in children_dog]
#    dog_labels_path = r"/mnt/local/guy.hacohen/ILSVRC2012_devkit_t12/data/dog_labels"

    children_cat = wnid_children(r"/mnt/local/guy.hacohen/ILSVRC2012_devkit_t12/data/cat_children.txt",
                                metadata)
    cat_labels = [data_dict["id"] for data_dict in children_cat]
    cat_labels_path = r"/mnt/local/guy.hacohen/ILSVRC2012_devkit_t12/data/cat_labels"
    
    print([data_dict["label"] for data_dict in children_cat])
    print([data_dict["id"] for data_dict in children_cat])
    
#    with open(cat_labels_path, "wb+") as out_file:
#        pickle.dump(cat_labels, out_file)
#    with open(dog_labels_path, "wb+") as out_file:
#        pickle.dump(dog_labels, out_file)
#    print([data_dict["label"] for data_dict in metadata])
#    print(len(children_dog))
#    print([data_dict["label"] for data_dict in children_dog])
#    print([data_dict["id"] for data_dict in children_dog])
    
if __name__ == "__main__":
    main()