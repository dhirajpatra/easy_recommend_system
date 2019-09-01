#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 20:17:58 2019

@author: dhirajpatra
"""
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# fetch data and format it
data = fetch_movielens(min_rating=4.0)

# print training and testing data
print(repr(data['train']))
print(repr(data['test']))

# create model
model = LightFM(loss='warp')
# train model
model.fit(data['train'], epochs=30, num_threads=2)


# taking sample recommendation from model and data
def sample_recommendation(model, data, user_ids):
    # number of users and movies in training data
    n_users, n_items = data['train'].shape

    # generate recommendation for each user we input
    for user_id in user_ids:

        # movie they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

        # movies our model predicts that they will like
        scares = model.predict(user_id, np.arange(n_items))
        # rank them in order most liked to least
        top_items = data['item_labels'][np.argsort(-scares)]

        # print out the results
        print("\r\n\r\nUser ID: %s" % user_id)
        print("    Known positives from user's review are following:")

        i = 1
        for x in known_positives[:3]:
            print("%d)  %s" % (i, x))
            i += 1

        print("\r\n    Recommended movies to watch:")

        i = 1
        for x in top_items[:3]:
            print("%d)  %s" % (i, x))
            i += 1


# checking with 3 random user ids        
sample_recommendation(model, data, [4, 35, 400])
