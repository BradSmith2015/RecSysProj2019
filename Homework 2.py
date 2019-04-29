#!/usr/bin/env python

import pandas as pd
from User_KNN import User_KNN
import time
import random
# this will import all the files from the tools file.
from tools import *
import pickle

# get a joke to lighten the mood.


all_jokes = load_jokes()
print(len(all_jokes.index))
test_var = random.randint(min(all_jokes.index), max(all_jokes.index))

print(get_joke(test_var))



start = time.time()
print('starting')

ratings_file = 'jester_ratings_100.csv'
ratings = get_ratings(ratings_file)
print("rating length is", len(ratings['userId'].unique()))


alg = User_KNN(20, sim_threshold=0)
print('before fitting ')



# how are we going to store this value. it has the type of <class 'User_KNN.User_KNN'>
alg.fit(ratings)


"""
I need to store alg
 _profile_lengths
 _profile_means
 _ratings
 _sim_cache
 
"""

print("the type of the algoritm is", type(alg)) #<class 'User_KNN.User_KNN'>
done = time.time()
elapsed = done - start
print('elapsed time \n\n', elapsed)



print('after fitting ') # for 300 itmes it is close to 300 seconds.

# so this is saying for user 10, what are the rankings for the following products.
# it has _item_means,
aa = alg.predict_for_user(10, [2,4,9,11])

bb = alg.predict_for_user(1, [1,1,1,1])
print(aa, bb)

print('-----------------------------------------------------------------------------------------')

print(alg.predict_for_user(1, [1,1,1,1]))


