#!/usr/bin/env python



import pandas as pd

from User_KNN import User_KNN

print('starting')


ratings = pd.read_csv('jester_ratings_300.csv', sep=',', header=0, names=['userId','movieId','rating'])
print("rating length is", len(ratings['userId'].unique()))


# type(ratings)
# ratings.index
#
#
# alg = User_KNN(20, sim_threshold=0)
# alg.fit(ratings)



print('user test user one' ,alg.predict_for_user(870, [2,4,9,11]))

print('test user two ' ,alg.predict_for_user(870, [1,1,1,1]))





