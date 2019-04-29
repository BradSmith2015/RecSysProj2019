"""
this spike is more or less and example from the getting started,
https://lkpy.readthedocs.io/en/stable/GettingStarted.html

the data format from the GS is a little different
They only use user and ratting 
    user	item	rating	timestamp
0	196	    242	    3	    881250949
1	186	    302	    3	    891717742
2	22	    377	    1	    878887116
3	244	    51	    2	    880606923
4	166	    346	    1	    886397596

some work to transform our data set 
could be made
jester-data-3_headers.xls

we could sort by user, and 

"""


from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, item_knn as knn
from lenskit import topn


import pandas as pd
import time
import pickle
import numpy as np
import xlrd
import csv, json
from tools import *


# items_file = '../dataSets/jester_dataset_2/jester_items.dat'

# rating_file = '../dataSets/jester_dataset_2/jester_ratings.dat'
rating_file = '../jester_ratings_100.csv'

ratings = pd.read_csv(rating_file, sep=',', header=0, names=['user', 'Id', 'rating'])


# ratings = pd.read_csv(rating_file , sep='\t\t', names=['user', 'joke', 'rating'])


print(ratings)

start = time.time()
print('starting')

# alg = User_KNN(20, sim_threshold=0)
print('before fitting ')

algo_ii = knn.ItemItem(20)

algo_ii.fit(ratings)

# pickle_out = open("testing.pickle", "wb")
# pickle.dump(algo_ii, pickle_out)
# pickle_out.close()


done = time.time()
elapsed = done - start
print('elapsed time ', elapsed)


#
# # als could not be loaded.
# # algo_als = als.BiasedMF(50)
#
#
# def eval(aname, algo, train, test):
#     fittable = util.clone(algo)
#     fittable = Recommender.adapt(fittable)
#     fittable.fit(train)
#     users = test.user.unique()
#     # now we run the recommender
#     recs = batch.recommend(fittable, users, 100)
#     # add the algorithm name for analyzability
#     recs['Algorithm'] = aname
#     return recs
#
# all_recs = []
# test_data = []
#
# for train, test in xf.partition_users(ratings[['user', 'Joke 1']], 5, xf.SampleFrac(0.2)):
#     test_data.append(test)
#     all_recs.append(eval('ItemItem', algo_ii, train, test))
#     # all_recs.append(eval('ALS', algo_als, train, test))
#
# all_recs = pd.concat(all_recs, ignore_index=True)
# all_recs.head()
#
#
# test_data = pd.concat(test_data, ignore_index=True)
#
#
# rla = topn.RecListAnalysis()
# rla.add_metric(topn.ndcg)
# results = rla.compute(all_recs, test_data)
# results.head()
#
# results.groupby('Algorithm').ndcg.mean()
#
#
# results.groupby('Algorithm').ndcg.mean().plot.bar()
