#!/usr/bin/env python
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import lenskit as lk
import pandas as pd
import random
import time
from tools import *


if __name__ == '__main__':

    try:
        while True:

            print("lets start by rating a few jokes, before we can recomed anything")

            # LOAD THE DATA IN.
            ratings = get_ratings('dataSets/jester_dataset_2/jester_ratings_1000.csv')


            joke_index, rating =  ask_if_joke_is_wanted()

            # append the ratings.
            ratings = append_to(ratings, 0, joke_index, rating)

            print('coming up with a new joke....\n')

            # list everything for one user
            ratings[ratings['user'] == 0]

            start = time.time()
            # print('starting')

            # this will refit the algo with the new changes.
            recs = get_recs(ratings)

            print(recs[recs['user'] == 0])
            done = time.time()
            elapsed = done - start
            print('elapsed time -------------------------------------------------------{:.3f}\n\n'.format( elapsed))
            user_num = 0

            # the top joke was
            item_rec_1 = recs[recs['user'] == user_num].values[0][0]

            print('the top joke was \n\n{}\n '.format(get_joke(item_rec_1)))

    except KeyboardInterrupt:
        pass

    # # add the users to the rated moves.
    # for i in range(0, 3):
    #     item_rec_1 = recs[recs['user'] == user_num].values[i][0]
    #     score_rec_1 = recs[recs['user'] == user_num].values[i][1]
    #
    #     ratings = append_to(ratings, user_num, item_rec_1, score_rec_1)





    #
    # ## auto run.
    # # this is the number of time you want to get three recommendations.
    # for x in range(0, 5):
    #     # currently this is faking a user. The main user will be user 0.
    #     ratings = append_to(ratings, 0, random.randint(0, 150), random.randint(-10, 10))
    #
    #     # list everything for one user
    #     ratings[ratings['user'] == 0]
    #
    #     start = time.time()
    #     print('starting')
    #
    #     # this will refit the algo with the new changes.
    #     recs = get_recs(ratings)
    #
    #     print(recs[recs['user'] == 0])
    #     done = time.time()
    #     elapsed = done - start
    #     print('elapsed time -----------------------------------------------------------\n\n', elapsed)
    #     user_num = 0
    #
    #     # add the users to the rated moves.
    #     for i in range(0, 3):
    #
    #         item_rec_1 = recs[recs['user'] == user_num].values[i][0]
    #         score_rec_1 = recs[recs['user'] == user_num].values[i][1]
    #
    #         ratings = append_to(ratings, user_num, item_rec_1, score_rec_1)
    #

