import pandas as pd
import random
import pickle
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit import topn
import lenskit as lk
import pandas as pd
import random
import time
from main_2 import *




def load_jokes():
    """
    Read in the file each time for now as it is pretty quick to load 150 lines.
    1) Read in all of the jokes from the source.
    2) we can
    :return: A pandas data frame.
    """
    # this is the joke file that is then pushed in to a pandas data frame.
    items_file = 'dataSets/jester_dataset_2/jester_items.dat'
    df = pd.read_csv(items_file, delimiter=':', names=['item', 'jokes'])
    return df


def get_joke(index):
    """
    get load all of the jokes then return the one that is selected.
    :param number:
    :return:
    """
    list_of = load_jokes()
    # index starts at 0 and the items start at 1
    return list_of.jokes[index - 1]


def get_ratings(file):
    """
    Ratings must have the form 'userId', 'movieId', 'rating'
    :param file: csv file to read in the ratings.
    :return: a data frame of the data from the file.
    """
    df = pd.read_csv(file, skiprows=1, names=['user', 'item', 'rating'])
    return df


def get_random_index_and_joke():
    """
    Get the index and a random joke.
    be between 0 -150
    :param x: this is the number of jokes that you want to return.
    :return: an array of tuples (index , joke).
    """
    rand_int = random.randint(0, 150)

    return rand_int,  get_joke(rand_int)


def refit_knn(nhood_size=20, sim_threshold=0):
    """
    This should not be run more than it is needed As it is takes a long time to fit the data.
    :param nhood_size:
    :param sim_threshold:
    :return: the class <class 'User_KNN.User_KNN'>
    """
    alg = User_KNN(nhood_size, sim_threshold)
    ratings = get_ratings()
    print("fitting ")
    alg.fit(ratings)
    print(alg)
    pickle_class(alg)
    print("fitted ")
    # return alg


def pickle_class(class_name):
    """
    https://stackoverflow.com/questions/2345151/how-to-save-read-class-wholly-in-python
    :param class_name:
    :return:
    """
    # example_dict = {1: "6", 2: "2", 3: "f"}

    pickle_out = open("classPickler.pickle", "wb")
    pickle.dump(class_name, pickle_out)
    pickle_out.close()

def de_pickle_class(file_name_string):
    pickle_in = open(file_name_string, "rb")
    de_pickled_class_name = pickle.load(pickle_in)

    print(de_pickled_class_name)
    print(type(de_pickled_class_name))
    return de_pickled_class_name


def store_fited_data():
    algo = refit_knn()

    algo = de_pickle_class()

    print("whooo it worked")
    #
    # print(algo.compute_similarity_cache())
    # print(algo.get_profile_length())
    # print(algo.get_rating())
    # algo.predict_for_user(1, [1, 2, 3, 4, 5, 6, 7, 8])


def append_to(df_og, user, item, rating):
    df_new = pd.DataFrame([[user, item, rating]], columns=['user', 'item', 'rating'])
    df_og = df_og.append(df_new)
    return df_og

def get_recs(data):

    # Define the algo to use. compare to the 10 nearest nabors
    algo_ii = knn.ItemItem(10)

    # fittable = Recommender.adapt(algo_ii)
    #
    # # fit the data this may take a second depending on the side of the data file.
    # fittable.fit(ratings)

    fittable = fitting(data , algo_ii)

    users = data.user.unique()

    # The number of non see itmes that have not been recommended yet. the number
    recs = batch.recommend(fittable, users, 3)

    return recs

def fitting(data, algo):
    fittable = Recommender.adapt(algo)
    return fittable.fit(data)


def check_rating(x):
    if x < -10 or x > 10:
        x = int(input("Nimber out of range please enter a number between -10 and 10."))
        return x
    else:
        return x


def ask_if_joke_is_wanted():
    joke = str(input("Would you like to hear a joke? "))
    # joke = 'yes'

    if joke == 'yes' or '\n':

        print("Ok I will tell you a joke.\n\n")

        joke_index , current_joke = get_random_index_and_joke()
        # get_joke( int(joke_index))

        print(current_joke)
        rating = int(input("on a scale from -10 - 10 how would you rate that joke? "))

        rating_updated = check_rating(rating)

        print("Thanks for giving joke: {}, a rating of {}.".format(joke_index, rating_updated))

        return joke_index, rating

    else:
        print("Ok good bye")
        quit()


