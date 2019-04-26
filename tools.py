import pandas as pd
import random
from User_KNN import User_KNN
import pickle 


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


def get_ratings(file='jester_ratings_300.csv'):
    """
    Ratings must have the form 'userId', 'movieId', 'rating'
    :param file: csv file to read in the ratings.
    :return: a data frame of the data from the file.
    """
    df = pd.read_csv(file, sep=',', header=0, names=['userId', 'movieId', 'rating'])
    return df


def get_random_index_joke(num_of_jokes):
    """
    Get the index and a random joke.
    be between 0 -150
    :param x: this is the number of jokes that you want to return.
    :return: an array of tuples (index , joke).
    """
    index_joke_array = []
    for x in range(0, num_of_jokes):
        rand_int = random.randint(0, 150)

        index_joke_array.append((rand_int, get_joke(rand_int)))
    return index_joke_array


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

def de_pickle_class():
    pickle_in = open("classPickler.pickle", "rb")
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


store_fited_data()

algo = de_pickle_class()

print(algo.predict_for_user(1, [1, 2, 3, 4, 5, 6, 7, 8]))

