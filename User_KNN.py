# Homnework 2
# INFO 4871/5871, Spring 2019
# Aaron Barge
# University of Colorado, Boulder

from typing import Dict
import pandas as pd
import numpy as np
import logging
from heapq import nlargest

_logger = logging.getLogger(__name__)

class User_KNN:
    """
    User-user nearest-neighbor collaborative filtering with ratings. Not a very efficient implementation
    using data frames and tables instead of numpy arrays, which would be _much_ faster.

    Attributes:
        _ratings (pandas.DataFrame): Ratings with user, item, ratings
        _sim_cache (Dict of Dicts): a multi-level dictionary with user/user similarities pre-calculated
        _profile_means (Dict of float): a dictionary of user mean ratings
        _profile_lenghts (Dict of float): a dictionary of user profile vector lengths
        _item_means (Dict of float): a dictionary of item mean ratings
        _nhood_size (int): number of peers in each prediction
        _sim_threshold (float): minimum similarity for a neighbor
    """
    _ratings = None
    _sim_cache: Dict[int, Dict] = {}
    _profile_means: Dict[int, float] = {}
    _profile_lengths: Dict[int, float] = {}
    _item_means: Dict[int, float] = {}
    _nhood_size = 1
    _sim_threshold = 0

    def __init__(self, nhood_size, sim_threshold=0):
        """
        Args:
        :param nhood_size: number of peers in each prediction
        :param sim_threshold: minimum similarity for a neighbor
        """
        self._nhood_size = nhood_size
        self._sim_threshold = sim_threshold

    def get_users(self): return list(self._ratings.index.levels[0])

    def get_items(self): return list(self._ratings.index.levels[1])

    def get_profile(self, u): return self._ratings.loc[u]

    def get_profile_length(self, u): return self._profile_lengths[u]

    def get_profile_mean(self, u): return self._profile_means[u]

    def get_item_mean(self, u): return self._item_means[u]

    def get_similarities(self, u): return self._sim_cache[u]

    def get_rating(self, u, i):
        """
        Args:
        :param u: user
        :param i: item
        :return: user's rating for item or None
        Issues a warning if the user has more than one rating for the same item. This indicates a problem
        with the data.
        """
        if (u,i) in self._ratings.index:
            maybe_rating = self._ratings.loc[u,i]
            if len(maybe_rating) == 1:
                return float(maybe_rating.iloc[0])
            else:  # More than one rating for the same item, shouldn't happen
                _logger.warning('Multiple ratings for an item - User %d Item %d', u, i)
                return None
        else: # User, item pair doesn't exist in index
            return None

    def compute_profile_length(self, u):
        """
        Computes the geometric length of a user's profile vector.
        :param u: user
        :return: length
        """
        arr = self.get_profile(u).rating.values
        squares = [x**2 for x in arr]
        sum_of_squares = sum(squares)
        ans = np.sqrt(sum_of_squares)
        return ans

    def compute_profile_lengths(self):
        """
        Computes the profile length table `_profile_lengths`
        :return: None
        """
        for xx in self.get_users():
            ln = self.compute_profile_length(xx)
            self._profile_lengths[xx] = ln


    def compute_profile_means(self):
        """
        Computes the user mean rating table `_user_means`
        :return: None
        """
        for uu in self.get_users():

            arr = self.get_profile(uu).rating.values
            mean = sum(arr)/len(arr)
            self._profile_means[uu] = mean


    def compute_item_means(self):
        """
        Computes the item means table `_item_means`
        :return: None
        """
        # todo i can't seem to test this i hope it works.
        df = pd.DataFrame()
        for x in self.get_users():
            acc = self.get_profile(x).T
            df= df.append(acc)

        arrMeans = df.mean()


        for xx in range(1, len(arrMeans)+1):
            self._item_means[xx] =arrMeans[xx]


    # this is a 2d table. look the lenskit code?
    #so the syntax for adding to a dictionary is dict[key] = value
    def compute_similarity_cache(self):
        """
        Computes the similarity cache table `_sim_cache`
        :return: None
        """
        ll = self.get_users()


        for u in ll:
            self._sim_cache[u] = {}

            for v in ll:
                self._sim_cache[u][v] = self.cosine(u, v)

                if u == v:
                    self._sim_cache[u][v] = 0


    def get_overlap(self, u, v):
        """
        Computes the items in common between profiles. Hint: use set operations
        :param u: user1
        :param v: user2
        :return: set intersection
        """
        A = self.get_profile(u)
        B = self.get_profile(v)

        result = pd.concat([A, B], axis=1, join='inner')

        return result

    def cosine(self, u, v):
        """
        Computes the cosine between u and v vectors
        :param u:
        :param v:
        :return: cosine value
        """

        overlap = self.get_overlap(u, v)

        accArr =[]
        aaArr =[]
        bbArr=[]

        for movieId in range(0, len(overlap)):
            # This will acess the two values that are bing compared.
            A = overlap.values[movieId][0:2][0]
            B = overlap.values[movieId][0:2][1]

            accArr.append(A * B )
            aaArr.append(A **2)
            bbArr.append(B **2)

        # you do want to use the geometic lenght

        AA = self.get_profile_length(u)
        BB = self.get_profile_length(v)

        # ans = sum(accArr) /  ((sum(aaArr)**.5)  * ((sum(bbArr)**.5) ))
        ans = sum(accArr) /  ( AA * BB )

            
        return ans

    def fit(self, ratings):
        """
        Trains the model by computing the various cached elements. Technically, there isn't any training
        for a memory-based model.
        :param ratings:
        :return: None
        """
        self._ratings = ratings.set_index(['userId', 'movieId'])
        self.compute_profile_lengths()
        self.compute_profile_means()
        self.compute_similarity_cache()

    def neighbors(self, u, i):
        """
        Computes the user dictN
        :param u: user
        :param i: item
        :return: user and item !!!!!!!!!!!!!!!!!!!
        """
        dictionary = {}

        for v in self.get_users(): # get all the user values
            currentRating = self.get_rating(v, i) # get the users values

            # currentRating has large than .5 and not == 0.
            # if currentRating != 0 and currentRating != None and self._sim_cache[u][v] > self._sim_threshold :
            if currentRating != 0 and currentRating != None and self._sim_cache[u][v] > self._sim_threshold :

                dictionary[v] = self._sim_cache[u][v] # add

        # return the _nhood_size of values.
        topneighnors = nlargest(self._nhood_size , dictionary, key=dictionary.get)


        return topneighnors


    def predict(self, u, i):
        """
        Predicts the rating of user for item
        :param u: user
        :param i: item
        :return: predicted rating
        """

        top = 0
        bottom = 0

        for xx in self.neighbors(u, i):

            top = top + (self._sim_cache[u][xx] * (self.get_rating(xx, i) - self._profile_means[xx]))

            bottom = bottom + self._sim_cache[u][xx]

        # don't davide by zero
        if bottom == 0:
            return self._profile_means[u]

        predicted = self._profile_means[u] + (top / bottom)

        return predicted


    def predict_for_user(self, user, items, ratings=None):
        """
        Predicts the ratings for a list of items. This is used to calculate ranked lists.
        Note that the `ratings` parameters is required by the LKPy interface, but is not
        used by this algorithm.
        :param user:
        :param items:
        :param ratings:
        :return (pandas.Series): A score Series indexed by item.
        """
        scores = [self.predict(user, i) for i in items]

        return pd.Series(scores, index=pd.Index(items))





