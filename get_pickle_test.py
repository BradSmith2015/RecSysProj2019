
import pickle
from tools import *


alg_file = "classPickler.pickle"

alg_two_file = "alg_two.pickle"


# pickle_in = open(alg_file, "rb")
# de_pickled_class_name = pickle.load(pickle_in)


aa = de_pickle_class(alg_file)

bb = de_pickle_class(alg_two_file)

print(aa)
print(type(aa))


print(bb)
print(type(bb))