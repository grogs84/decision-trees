from data_sets import iris, gini_data
from functions import gini_impurity, gini_gain, feature_splits, split


if __name__ == "__main__":

	x = iris.data[:,0]
	y = iris.target

	split_vals = feature_splits(x, y)

	max_gain = -1
	best_split = None

	for split_val in split_vals:
		splits = split(x, y, split_val)
		gain = gini_gain(y, splits)
		if gain > max_gain:
			max_gain = gain
			best_split = split_val
			best_l = splits[0]
			best_r = splits[1]

	print "Best split value was: {} for a gini_gain of {}".format(best_split, max_gain)
	print "\n"
	print best_l
	print best_r