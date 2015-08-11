from data_sets import iris, gini_data
from functions import gini_impurity, gini_gain, feature_splits, split, distribution_from_array, np


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
			best_l = np.array(splits[0])
			best_r = np.array(splits[1])

	print"\n"
	print "Best split value was: {} for a gini_gain of {}".format(best_split, max_gain)
	print "\n"
	l = distribution_from_array(best_l)
	r = distribution_from_array(best_r)
	print "left leaf count: {}".format(len(best_l))
	print l
	print "\n"
	print "right leaf count: {}".format(len(best_r))
	print r
	print "\n"