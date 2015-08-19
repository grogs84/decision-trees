import matplotlib.pyplot as plt


from data_sets import iris, gini_data
from functions import gini_impurity, gini_gain, feature_splits, split, distribution_from_array, np
from plotting import Graph
from trees import Stump, AdaBoost

def test_iris_split():
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

	print "-------IRIS DATA TEST -----------"

	print"\n"
	print "Best split value was: {} for a gini_gain of {}".format(best_split, max_gain)
	print "\n"
	l = distribution_from_array(best_l)
	r = distribution_from_array(best_r)
	print "left leaf count: {}".format(len(best_l))
	print best_l
	print l
	print "\n"
	print "right leaf count: {}".format(len(best_r))
	print best_r
	print r
	print "\n"

def test_gini_data_split():
	x = gini_data[0]
	y = gini_data[3]

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

	print "-------GINI DATA TEST -----------"

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


def test_plot_iris():
	x = iris.data[:,:2]
	y = iris.target

	u = np.unique(y)
	c = ('blue', 'red', 'yellow')

	t = {}
	for k,v in zip(u,c):
		t[k]=v



	plt.figure(2, figsize=(8,8))

	for pt, color in zip(x,y):
		plt.scatter(pt[0], pt[1], c=t[color])

	plt.plot([5.4,5.4], [2, 4.5], 'k-')

	plt.show()

	f = raw_input("")



def test_adaboost():
	X = gini_data[0:3]
	y = gini_data[3]

	ab = AdaBoost(X,y)

	ab.train()

	print len(ab.learners)


if __name__ == "__main__":
	# test_iris_split()
	# test_gini_data_split()
	# test_plot_iris()
	test_adaboost()

