from __future__ import division
import numpy as np


def distribution_from_array(array):
	"""given an array returns dict of unique
	   values and distribution: [1,1,1,2,2,2] => {1:0.5, 2:0.5}"""

	values,counts = np.unique(array, return_counts=True)
	return {values[i]:float(counts[i])/sum(counts) for i in range(len(values))}


def gini_impurity(array):
	prob = distribution_from_array(array)
	p = np.array(prob.values())
	return 1 - sum(p*p)


def gini_gain(array, splits):
    # Average child gini impurity
    splits_impurity = sum([gini_impurity(split)*float(len(split))/len(array) for split in splits])
    return gini_impurity(array) - splits_impurity



def feature_splits(x, y):
	"""calculates all the possible feature splits using brute force"""
	splits = []
	feature = sorted(zip(x,y))
	for i in range(1, len(feature)):
		if feature[i-1][1] != feature[i][1]:
			avg = (feature[i-1][0] + feature[i][0])/2.0
			splits.append(avg)
	return splits


def split(X, Y, splitval):
    """utility function to split X,Y lists by value"""
    lesser = [y for x,y in zip(X, Y) if x < splitval]
    greater_equal = [y for x,y in zip(X, Y) if x >= splitval]
    return lesser, greater_equal
