
import numpy as np
from sklearn.cross_validation import train_test_split
from scipy.stats import mode
class AdaBoost(object):

	def __init__(self, features, lables):
		self.features = np.array(features)
		self.lables = np.array(lables)
		self.rows = len(self.features)
		self.feature_count = len(self.features[0])
		self.weights = np.ones(self.rows)/self.rows
		self.alpha = []
		self.learners = []

		x_train, x_test, y_train, y_test = train_test_split(self.features, self.lables, test_size=0.33)

		self.x_train = x_train
		self.y_train = y_train
		self.x_test  = x_test
		self.y_test  = y_test
		

	def train(self):
		 # evaluate 50 stumps
		for i  in range(50):
			min_error = np.inf
			for feature in range(feature_count):
				stump = Stump(self.x_train[:, feature], self.y_train)
				if stump.error < min_error:
					min_error = stump.error
					winner = stump
			self.learners.append(stump)









class Stump(object):
	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.lt_predict = None
		self.gte_predict = None
		self.best_split = None
		self.error = None
		self.alpha = 0


		self.start(x,y)

	def start(self,x,y):
		split, lt_elements, gte_elements = self.find_best_split(x,y)
		self.best_split = split

		self.lt_predict = mode(lt_elements)[0][0]
		self.gte_predict = mode(gte_elements)[0][0]


	def predict(self, x):
		if x < self.best_split:
			return self.lt_predict
		else:
			return self.gte_predict

	def find_best_split(self,x,y):
		split_vals = self.feature_splits(x, y)

		max_gain = -1
		best_split = None

		for split_val in split_vals:
			splits = self.split(x, y, split_val)
			gain = self.gini_gain(y, splits)
			if gain > max_gain:
				max_gain = gain
				best_split = split_val
				lt_split = np.array(splits[0])
				gte_split = np.array(splits[1])

		return best_split, lt_split, gte_split


	def distribution_from_array(self, array):
		"""given an array returns dict of unique
		   values and distribution: [1,1,1,2,2,2] => {1:0.5, 2:0.5}"""

		values,counts = np.unique(array, return_counts=True)
		return {values[i]:float(counts[i])/sum(counts) for i in range(len(values))}


	def gini_impurity(self, array):
		prob = self.distribution_from_array(array)
		p = np.array(prob.values())
		return 1 - sum(p*p)


	def gini_gain(self, array, splits):
	    # Average child gini impurity
	    splits_impurity = sum([self.gini_impurity(split)*float(len(split))/len(array) for split in splits])
	    return self.gini_impurity(array) - splits_impurity


	def feature_splits(self, x, y):
		"""calculates all the possible feature splits using brute force"""
		splits = []
		feature = sorted(zip(x,y))
		for i in range(1, len(feature)):
			if feature[i-1][1] != feature[i][1]:
				avg = (feature[i-1][0] + feature[i][0])/2.0
				splits.append(avg)
		return splits


	def split(self, X, Y, splitval):
	    """utility function to split X,Y lists by value"""
	    lesser = [y for x,y in zip(X, Y) if x < splitval]
	    greater_equal = [y for x,y in zip(X, Y) if x >= splitval]
	    return lesser, greater_equal





