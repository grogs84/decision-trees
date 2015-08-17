
import unittest
import numpy as np

from decision_trees.project.data_sets import iris, gini_data
from decision_trees.project.trees import AdaBoost, Stump



class TestAdaBoost(unittest.TestCase):

	def test_features_lables_are_np_arrays(self):
		X = iris.data
		y = iris.target

		boost = AdaBoost(X,y)
		self.assertEqual(X.rows, 'np.ndarray')
		self.assertEqual(y.type, 'np.ndarray')




if __name__ == "__main__":
	unittest.main()

