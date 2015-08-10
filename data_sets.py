from sklearn import datasets
import numpy as np

iris = datasets.load_iris()

gini_data = np.array([[1,1,1,0,0,0,0,1,0],
					  [1,1,0,0,1,1,0,0,1],
					  [1.0,6.0,5.0,4.0,7.0,3.0,8.0,7.0,5.0],
					  [1,1,0,1,0,0,0,1,0]])