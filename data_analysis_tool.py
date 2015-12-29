# Ken Ling

# Decision Tree Classifer
# Input: X - list of features, y - list of categories
# e.g: X = [(a1, b1), (a2, b2)], y = [y1, y2] (y should be integers)
# Output: a decision tree classifier

# Usage:
# if X_test is a list of features from test set, e.g: X_test = [(a1', b1'), (a2', b2')]
# y_test = clf.predict(X_test)
# y_test is the predicted result of X_test from clf
from sklearn import tree
def get_decision_tree_classifier(X, y):
	# parameters reference: http://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
	clf = tree.DecisionTreeClassifier(max_depth=None, criterion='entropy', min_samples_leaf=1, min_samples_split=2)
	clf.fit(X, y)
	return clf

# Relative Information Gain
# Input: two list of variables
# Output: relative information gain
# Notice: Data should be binned first
from sklearn import metrics
def compute_relative_information_gain(X, Y):
	H_Y = metrics.mutual_info_score(Y, Y)
	info_gain = metrics.mutual_info_score(Y, X)
	relative_info_gain = info_gain / H_Y
	return relative_info_gain

# Kendall Correlation
# Input: two list of variables
# Output: Kendall Correlation, and p_value
# Notice: Data should be binned first
from scipy import stats
def compute_kendall_correlation(X, Y):
	kendall_correlation, p_value = stats.kendalltau(X, Y)
	return kendall_correlation, p_value

