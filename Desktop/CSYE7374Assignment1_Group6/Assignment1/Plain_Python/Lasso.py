import numpy as np
def loadDataSet():
	feature_train = []
	feature_lable = []
	test_feature = []
	test_lable = []
	fr = open ('winequality-white.csv')
	count = 0
	for line in fr.readlines():
		if count == 0:
			header = line.strip().split(';')
		elif count > 0 and count < 4000:
			lineAttr = line.strip().split(';')
			feature_train.append([float(lineAttr[0]),float(lineAttr[1]),float(lineAttr[2]),float(lineAttr[3]),
				float(lineAttr[4]),float(lineAttr[5]),float(lineAttr[6]),float(lineAttr[7]),
				float(lineAttr[8]),float(lineAttr[9]),float(lineAttr[10])])
			feature_lable.append(int(lineAttr[11]))
		else:
			lineAttr = line.strip().split(';')
			test_feature.append([float(lineAttr[0]),float(lineAttr[1]),float(lineAttr[2]),float(lineAttr[3]),
				float(lineAttr[4]),float(lineAttr[5]),float(lineAttr[6]),float(lineAttr[7]),
				float(lineAttr[8]),float(lineAttr[9]),float(lineAttr[10])])
			test_lable.append(int(lineAttr[11]))
		count += 1
	return np.array(feature_train), np.array(feature_lable), np.array(test_feature), np.array(test_lable)

feature_train, feature_lable, test_feature,test_lable = loadDataSet()

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit (feature_train,feature_lable)
pred = clf.predict (test_feature)
from sklearn.metrics import mean_absolute_error
print 'mean_absolute_error:'
print mean_absolute_error(test_lable, pred)

print 'score:'
print clf.score (test_feature, test_lable)

print clf.coef_
print clf.intercept_
