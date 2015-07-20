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
	return feature_train, feature_lable, test_feature, test_lable

feature_train, feature_lable, test_feature,test_lable = loadDataSet()

#sklearn.linear_model.LinearRegression
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_absolute_error
clf = linear_model.LinearRegression()
clf.fit(feature_train, feature_lable)
y_pred = clf.predict (test_feature)
y_true =  np.array(test_lable)
print 'The accuracy for linear_model is: '
accuracy = mean_absolute_error(y_true, y_pred)
print accuracy
