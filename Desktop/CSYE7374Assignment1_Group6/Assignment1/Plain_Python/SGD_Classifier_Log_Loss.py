import numpy as np
from sklearn.linear_model import SGDClassifier
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

clf = SGDClassifier(loss="log", penalty="l2", alpha=0.001, n_iter=100)
clf.fit (feature_train,feature_lable)
pred = clf.predict (test_feature)
from sklearn.metrics import mean_absolute_error
print 'mean_absolute_error:'
print mean_absolute_error(test_lable, pred)

print 'score:'
print clf.score (test_feature, test_lable)

print clf.coef_
print clf.intercept_

print '------------------------'
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(feature_train)  # Don't cheat - fit only on training data
feature_train = scaler.transform(feature_train)
clf1 = SGDClassifier(loss="log", penalty="l2", alpha=0.001, n_iter=100)
clf1.fit (feature_train,feature_lable)
test_feature = scaler.transform(test_feature)
pred = clf1.predict (test_feature)
from sklearn.metrics import mean_absolute_error
print 'mean_absolute_error:'
print mean_absolute_error(test_lable, pred)
