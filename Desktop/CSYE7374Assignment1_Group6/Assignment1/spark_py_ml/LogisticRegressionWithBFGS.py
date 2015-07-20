from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from numpy import array
from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(';')]
    return LabeledPoint(values[11], values[0:10])


sc = SparkContext("local", "Simple App")
data = sc.textFile("../winequality.csv")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))