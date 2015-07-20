from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from numpy import array
from pyspark import SparkContext
# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(';')]
    return LabeledPoint(values[11], values[0:10])


sc = SparkContext("local", "Simple App")
data = sc.textFile("../winequality.csv")
parsedData = data.map(parsePoint)

# Build the model
model = LinearRegressionWithSGD.train(parsedData)

# Evaluate the model on training data
valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds.map(lambda (v, p): (v - p)**2).reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))