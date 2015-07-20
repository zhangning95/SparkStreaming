/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD
object SimpleApp {
  def main(args: Array[String]) {
    val csvPath = "/Users/ningzhang/Desktop/Assignment1/winequality-white.csv" // Should be some file on your system
    val conf = new SparkConf().setAppName("Simple Application")
    val sc = new SparkContext(conf)
    val csv = sc.textFile(csvPath)
    val headerAndRows= csv.map(line=>line.split(";").map(_.trim))
    val header = headerAndRows.first
    val data = headerAndRows.filter(_(0) != header(0))
    val parsedData = data.map{line => LabeledPoint(line(11).toDouble, Vectors.dense(line(0).toDouble,line(1).toDouble))}
    parsedData.cache()
    val numIterations = 20
    val model1 = RidgeRegressionWithSGD.train(parsedData, numIterations)
    val valuesAndPreds1 = parsedData.map { point =>
           val prediction = model1.predict(point.features)
            (point.label, prediction)}
     val MSE1 = valuesAndPreds1.map{ case(v, p) => math.pow((v - p), 2)}.reduce(_ + _)/valuesAndPreds1.count
     println("RidgeRegressionWithSGD training Mean Squared Error = " + MSE1)
     println("---------------------------------")
  }
}