/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
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
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)
    val numIterations = 100
    val model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(training)
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
     val prediction = model.predict(features)
           (prediction, label)}
    

    // Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val precision = metrics.precision
println("Precision = " + precision)
  }
}