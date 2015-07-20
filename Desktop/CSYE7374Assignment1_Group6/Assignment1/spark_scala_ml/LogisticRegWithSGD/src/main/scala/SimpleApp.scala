/* SimpleApp.scala */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
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
    // Run training algorithm to build the model
        val numIterations = 20
        val model = LogisticRegressionWithSGD.train(parsedData, numIterations)

        // Evaluate model on training examples and compute training error
        val labelAndPreds = parsedData.map { point =>
          val prediction = model.predict(point.features)
          (point.label, prediction)
        }
        val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / parsedData.count
        val n1 = labelAndPreds.filter(r => (r._1==1)&&(r._2==1)).count.toDouble
        val n2 = labelAndPreds.filter(r => (r._1==0)&&(r._2==0)).count.toDouble
        val d1 = labelAndPreds.filter(r => (r._1==1)).count.toDouble
        val d2 = labelAndPreds.filter(r => (r._1==0)).count.toDouble
        val sensitivity = n1/d1
        val specificity = n2/d2

        println("\nTraining Error = " + trainErr)
        println("Sensitivity = " + sensitivity)
        println("Specificity = " + specificity)
        println();
  }
}