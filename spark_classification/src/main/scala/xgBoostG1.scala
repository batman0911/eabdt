
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object xgBoostG1 {
  def main(args: Array[String]): Unit = {
    // Create a Spark session
    val spark = SparkSession.builder()
      .appName("xgBoostG1")
      .master("local[*]")
      .config("executor.memory", "4g")
      .getOrCreate()

    // Convert label column to Double type
    val trainData = spark.read.options(Map("inferSchema" -> "true", "delimiter" -> """;""", "header" -> "true")).csv("data/train.csv")
    val testData = spark.read.options(Map("inferSchema" -> "true", "delimiter" -> """;""", "header" -> "true")).csv("data/test.csv")

    // Convert label column to Double type
    val trainDF = trainData.withColumn("label", trainData("label").cast("double"))
    val testDF = testData.withColumn("label", testData("label").cast("double"))

    // Assemble features using VectorAssembler
    val featureCols = (0 to 383).map(_.toString)
    val assembler = new VectorAssembler()
      .setInputCols(featureCols.toArray)
      .setOutputCol("features")

    // Create an XGBoostEstimator
    val xgbParam = Map("eta" -> 0.1f,
      "max_depth" -> 5,
      "objective" -> "multi:softprob",
      "num_class" -> 2,
      "num_round" -> 100)
    val xgbClassifier = new XGBoostClassifier(xgbParam).
      setFeaturesCol("features").
      setLabelCol("label")


    // Create a pipeline
    val pipeline = new Pipeline().setStages(Array(assembler, xgbClassifier))

    // Train the model
    val model = pipeline.fit(trainDF)

    // Make predictions on test data
    val predictions = model.transform(testDF)

    // Evaluate the model
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val auc = evaluator.setMetricName("areaUnderPR").evaluate(predictions)

    // Print accuracy and AUC
    println(s"Accuracy XGBoost: $accuracy")
    println(s"AUC XGBoost: $auc")

    // Stop the Spark session
    spark.stop()
  }
}

