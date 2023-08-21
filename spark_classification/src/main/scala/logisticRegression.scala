import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

object logisticRegression {
  def main(args: Array[String]): Unit = {
    // Create a Spark session
    val spark = SparkSession.builder()
      .appName("logisticRegression")
      .master("local[*]")
      .config("executor.memory", "4g")
      .getOrCreate()

    // Load train and test data
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

    // Create a LogisticRegression model
    val lr = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")

    // Create a pipeline
    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    // Train the model
    val model = pipeline.fit(trainDF)

    // Make predictions on test data
    val predictions = model.transform(testDF)

    // Evaluate the model
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("rawPrediction")

    val accuracy = evaluator.setMetricName("areaUnderROC").evaluate(predictions)
    val auc = evaluator.setMetricName("areaUnderPR").evaluate(predictions)

    // Print accuracy and AUC
    println(s"Accuracy Logistic Regression: $accuracy")
    println(s"AUC Logistic Regression: $auc")

    // Stop the Spark session
    spark.stop()
  }
}
