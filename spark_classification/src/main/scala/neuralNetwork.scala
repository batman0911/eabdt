import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object neuralNetwork {
  def main(args: Array[String]): Unit = {
    // Create a Spark session
    val spark = SparkSession.builder()
      .appName("neuralNetwork")
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

    // Create a MultilayerPerceptronClassifier
    val layers = Array[Int](384, 64, 2)  // Input, hidden, and output layer sizes
    val mlp = new MultilayerPerceptronClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setTol(1e-4)
      .setMaxIter(100)

    // Create a pipeline
    val pipeline = new Pipeline().setStages(Array(assembler, mlp))

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
    println(s"Accuracy neural network: $accuracy")
    println(s"AUC neural network: $auc")

    // Stop the Spark session
    spark.stop()
  }
}
