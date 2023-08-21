ThisBuild / version := "1.0"

ThisBuild / scalaVersion := "2.13.10"

val sparkVersion = "3.2.2"

lazy val root = (project in file("."))
  .settings(
    name := "amazon",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "ml.dmlc" %% "xgboost4j" % "1.7.1",
      "ml.dmlc" %% "xgboost4j-spark" % "1.7.1"
    )
  )
