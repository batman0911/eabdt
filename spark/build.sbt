lazy val root = (project in file(".")).
  settings(
    name := "spark",
    version := "1.0",
    mainClass in Compile := Some("com.linhnm.Main"),
    mainClass in assembly := Some("com.linhnm.Main")
  )

val sparkVersion = "3.2.2"

libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion % "provided"
libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
// https://mvnrepository.com/artifact/org.apache.spark/spark-sql-kafka-0-10
libraryDependencies += "org.apache.spark" %% "spark-sql-kafka-0-10" % sparkVersion


assemblyJarName in assembly := "eabdt.jar"

// META-INF discarding
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}