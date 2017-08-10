// :load /home/ldali/wkspace/sparkscripts/mllib_logreg_sparse.scala

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.classification.LogisticRegression

//val DATA_FILE = "/home/ldali/wkspace/slrn/data2.libsvm.2"
val DATA_FILE = "/home/ldali/wkspace/slrn/airline.libsvm.2"
val LAMBDA_REG = 0.1   // L2 regularization
val N_ITERATIONS = 10

def loadLibSVM(filepath: String): (Int, RDD[(Double, Array[Int], Array[Double])]) = {
	val lines = sc.textFile(filepath)
	
	val data = lines.map(line => {
	  val items = line.split(' ')
      val label = items.head.toDouble
      val (indices, values) = items.tail.filter(_.nonEmpty).map { item =>
	      val indexAndValue = item.split(':')
	      val index = indexAndValue(0).toInt - 1 // Convert 1-based indices to 0-based.
	      val value = indexAndValue(1).toDouble
	      (index, value)
	  }.unzip

	  (label, indices, values)
	})

	val nFeatures = data.map{ case (_, indices, _) => indices.lastOption.getOrElse(0) }.reduce(math.max) + 1

	(nFeatures, data)
}


case class LabeledPointML(label: Double, features: org.apache.spark.ml.linalg.Vector)


val (nDim, rawData) = loadLibSVM(DATA_FILE)

val labeledPoints = rawData.map{ case (label, indices, values) => LabeledPoint(label, Vectors.sparse(nDim, indices, values)) }.cache()

val dataFrame = spark.createDataFrame(labeledPoints.map(lp => LabeledPointML(lp.label, lp.features.asML)))

// val logreg = new LogisticRegression
// logreg.setElasticNetParam(0.0)
// logreg.setMaxIter(N_ITERATIONS)

// val model = logreg.fit(dataFrame)

val logreg = new LogisticRegressionWithSGD().setIntercept(true)
logreg.optimizer.setRegParam(LAMBDA_REG).setNumIterations(N_ITERATIONS)

//println(labeledPoints.count)

val model = logreg.run(labeledPoints)

println(model.intercept)
