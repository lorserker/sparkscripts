// :load /home/ldali/wkspace/sparkscripts/sparse_logreg.scala

import org.apache.spark.rdd.RDD
import org.apache.spark.HashPartitioner


//val DATA_FILE = "/home/ldali/wkspace/slrn/data2.libsvm"
//val DATA_FILE = "/home/ldali/wkspace/slrn/training_binary.libsvm.gz"
val DATA_FILE = "/home/ldali/wkspace/slrn/airline.libsvm.2"
val LEARNING_RATE = 1.0
val LAMBDA_REG = 0.1   // L2 regularization
val MEMORY = 0.9  // momentum term (if zero, then regular gradient descent)
val N_ITERATIONS = 10
val N_PARTITIONS = 32

val partitioner = new HashPartitioner(N_PARTITIONS)

def loadLibSVM(filepath: String): (RDD[(Long, Double)], RDD[(Long, Seq[(Int, Double)])]) = {
  val lines = sc.textFile(filepath)
  val colLines: RDD[(Array[String], Long)] = lines.map(_.split(" ")).zipWithIndex
  val y: RDD[(Long, Double)] = colLines.map{ case (cols, i) => (i, cols.head.toDouble) }.partitionBy(partitioner).persist()
  val X: RDD[(Long, Seq[(Int, Double)])] = 
	colLines.map{ case (cols, i) => (i, cols.tail.map(_.split(":")).map(parts => (parts(0).toInt, parts(1).toDouble)).toSeq) }.partitionBy(partitioner).persist()

  (y, X)
}

val (y, matrix) = loadLibSVM(DATA_FILE)

val nRows: Long = y.map(_._1).max + 1

val nDim: Int = matrix.map{ case (_, kvals) => kvals.map(_._1).max }.max + 1
//val nDim = 579851


// initialize weights to all zeros
val w: Array[Double] = new Array[Double](nDim)
val g: Array[Double] = new Array[Double](nDim)


def log2(x: Double): Double = math.log10(x)/math.log10(2.0)


def normalizedEntropy(actual: RDD[(Long, Double)], predicted: RDD[(Long, Double)]): Double = {
  val avgP = actual.values.reduce(_ + _) / nRows
  val entropy = -(avgP * log2(avgP) + (1- avgP)*log2(1 - avgP))

  val itemLogLoss = 
  	(y: Double, p:Double) => {
  	  if (y - 0.0 < 1e-6) {
	  	-(1 - y) * log2(1-p)
	  } else if (1 - y < 1e-6) {
	  	-y * log2(p)
	  }	else {
	  	-(1 - y) * log2(1-p) - y * log2(p)
	  }
  	}

  val logLoss = 
    actual.join(predicted).mapValues{ case (y, p) => itemLogLoss(y, p) }.values.reduce(_ + _) / nRows

  //return logLoss / entropy
  logLoss
}


def trainingIteration(weights: Array[Double], avgGradients: Array[Double], learningRate: Double): Unit = {

  val dotProds = matrix.mapValues(jvals => jvals.map{ case (j, x) => x * weights(j) }.sum)
  //println(" ===== dotProds =====")
  //println(dotProds.toDebugString)
  
  val predictions = dotProds.mapValues(v => 1.0 / (1.0 + math.exp(-v)))

  //println(" ===== predictions =====")
  //println(predictions.toDebugString)

  println(s"NE = ${normalizedEntropy(y, predictions)}")

  val deltas = predictions.join(y).mapValues{ case (predicted, correct) => (predicted - correct)/nRows }


  //println(" ===== deltas =====")
  //println(deltas.toDebugString)

  val updates = matrix.join(deltas).flatMap{ case (i, (jvals, d)) => jvals.map{ case (j, x) => (j, x * d) } }.reduceByKey(_ + _)

  //println(" ===== updates =====")
  //println(updates.toDebugString)

  val uMap: Map[Int, Double] = updates.collect.toMap

  //val newWeights = new Array[Double](nDim)
  var j = 0
  while (j < nDim) {
  	//val l2regTerm = learningRate*LAMBDA_REG*weights(j)/nRows
  	val gradient = uMap.getOrElse[Double](j, 0) + LAMBDA_REG*weights(j)/nRows
  	avgGradients(j) = avgGradients(j)*MEMORY + gradient
  	weights(j) = weights(j) - learningRate*avgGradients(j)
  	j += 1
  }

}

for (iteration <- 1 to N_ITERATIONS) {
	println(s"iteration $iteration")
	trainingIteration(w, g, LEARNING_RATE / math.sqrt(iteration))
	//trainingIteration(w, LEARNING_RATE)
}
