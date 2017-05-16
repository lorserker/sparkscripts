// :load /home/ldali/blackboard/distrw_logreg/src/sparse_distr_logreg.scala

import org.apache.spark.rdd.RDD

val DATA_FILE = "/home/ldali/wkspace/slrn/data2.libsvm"
val LEARNING_RATE = 0.1


def loadLibSVM(filepath: String): (RDD[(Long, Double)], RDD[(Long, Long, Double)]) = {
  val lines = sc.textFile(filepath)
  val colLines: RDD[(Array[String], Long)] = lines.map(_.split(" ")).zipWithIndex
  val y: RDD[(Long, Double)] = colLines.map{ case (cols, i) => (i, cols.head.toDouble) }.cache()
  val X: RDD[(Long, Long, Double)] = 
	colLines.flatMap{ case (cols, i) => cols.tail.map(_.split(":")).map(parts => (i, parts(0).toLong, parts(1).toDouble)) }

  (y, X)
}

val (y, matrix) = loadLibSVM(DATA_FILE)

val nRows = y.map(_._1).max
val nDim = matrix.map(_._2).max

// initialize weights to all zeros
val w: RDD[(Long, Double)] = sc.parallelize((0 until nDim.toInt).map(j => (j.toLong, 0.0)))

// we need these two representations for multiplications
val iMatrix = matrix.map{ case (i, j, x) => (i, (j, x)) }.cache()
val jMatrix = matrix.map{ case (i, j, x) => (j, (i, x)) }.cache()


def trainingIteration(weights: RDD[(Long, Double)]): RDD[(Long, Double)] = {

  val dotProds = jMatrix.join(weights).map{ case (j, ((i, x), w)) => (i, x * w) }.reduceByKey(_ + _).cache()
  val predictions = dotProds.map{ case (i, v) => (i, 1.0 / (1.0 + math.exp(-v))) }

  val deltas = predictions.join(y).map{ case (i, (predicted, correct)) => (i, LEARNING_RATE*(predicted - correct)) }

  val updates = iMatrix.join(deltas).map{ case (i, ((j, x), d)) => (j, x * d) }.reduceByKey(_ + _).mapValues(_ / nRows)

  val newWeights = 
    weights.leftOuterJoin(updates).map{ case (j, (oldW, optU)) => (j, oldW - optU.getOrElse(0.0)) }

  newWeights
}
