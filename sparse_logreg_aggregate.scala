// :load /home/ldali/wkspace/sparkscripts/sparse_logreg_aggregate.scala

import org.apache.spark.rdd.RDD


//val DATA_FILE = "/home/ldali/wkspace/slrn/airline.libsvm.2"
//val DATA_FILE = "/home/ldali/wkspace/slrn/data2.libsvm.2"
//val DATA_FILE = "/home/ldali/wkspace/slrn/training_binary.libsvm.gz"
val DATA_FILE = "/home/ldali/datasets/kdd2012/kdd12.val"
val LEARNING_RATE = 1.0
val LAMBDA_REG = 0.0   // L2 regularization
val MEMORY = 0.8  // momentum term (if zero, then regular gradient descent)
val N_ITERATIONS = 100
val N_PARTITIONS = 32


def loadData(filepath: String): RDD[(Double, Array[Int], Array[Double])] = {
	val lines = sc.textFile(filepath, N_PARTITIONS)
	lines.map(_.split(" "))
	  .map(cols => {
	  	val label = cols.head.toDouble
	  	val (indexes, values) = (cols.tail ++ Array("0:1")).map(col => {
	  		val i = col.indexOf(":")
	  		(col.slice(0, i).toInt, col.slice(i+1, col.length).toDouble)
	  	}).unzip
	  	(label, indexes, values)
	  })
}

val data = loadData(DATA_FILE).cache()

val nRows = data.count
val nDim = data.map { case (_, indexes, _) => indexes.max }.reduce(_ max _) + 1
val pAvg = data.map(_._1).reduce(_ + _) / nRows

def trainingIteration(weights: Array[Double], avgG: Array[Double], learningRate: Double): Unit = {
	val seqOp = (gloss: (Array[Double], Double), instance: (Double, Array[Int], Array[Double])) => {
		val (gradient, loss) = gloss
		val (label, indexes, values) = instance
		var dotProd = 0.0
		var k = 0
		while (k < indexes.length) {
			dotProd += values(k) * weights(indexes(k))
			k += 1
		}
		val pred = 1.0 / (1.0 + math.exp(-dotProd))
		k = 0
		while (k < indexes.length) {
			gradient(indexes(k)) += (pred - label) * values(k)
			k += 1
		}
		val lossElem = if (label == 0) {
				math.log1p(math.exp(-dotProd)) + dotProd
			} else {
				math.log1p(math.exp(-dotProd))
			}
		(gradient, loss + lossElem)
	}

	val combOp = (gloss1: (Array[Double], Double), gloss2: (Array[Double], Double)) => {
		val (gradient1, loss1) = gloss1
		val (gradient2, loss2) = gloss2
		((gradient1 zip gradient2).map{ case (g1, g2) => g1 + g2 }, loss1 + loss2)
	}

	val (g, loss) = data.treeAggregate((Array.ofDim[Double](nDim), 0.0))(seqOp, combOp)
	println(s"loss = ${loss/nRows}")

	var k = 0
	while (k < weights.length) {
		avgG(k) = avgG(k) * MEMORY + (g(k) + LAMBDA_REG * weights(k)) / nRows
		weights(k) = weights(k) - learningRate * avgG(k)
		k += 1
	}
}

val w = Array.ofDim[Double](nDim)
val avgGradient = Array.ofDim[Double](nDim)

w(0) = math.log(pAvg) - math.log(1 - pAvg)
println(w(0))

for (iteration <- 1 to N_ITERATIONS) {
	println(s"iteration $iteration")
	//trainingIteration(w, avgGradient, LEARNING_RATE / math.sqrt(iteration))
	trainingIteration(w, avgGradient, LEARNING_RATE)
}
