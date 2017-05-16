// :load /home/ldali/blackboard/distrw_logreg/src/spark_multiply.scala

import org.apache.spark.rdd.RDD

//val MATRIX_FILE = "/home/ldali/blackboard/distrw_logreg/data/matrix_small_2.tsv"
val MATRIX_FILE = "/home/ldali/blackboard/distrw_logreg/data/matrix.tsv"
//val W_FILE = "/home/ldali/blackboard/distrw_logreg/data/w_vals.tsv"
val W_FILE = "/home/ldali/blackboard/distrw_logreg/data/w1m_vals.tsv"

//val RESULT_FILE = "/home/ldali/blackboard/distrw_logreg/data/result"
val RESULT_FILE = "/home/ldali/blackboard/distrw_logreg/data/result_big"

val DELTAS_FILE = "/home/ldali/blackboard/distrw_logreg/data/deltas.tsv"

//val UPDATES_FILE = "/home/ldali/blackboard/distrw_logreg/data/updates"
val UPDATES_FILE = "/home/ldali/blackboard/distrw_logreg/data/updates_big"

val matrix = sc.textFile(MATRIX_FILE).map(_.split("\t")).map(cols => (cols(0).toInt, cols(1).toInt, cols(2).toDouble)).cache()
val w = sc.textFile(W_FILE).map(_.split("\t")).map(cols => (cols(0).toInt, cols(1).toDouble))

val jMatrix = matrix.map{ case (i, j, x) => (j, (i, x))}

val result = jMatrix.join(w).map{ case (j, ((i, x), w)) => (i, x * w) }.reduceByKey(_ + _).cache()


def saveVector(vector: RDD[(Int, Double)], filepath: String): Unit = {
	vector.map{ case (i, v) => s"$i\t$v" }.saveAsTextFile(filepath)
}

saveVector(result, RESULT_FILE)

val deltas = sc.textFile(DELTAS_FILE).map(_.split("\t")).map(cols => (cols(0).toInt, cols(1).toDouble))


val iMatrix = matrix.map{ case (i, j, x) => (i, (j, x)) }

//val testUpdates = iMatrix.join(deltas).map{ case (i, ((j, x), d)) => (j, x * d) }.reduceByKey(_ + _)
val updates = iMatrix.join(deltas).map{ case (i, ((j, x), d)) => (j, x * d) }.reduceByKey(_ + _)

saveVector(updates, UPDATES_FILE)

