// :load /home/ldali/blackboard/distrw_logreg/src/ml_logreg.scala


import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.util.MLUtils

val DATA_FILE = "/home/ldali/wkspace/slrn/training_binary.libsvm.gz"
//val DATA_FILE = "/home/ldali/datasets/kdd2012/kdd12.val"
//val DATA_FILE = "/home/ldali/okr/tree_embedding/nb/clean/training_scorer.libsvm.gz"
val LAMBDA_REG = 0.1   // L2 regularization
val N_ITERATIONS = 10

val data = spark.read.format("libsvm").load(DATA_FILE)

val logreg = new LogisticRegression().setMaxIter(N_ITERATIONS).setRegParam(LAMBDA_REG)

val lrModel = logreg.fit(data)

lrModel.summary.objectiveHistory.foreach(println)

val binarySummary = lrModel.summary.asInstanceOf[BinaryLogisticRegressionSummary]
//binarySummary.roc.show

println(binarySummary.areaUnderROC)

println(lrModel.intercept)

// val data = MLUtils.loadLibSVMFile(sc, DATA_FILE).cache()

// val logreg = new LogisticRegressionWithLBFGS().setIntercept(true)
// logreg.optimizer.setRegParam(LAMBDA_REG).setNumIterations(N_ITERATIONS)

// val model = logreg.run(data)

// println(model.intercept)
