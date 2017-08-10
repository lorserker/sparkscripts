

val DATA_FILE = "/home/ldali/okr/tree_embedding/libsvm/training_sparse.libsvm.subneg.gz"

val data = spark.read.format("libsvm").load(DATA_FILE)

