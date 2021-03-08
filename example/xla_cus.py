import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'

import numpy as np
import tensorflow as tf

a = np.arange(1, 10, dtype=tf.bfloat16)

b = np.arange(1, 10, dtype=tf.bfloat16)

# inp = np.random.rand(10, 10)

@tf.function(experimental_compile=True)
def model_fn(a, b):
    res = tf.add(a, b)
    return res

print(model_fn(a, b))
