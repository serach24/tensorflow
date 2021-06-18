import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/tmp/tf_graphs/'
os.environ['XLA_FLAGS'] = "--xla_dump_hlo_as_text --xla_dump_to=/home/chenhao/projects/xla_dump"
import signal

import numpy as np
import tensorflow as tf

npa = np.arange(1, 10, 1.6) # 1, 2.6, 4.2, 5.8, 7.4, 9
a = tf.cast(npa, tf.float32)

@tf.function(experimental_compile=True)
def model_fn(a):
    b = tf.cast(a, tf.cus)
    c = tf.matmul(tf.reshape(b, [2,3]), tf.reshape(b, [3,2]))
    return c
print(model_fn(a))
