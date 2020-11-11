import tensorflow as tf
import tensorflow.keras.backend as K
from keras.applications.mobilenet import MobileNet

def stats_graph(graph):
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: %.3fG;\nTrainable params: %.2fM' % (flops.total_float_ops / 2e9, params.total_parameters / 1e6))

if __name__ == '__main__':
    input_shape = (1, 224, 224, 3)  # NHWC
    input_type = 'float32'
    _input_tensor = tf.placeholder(input_type, shape=input_shape)
    model = MobileNet(alpha=1, weights=None, input_tensor=_input_tensor)
    graph = K.get_session().graph
    stats_graph(graph)
    model.summary()
