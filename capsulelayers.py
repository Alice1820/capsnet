import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

class Length(layers, layers):
    """
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def complete_output_shape(self, input_shape):
        return input_shape[:-1]

class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    :params vectors: N-dim
    :params axis
    :return: Tensor with the same shape with squash
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon()) # ???

    return scale * vectors

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D dim_casule times and concatenate all CapsuleLayer
    :params intpus: [None, width, height, channels]
    :params dim_casule
    :params n_channels
    :return output_tensor: [None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filter=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

class CapsuleLayer(layers.layers):
    """
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    :return outputs: [None, num_capsule, dim_capsule], [None, 10, 16]
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_initializer='glorot_uniform', **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shaoe[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_dim_capsule, self.dim_capsule, self.input_dim_capsule], initializer = self.kernel_initializer, name='W')
        self.build= True

    def call(self, inputs, training='None'):
        # inputs [None, input_num_capsule, input_dim_capsule]
        # inputs_tiled [None, num_capsule, input_num_capsule, input_dim_capsule]
        inputs_tiled = K.tile(K.expand_dims(inputs, 1), [1, self.num_capsule, 1, 1])

        inputs_hat = K.map_fn(lambda x : K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        # Routing algorithm
        b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        for i in range(self.routings): # 3
            c = tf.nn.softmax(b, dim=1)
            outputs = squash(K.batch_dot(c, inputs_hat, [2, 2])) # [None, num_capsule, dim_casule]

            b += K.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs


    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])
