from keras.layers import Layer
from keras import backend as K

class RepeatEmbedding(Layer):

    def __init__(self, repeat_count, **kwargs):
        self.repeat_count = repeat_count
        super(RepeatEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RepeatEmbedding, self).build(input_shape) # Be sure to call this at the end

    def call(self, x):
        return K.repeat_elements(x, self.repeat_count, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.repeat_count, input_shape[2])
