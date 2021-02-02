import tensorflow as tf
import config
from transformer_model import transformer_config
import numpy as np

class PositionEncoder(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.max_len = max_len
        self.depth = d_model

    def postion_encoding(self):
        pos_vector = np.arange(self.max_len)
        depth_vector = np.arange(self.depth)
        depth_vector = 1 / np.power(10000, (2 * (depth_vector//2)) / np.float32(self.depth))
        # (pos, depth) = (pos, 1) * (1, depth)
        position_encoding = np.matmul(np.expand_dims(pos_vector,axis=-1),np.expand_dims(depth_vector, axis=0))
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        return tf.cast(position_encoding, dtype=tf.float32)

    def call(self, inputs):  # inputs:(batch, max_len, d_model)
        position_encoding = self.postion_encoding()  # position_encoding:(maxlen, d_model)
        # postion_embedding:(batch, max_len, d_model) = (batch, max_len, d_model) + (1, max_len, d_model)
        position_embedding = inputs + tf.expand_dims(position_encoding, axis=0)
        return position_embedding

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, attn_size):
        super().__init__()
        self.attn_size = attn_size
        self.Q = tf.keras.layers.Dense(attn_size)
        self.K = tf.keras.layers.Dense(attn_size)
        self.V = tf.keras.layers.Dense(attn_size)

    def call(self, inputs, mask, K_matrix=None, V_matrix=None):  # inputs(batch, input_len, embedding_size)
        # inputs = q = k = v
        attn_Q = self.Q(inputs)  # attn_q:(batch, input_len, attention_matrix_size)
        if K_matrix!=None: attn_K = self.K(K_matrix)
        else: attn_K = self.K(inputs)  # attn_k:(batch, max_len, attention_matrix_size)
        if V_matrix!=None: attn_V = self.V(V_matrix)
        else: attn_V = self.V(inputs)  # attn_v:(batch, max_len, attention_matrix_size)
        # attention_weight:(batch, input_len, max_len)
        attention_weights = tf.matmul(attn_Q, attn_K, transpose_b=True)
        if mask!=None:
            attention_weights *= mask*1e-9
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        attention_weights = attention_weights / tf.math.sqrt(tf.cast(self.attn_size, tf.float32))
        attention_values = tf.matmul(attention_weights, attn_V)  # attention_values:(batch, input, attention_matrix_size)
        # return attention_values, attention_weights
        return attention_values

# 这里写的跟官网有很大不同(官网并没有将多个头当做不同的sa，不知道是不是我理解的有问题)
class MultiheadSelfAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(MultiheadSelfAttention, self).__init__()
        assert transformer_config.d_model % transformer_config.head_count == 0
        attn_size = transformer_config.d_model / transformer_config.head_count
        self.self_attention_list = [SelfAttention(attn_size) for _ in range(transformer_config.head_count)]

    def call(self, inputs, mask, K_matrix=None, V_matrix=None):  # inputs:(batch, max_len, embedding_size)
        # 分割为多头，分别进行self_attention
        # heads[i]:(batch, max_len/head_counts, attention_size)
        heads = tf.split(inputs, num_or_size_splits=transformer_config.head_count, axis=-1)
        # 每一个头进行分别进行self_attention，且多个头之间参数不共享
        # multi_attention[i]:(batch, max_len/head_counts, attention_size)
        multi_attention = [self.self_attention_list[index](single_head, mask, K_matrix=K_matrix, V_matrix=V_matrix) for index, single_head in enumerate(heads)]
        # 将多个头合并
        concat_attention_values = tf.concat(multi_attention, axis=-1)  # concat_attention:(batch, max_len, atten_size)
        return concat_attention_values

class FFN(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(transformer_config.d_ff)
        self.dense2 = tf.keras.layers.Dense(transformer_config.d_model)

    def call(self, inputs):  # inputs:(batch, max_len, d_model)
        x = self.dense1(inputs)
        output = self.dense2(x)
        return output  # output:(batch, max_len, d_model)

if __name__=='__main__':
    def shape_test():
        multi_heads = MultiheadSelfAttention()
        inputs = tf.random.normal((config.batch_size, config.max_len, transformer_config.d_model))
        res = multi_heads(inputs)
        print(res.shape)
    def position_embedding_test():
        pos_obj = PositionEncoder(5, 5)
        pos_obj.postion_encoding()
    position_embedding_test()
