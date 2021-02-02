import tensorflow as tf
import config
from transformer_model.base_model import *

class EncoderUnit(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.multi_heads_self_attention = MultiheadSelfAttention()
        self.ffn = FFN()

        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate=transformer_config.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=transformer_config.dropout_rate)

    def call(self, inputs, training, mask):  # inputs(batch, max_len)
        ln1_input = self.multi_heads_self_attention(inputs, mask)
        ln1_output = self.dropout1(ln1_input, training=training)
        ln2_input = self.ln1(ln1_input + ln1_output)

        ln2_input = self.ffn(ln2_input)
        ln2_output = self.dropout2(ln2_input, training=training)
        ln2_output = self.ln2(ln2_input + ln2_output)
        return ln2_output  # output:(batch, max_len, d_model)

class DecoderUnit(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.multi_heads_self_attention1 = MultiheadSelfAttention()
        self.multi_heads_self_attention2 = MultiheadSelfAttention()
        self.ffn = FFN()

        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.ln3 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate=transformer_config.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=transformer_config.dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=transformer_config.dropout_rate)

    def call(self, inputs, training, enc_output, seq_ahead_mask, padding_mask):  # inputs(batch, max_len), enc_output(batch, max_len, d_model)
        ln1_input = self.multi_heads_self_attention1(inputs, seq_ahead_mask)
        ln1_output = self.dropout1(ln1_input, training=training)
        ln2_input = self.ln1(ln1_input + ln1_output)

        ln2_input = self.multi_heads_self_attention2(ln2_input, padding_mask, K_matrix=enc_output, V_matrix=enc_output)
        ln2_output = self.dropout2(ln2_input, training=training)
        ln3_input = self.ln2(ln2_input + ln2_output)

        ln3_input = self.ffn(ln3_input)
        ln3_output = self.dropout3(ln3_input, training=training)
        ln3_output = self.ln3(ln3_input + ln3_output)
        return ln3_output  # output:(batch, max_len, d_model)

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(config.word_dict_size, transformer_config.d_model)
        self.position_encoder = PositionEncoder(config.max_len, transformer_config.d_model)
        self.encoder_layer_list = [EncoderUnit() for _ in range(transformer_config.encoder_num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate=transformer_config.dropout_rate)

    def call(self, inputs, training):
        mask = tf.expand_dims(tf.cast(tf.equal(inputs, 0), dtype=tf.float32), axis=1)
        inputs = self.embedding(inputs)
        inputs = inputs / tf.math.sqrt(tf.cast(transformer_config.d_model, tf.float32))  ## 猜测这一步是为了减少self_attention的点积值太大
        positional_inputs = self.position_encoder(inputs)
        positional_inputs = self.dropout(positional_inputs, training=training)
        for num in range(transformer_config.encoder_num_layers):
            positional_inputs = self.encoder_layer_list[num](positional_inputs, training, mask)
        return positional_inputs  # positional_inputs:(batch, max_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(config.word_dict_size, transformer_config.d_model)
        self.position_encoder = PositionEncoder(config.max_len, transformer_config.d_model)
        self.decoder_layer_list = [DecoderUnit() for _ in range(transformer_config.decoder_num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate=transformer_config.dropout_rate)

    def call(self, targets, enc_output, training):
        padding_mask = tf.expand_dims(tf.cast(tf.equal(targets, 0), dtype=tf.float32), axis=1)
        seq_ahead_mask = 1 - tf.linalg.band_part(tf.ones((config.max_len, config.max_len)), -1, 0)
        targets = self.embedding(targets)
        targets = targets / tf.math.sqrt(tf.cast(transformer_config.d_model, tf.float32))  ## 猜测这一步是为了减少self_attention的点积值太大
        positional_targets = self.position_encoder(targets)
        positional_targets = self.dropout(positional_targets, training=training)
        for num in range(transformer_config.decoder_num_layers):
            positional_targets = self.decoder_layer_list[num](positional_targets, training, enc_output, seq_ahead_mask, padding_mask)
        return positional_targets  # positional_inputs:(batch, max_len, d_model)
    
class Transformer(tf.keras.layers.Layer):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.final_dense = tf.keras.layers.Dense(config.word_dict_size)

    def call(self, inputs, targets, training):
        enc_output = self.encoder(inputs, training)
        decoder_output = self.decoder(targets, enc_output, training)
        output = self.final_dense(decoder_output)
        return tf.nn.softmax(output)

if __name__=='__main__':
    inputs = tf.random.uniform((config.batch_size, config.max_len), minval=0, maxval=config.word_dict_size-1, dtype=tf.int32)
    targets = tf.random.uniform((config.batch_size, config.max_len), minval=0, maxval=config.word_dict_size-1,dtype=tf.int32)
    enc_output = tf.random.normal((config.batch_size, config.max_len, transformer_config.d_model))
    #encoder_unit = EncoderUnit()
    #enc_res = encoder = Encoder()
    #output = encoder(inputs, config.training)
    #decoder = Decoder()
    #output = decoder(inputs, enc_output, config.training)
    transformer = Transformer()
    output = transformer(inputs, targets, config.training)
    print(inputs.shape)
    print(output.shape)