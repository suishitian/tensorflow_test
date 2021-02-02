import tensorflow as tf
import config
from seq2seq_model import seq2seq_config


class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(config.word_dict_size, seq2seq_config.embedding_size)
        self.gru = tf.keras.layers.GRU(
            seq2seq_config.gru_hidden_size,
            return_sequences=True,
            return_state=True
        )

    def call(self, inputs):  ##inputs:(batch, max_len)
        x = self.embedding(inputs)  ##inputs:(batch, max_len, embedding_size)
        hidden_list, output_hidden = self.gru(x)  ##hidden_list:(batch, max_len, hidden_size)  output_hidden:(batch, hidden_size)
        return hidden_list, output_hidden

class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_hidden_size):
        super().__init__()
        self.attention_hidden_size = attention_hidden_size
        self.dense1 = tf.keras.layers.Dense(self.attention_hidden_size)
        self.dense2 = tf.keras.layers.Dense(self.attention_hidden_size)
        self.score = tf.keras.layers.Dense(1)

    def call(self, hidden_list, output_hidden):  ##hidden_list:(batch, max_len, hidden_size)  output_hidden:(batch, hidden_size)
        output_hidden_scaled = tf.expand_dims(output_hidden, axis=1)  ##output_hidden_scaled:(batch, 1, hidden_size)
        output_hidden_fc = self.dense1(output_hidden_scaled)  ##output_hidden_fc:(batch, 1, attention_hidden_size)
        hidden_list_fc = self.dense2(hidden_list)  ##hidden_list_fc:(batch, max_len, attention_hidden_size)
        attention_weights = tf.nn.tanh(output_hidden_fc + hidden_list_fc)  ##attention_weights:(batch, max_len, attention_hidden_size)
        attention_weights = self.score(attention_weights)  ##attention_weights:(batch, max_len, 1)
        attention_weights = tf.nn.softmax(attention_weights, axis=1)  ##attention_weights:(batch, max_len, 1)
        values = attention_weights * hidden_list  ##values:(batch, max_len, hidden_size)
        final_values = tf.reduce_sum(values, axis=1)  ##final_values:(batch, hidden_size)
        return final_values, attention_weights

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(config.word_dict_size, seq2seq_config.embedding_size)
        self.gru = tf.keras.layers.GRU(
            seq2seq_config.gru_hidden_size
        )
        self.dense = tf.keras.layers.Dense(config.word_dict_size)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, attention_value):  ##x:(batch, max_len), attention_value:(batch, hidden_size)
        x = self.embedding(x)  ##x:(batch, max_len, embedding_size)
        context_tensor = tf.concat([tf.expand_dims(attention_value, axis=1), x], axis=-1)  ##context_tensor:(batch, maxlen, hidden_size+embedding_size)
        hidden = self.gru(context_tensor)  ##hidden:(batch, hidden_size)
        pred = self.dense(hidden)  ##pred:(batch, word_dict_size)
        pred = self.bn(pred)
        output = tf.nn.softmax(pred)
        return output, hidden

class Seq2seq(tf.keras.Model):
    def __init__(self, token):
        super().__init__()
        self.token = token
        self.encoder = Encoder()
        self.attention = Attention(seq2seq_config.attention_hidden_size)
        self.decoder = Decoder()

    def call(self, inputs):
        enc_hidden_list, enc_output_hidden = self.encoder(inputs)
        cur_input_vector = tf.expand_dims([self.token.word_index[config.start]] * config.batch_size, axis=1)
        cur_hidden_vector = enc_output_hidden
        output = None
        for i in range(0, inputs.shape[1]):
            attention_context_vector, _ = self.attention(enc_hidden_list, cur_hidden_vector)
            cur_output, hidden = self.decoder(cur_input_vector, attention_context_vector)
            cur_hidden_vector = hidden
            cur_input_vector = tf.expand_dims(tf.argmax(cur_output, axis=-1), axis=1)
            if i==0: output = tf.expand_dims(cur_output, axis=1)
            else:
                cur_output = tf.expand_dims(cur_output, axis=1)
                output = tf.concat([output, cur_output], axis=1)
        return output

    def train_with_teaching_force(self, inputs, labels):
        enc_hidden_list, enc_output_hidden = self.encoder(inputs)
        cur_input_vector = tf.expand_dims([self.token.word_index[config.start]] * config.batch_size, axis=1)
        cur_hidden_vector = enc_output_hidden
        output = None
        for i in range(0, inputs.shape[1]):
            attention_context_vector, _ = self.attention(enc_hidden_list, cur_hidden_vector)
            cur_output, hidden = self.decoder(cur_input_vector, attention_context_vector)
            cur_hidden_vector = hidden
            #cur_input_vector = tf.expand_dims(tf.argmax(cur_output, axis=-1), axis=1)
            cur_input_vector = tf.expand_dims(labels[:,i,:], axis=1)
            if i == 0:
                output = tf.expand_dims(cur_output, axis=1)
            else:
                cur_output = tf.expand_dims(cur_output, axis=1)
                output = tf.concat([output, cur_output], axis=1)
        return output