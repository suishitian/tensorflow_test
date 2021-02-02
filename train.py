import tensorflow as tf
from seq2seq_model import *
from preprocess import *
import config
class Train:
    def __init__(self):
        self.preprocess = PreProcess()
        pass

    def loss_with_padding_filter(self, y_label, y_pred):  ##y_label:(batch, max_len, 1)  y_pred:(batch, max_len, word_dict_size)
        mask = tf.math.logical_not(tf.math.equal(y_label, 0))  ##mask:(batch, max_len, 1)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_label, y_pred, axis=-1)  ##loss:(batch, max_len)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_mean(loss)

    def drop_remainder(self, data, labels):
        length = data.shape[0]
        steps = int(length / config.batch_size)
        data = data[:(steps * config.batch_size), :]
        labels = labels[:(steps * config.batch_size), :]
        return data, labels

    def train_with_fit(self):
        train_ask, train_answer, test_ask, test_answer, token = self.preprocess.process()
        train_ask, train_answer = self.drop_remainder(train_ask, train_answer)
        test_ask, test_answer = self.drop_remainder(test_ask, test_answer)
        print(train_ask.shape)
        print(train_answer.shape)
        print(test_ask.shape)
        print(test_answer.shape)
        print("model")
        call_back = tf.keras.callbacks.ModelCheckpoint(filepath=config.model_file,
                                                       save_weights_only=False,
                                                       save_freq=config.save_freq,
                                                       monitor='val_loss',
                                                       verbose=1)
        self.model = Seq2seq(token)
        self.model.compile(
            optimizer=config.optimizer,
            loss=self.loss_with_padding_filter,
            metrics=config.metrics,
        )
        self.model.fit(
            x=train_ask, y=train_answer,
            batch_size=config.batch_size,
            #validation_split=0.1,
            epochs=config.epochs,
            shuffle=True,
            callbacks=[call_back]
        )
        print("evaluate")
        self.model.evaluate(test_ask, test_answer, batch_size=config.batch_size)


if __name__=='__main__':
    train = Train()
    train.train_with_fit()