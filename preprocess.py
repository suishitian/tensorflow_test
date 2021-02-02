import tensorflow as tf
import config, random

class PreProcess:
    def __init__(self):
        self.data_file = config.data_file
        pass

    def make_word_token(self, ask, answer):
        token = tf.keras.preprocessing.text.Tokenizer(num_words=config.word_dict_size, oov_token='oov')
        token.fit_on_texts(ask)
        token.fit_on_texts(answer)
        print(len(token.word_index))
        ask_data = token.texts_to_sequences(ask)
        answer_data = token.texts_to_sequences(answer)
        ask_data = tf.keras.preprocessing.sequence.pad_sequences(ask_data, maxlen=config.max_len)
        answer_data = tf.keras.preprocessing.sequence.pad_sequences(ask_data, maxlen=config.max_len)
        return ask_data, answer_data, token

    def split_data(self, ask, answer, rate=0.2):
        index = int(len(ask)*rate)
        test_ask = ask[:index]
        test_answer = answer[:index]
        train_ask = ask[index:]
        train_answer = answer[index:]
        return train_ask, train_answer, test_ask, test_answer

    def read_data(self):
        ask_raw_data = list()
        answer_raw_data = list()
        lines = open(self.data_file,'r', encoding='utf-8').readlines()
        r = random.random
        random.seed(config.data_seed)
        random.shuffle(lines, random=r)
        for index, line in enumerate(lines):
            if config.read_limit>0 and index==config.read_limit: break
            ask_answer = line.split('\t')
            ask = ask_answer[0].split(" ")
            answer = ask_answer[1].split(" ")
            ask_data = [config.start] + ask + [config.end]
            answer_data = [config.start] + answer + [config.end]
            ask_raw_data.append(ask_data)
            answer_raw_data.append(answer_data)
        ask_data, answer_data, token = self.make_word_token(ask_raw_data, answer_raw_data)
        return ask_data, answer_data, token

    def process(self):
        ask_data, answer_data, token = self.read_data()
        train_ask, train_answer, test_ask, test_answer = self.split_data(ask_data, answer_data)
        return train_ask, train_answer, test_ask, test_answer, token


if __name__=='__main__':
    preprocess = PreProcess()
    preprocess.read_data()