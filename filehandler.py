from seq2seq_model import seq2seq_config
import re, random
import jieba


class FileHandler:
    def __init__(self):
        self.file_name = seq2seq_config.file_name

    def line_clean(self, line):
        punctuation = '\r\n'
        line = re.sub(r'[{}]+'.format(punctuation), ' ', line)
        return line

    def process(self):
        ask_list = list()
        answer_list = list()
        lines = open(self.file_name, 'r', encoding='utf-8').read()
        line_list = lines.split('E')
        random.shuffle(line_list)
        if seq2seq_config.max_file_size > 0:
            line_list = line_list[:seq2seq_config.max_file_size]
        for line in line_list:
            line = self.line_clean(line)
            raw_str = line.replace(" ", "").split('M')
            if len(raw_str) == 3 and raw_str[0] == "":
                ask_str = [seq2seq_config.start_tag] + jieba.lcut(raw_str[1]) + [seq2seq_config.end_tag]
                answer_str = [seq2seq_config.start_tag] + jieba.lcut(raw_str[2]) + [seq2seq_config.end_tag]
                ask_list.append(ask_str)
                answer_list.append(answer_str)
                # print("%s : %s" % (ask_str, answer_str))
        print("input_list ask_list size: %d" % (len(ask_list)))
        print("input_list answer_list size: %d" % (len(answer_list)))
        return ask_list, answer_list


if __name__ == '__main__':
    # test_str = '!,;:?"\'、，；，。、《》？；’‘：“”【】、{}|·~！@#￥%……&*（）——+\r\n'
    file_handler = FileHandler()
    file_handler.process()
