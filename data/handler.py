import jieba
def xiaohuangji_handler(file_name, target_file, slice_num=-1):
    ask_list = list()
    answer_list = list()
    lines = open(file_name,'r', encoding='utf=8').read()
    pairs_list = lines.split("E")
    for index, pair in enumerate(pairs_list):
        if slice_num>0 and index==slice_num: break
        pair = pair.replace('/',"").replace("\n","").replace("\r","").replace(" ","")
        ask_answer = pair.split("M")
        if len(ask_answer)==3:
            print(ask_answer)
            ask = jieba.lcut(ask_answer[1])
            answer = jieba.lcut(ask_answer[2])
            ask_list.append(" ".join(ask))
            answer_list.append(" ".join(answer))
    with open(target_file, 'w', encoding='utf=8') as f:
        for num in range(len(ask_list)):
           f.write("%s\t%s\n"%(ask_list[num], answer_list[num]))



if __name__=='__main__':
    xiaohuangji_handler("./xiaohuangji.txt", "xiaohuangji_handled.txt")