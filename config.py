data_file = "./data/xiaohuangji_handled.txt"
model_file = "./model/weights_epoch{epoch:02d}_loss{loss:.2f}_acc{accuracy:.2f}.hdf5"

start = "<bos>"
end = "<eos>"
read_limit = -1

word_dict_size = 100725
max_len = 10

batch_size=128
epochs = 3
optimizer = 'adam'
loss=['sparse_categorical_crossentropy']
metrics=['accuracy']
save_freq=1000

##data_seed = 666
data_seed = 666

training = True