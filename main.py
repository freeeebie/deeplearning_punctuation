import data
import utils
import models.rnns as rnns
import models.seq2seq as s2s

import models.modelbase as base


# text = data.read_data("data/training/4BH00005.txt", 50)
text = data.read_large_data("data/training")

dic_size = 100
input_chars = data.make_input_dic(text, dic_size)
output_chars = ['<nop>', ',', '.']

char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True)
output_char2vec = utils.Char2Vec(chars=output_chars)
input_size = char2vec.size
output_size = output_char2vec.size

# make and run multi layer LSTM network
hidden_size = 128

rnn_config = base.ModelConfiguration(input_size, hidden_size, output_size, epoch=1000)

multi_rnn = rnns.MultiLayerLSTM(rnn_config, char2vec, output_char2vec, text, seq_length=100, type="multi")
bidir_rnn = rnns.MultiLayerLSTM(rnn_config, char2vec, output_char2vec, text, seq_length=100, type="bimul")

rnn_models = [multi_rnn, bidir_rnn]
for rnn_model in rnn_models:
    rnn_model.run()


