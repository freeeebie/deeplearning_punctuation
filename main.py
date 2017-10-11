import tensorflow as tf
import numpy as np
import data
import utils
import model
import models.rnns as rnns
import models.seq2seq as s2s

import models.modelbase as base


print("=== start prediction of puncuation ===")
rawdata = data.read_data("data/4BE00006.txt")

dic_size = 100
input_chars, output_chars = data.make_dic(rawdata, dic_size)

text = rawdata

char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True)
output_char2vec = utils.Char2Vec(chars=output_chars)
input_size = char2vec.size
output_size = output_char2vec.size

# make and run multi layer LSTM network
hidden_size = 128


rnn_config = base.ModelConfiguration(input_size, hidden_size, output_size, epoch=500)

# base = rnns.MultiLayerLSTM(rnn_config, char2vec, output_char2vec, text, seq_length=100, type="multi")
# base.run()
# base = rnns.MultiLayerLSTM(rnn_config, char2vec, output_char2vec, text, seq_length=100, type="bimul")
# base.run()

base = s2s.Seq2Seq(rnn_config, char2vec, output_char2vec, text, seq_length=100, type="bimul")
base.run()

