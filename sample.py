import data
import utils
import models.modelbase as base
import time

text = data.read_data("data/training/4BH00005.txt", 100)
# text = data.read_large_data("data/training")

dic_size = 100
input_chars = data.make_input_dic(text, dic_size)
output_chars = ['<nop>', '.', ',']

char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True)
output_char2vec = utils.Char2Vec(chars=output_chars)
input_size = char2vec.size
output_size = output_char2vec.size

import tensorflow as tf

# make and run multi layer LSTM network

hidden_size = 128
seq_length = 100
modelconfig = base.ModelConfiguration(input_size, hidden_size, output_size, epoch=500)
evals = {}
types = ["multi", "bimul"]

type = "multi"
training_dataset, valid_dataset = data.make_sequences(text, char2vec, output_char2vec, seq_length)

input_batch = training_dataset.input_batch
target_batch = training_dataset.target_batch
seq_lens = training_dataset.seq_lens

hidden_size = modelconfig.hidden_size

tf.reset_default_graph()

X = tf.placeholder(tf.int32, [None, seq_length])  # X data
X_one_hot = tf.one_hot(X, modelconfig.input_size)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

Y = tf.placeholder(tf.int32, [None, seq_length])  # Y label

Sequences = tf.placeholder(tf.int32, [None])
keep_prob = tf.placeholder(tf.float32)

if type == "multi":
    print('\n****** MultiLayer LSTM start ******')

    with tf.variable_scope('cell_def'):
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=keep_prob)
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=keep_prob)
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

    with tf.variable_scope('rnn_def'):
        outputs, _states = tf.nn.dynamic_rnn(
            multi_cell, X_one_hot, dtype=tf.float32, sequence_length=Sequences)

elif type == "bimul":
    print('\n****** Bidirectional LSTM start ******')

    with tf.variable_scope('cell_def'):
        forward = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        forward = tf.nn.rnn_cell.DropoutWrapper(forward, output_keep_prob=keep_prob)
        backward = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        backward = tf.nn.rnn_cell.DropoutWrapper(backward, output_keep_prob=keep_prob)

    with tf.variable_scope('rnn_def'):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(forward, backward, inputs=X_one_hot, dtype=tf.float32,
                                                          sequence_length=Sequences)
        outputs = tf.concat(values=outputs, axis=2)

model = tf.layers.dense(outputs, modelconfig.output_size, activation=None)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
prediction = tf.argmax(model, axis=2)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

eval = utils.Evaluation(type, modelconfig.epoch, 25)

print('\n------------ Training ------------ ')
last_time = time.time()
for epoch in range(modelconfig.epoch):

    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch,
                                                     Y: target_batch,
                                                     Sequences: seq_lens,
                                                     keep_prob: 0.8})
    if epoch % 25 == 24:
        result = sess.run(prediction, feed_dict={X: valid_dataset.input_batch,
                                                 Y: valid_dataset.target_batch,
                                                 Sequences: valid_dataset.seq_lens,
                                                 keep_prob: 1})
        accuracy = tf.reduce_mean(tf.cast(tf.equal(result, tf.cast(Y, tf.int64)), tf.float32))
        accuracy_ret = sess.run(accuracy, feed_dict={Y: valid_dataset.target_batch})
        speed = time.time() - last_time
        print('Epoch:', '%04d  ' % (epoch + 1),
              'accuracy =', '{:.6f}  '.format(accuracy_ret),
              'cost =', '{:.6f}'.format(loss),
              'speed =', '{:.2f}'.format(speed), 'sec')
        last_time = time.time()

        avg_p, avg_r, avg_f = utils.print_evaluation(valid_dataset.target_batch, result, output_char2vec.char_dict)
        eval.set(epoch, accuracy_ret, loss, speed, avg_p, avg_r, avg_f)
        print('')

print('\n------------ Testing ------------ ')
test_sentences = data.read_data("data/test/BHXX0035.txt", 30)
test_dataset, _ = data.make_sequences(test_sentences, char2vec, output_char2vec, seq_length,
                                      make_valid=False)

result = sess.run(prediction, feed_dict={X: test_dataset.input_batch,
                                         Y: test_dataset.target_batch,
                                         Sequences: test_dataset.seq_lens,
                                         keep_prob: 1})

accuracy = tf.reduce_mean(tf.cast(tf.equal(result, tf.cast(Y, tf.int64)), tf.float32))
accuracy_ret = sess.run(accuracy, feed_dict={Y: test_dataset.target_batch})

print('Accuracy =', '{:.6f}'.format(accuracy_ret))

for index, predict_sequence in enumerate(result):
    target_output, prediction_output = data.compare_sentence(output_char2vec,
                                                             test_dataset.target_batch[index],
                                                             test_dataset.input_source[index],
                                                             predict_sequence)
    if index < 2:
        print("target sentence:    ", target_output[1])
        print("prediction sentence:", prediction_output[1])

avg_p, avg_r, avg_f = utils.print_evaluation(test_dataset.target_batch, result, output_char2vec.char_dict)

evals[type] = eval


import models.rnns as rnns

bidir_rnn = rnns.MultiLayerLSTM(modelconfig, char2vec, output_char2vec, text, seq_length=100, type="bimul")
evals[types[1]] = bidir_rnn.run()

import math, numpy as np, matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))

keys =list(evals["multi"].total.keys())
subplotnum = 321

multi_eval = evals["multi"].total
bimul_eval = evals["bimul"].total
small = 12
very_small = 10
for key in keys:
    plt.subplot(subplotnum)
    plt.title(key, fontsize=small)

    plt.plot([k[0] for k in multi_eval[key]], [k[1] for k in multi_eval[key]], label='multi')
    plt.plot([k[0] for k in bimul_eval[key]], [k[1] for k in bimul_eval[key]], label='bidir')
    plt.legend(loc='lower right')
    plt.ylabel(key, fontsize=very_small)
    plt.xlabel('Epoch', fontsize=very_small)
    subplotnum = subplotnum + 1
plt.tight_layout()

plt.show()
