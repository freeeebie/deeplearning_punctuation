import data
import utils
import models.rnns as rnns
import models.seq2seq as s2s

import models.modelbase as base


text = data.read_data("data/training/4BH00005.txt", 50)
# text = data.read_large_data("data/training")

dic_size = 100
input_chars = data.make_input_dic(text, dic_size)
output_chars = ['<nop>', ',', '.']

char2vec = utils.Char2Vec(chars=input_chars, add_unknown=True)
output_char2vec = utils.Char2Vec(chars=output_chars)
input_size = char2vec.size
output_size = output_char2vec.size

# make and run multi layer LSTM network
hidden_size = 128

import tensorflow as tf

seq_length = 100
modelconfig = base.ModelConfiguration(input_size, hidden_size, output_size, epoch=500)
type = "multi"

if type == "multi":
    print('****** MultiLayer LSTM Initialize ******')
elif type == "bimul":
    print('****** Bidirectional LSTM Initialize ******')

training_dataset, valid_dataset = data.make_sequences(self.input, self.char2vec, self.output_char2vec, self.seq_length)

input_batch = training_dataset.input_batch
target_batch = training_dataset.target_batch
seq_lens = training_dataset.seq_lens

hidden_size = self.modelconfig.hidden_size

X = tf.placeholder(tf.int32, [None, seq_length])  # X data
Y = tf.placeholder(tf.int32, [None, seq_length])  # Y label

Seqlen = tf.placeholder(tf.int32, [None])

X_one_hot = tf.one_hot(X, modelconfig.input_size)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

keep_prob = tf.placeholder(tf.float32)

if type != "multi":
    with tf.variable_scope('cell_def'):
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=keep_prob)
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=keep_prob)
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

    with tf.variable_scope('rnn_def'):
        outputs, _states = tf.nn.dynamic_rnn(
            multi_cell, X_one_hot, dtype=tf.float32, sequence_length=Seqlen)

elif type != "bimul":
    with tf.variable_scope('cell_def'):
        forward = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        forward = tf.nn.rnn_cell.DropoutWrapper(forward, output_keep_prob=keep_prob)
        backward = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        backward = tf.nn.rnn_cell.DropoutWrapper(backward, output_keep_prob=keep_prob)

    with tf.variable_scope('rnn_def'):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(forward, backward, inputs=X_one_hot, dtype=tf.float32,
                                                          sequence_length=Seqlen)
        outputs = tf.concat(values=outputs, axis=2)

model = tf.layers.dense(outputs, modelconfig.output_size, activation=None)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
prediction = tf.argmax(model, axis=2)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

target = tf.cast(Y, tf.int64)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(modelconfig.epoch):
    l, loss = sess.run([optimizer, cost], feed_dict={X: input_batch,
                                                     Y: target_batch,
                                                     Seqlen: seq_lens,
                                                     keep_prob: 0.8})
    if epoch % 25 == 24:
        print("epoch #", epoch)

        result = sess.run(prediction, feed_dict={X: valid_dataset.input_batch,
                                                 Y: valid_dataset.target_batch,
                                                 Seqlen: valid_dataset.seq_lens,
                                                 keep_prob: 1})
        accuracy = tf.reduce_mean(tf.cast(tf.equal(result, tf.cast(Y, tf.int64)), tf.float32))
        print(sess.run(accuracy, feed_dict={Y: valid_dataset.target_batch}))

        prediction_output_mat = []
        target_output_mat = []

        for index, sq in enumerate(result):
            target_output, prediction_output = data.compare_sentence(output_char2vec,
                                                                     valid_dataset.target_batch[index],
                                                                     valid_dataset.input_source[index], sq)
            prediction_output_mat.append(prediction_output[0])
            target_output_mat.append(target_output[0])

        utils.print_pc_matrix(prediction_output_mat, target_output_mat)
        # if epoch == modelconfig.epoch - 1:
        #     print("target sentence:    ", target_output[1])
        #     print("prediction sentence:", prediction_output[1])

test_sentences = data.read_data("data/test/BHXX0035.txt", 30)
test_dataset, _ = data.make_sequences(test_sentences, char2vec, output_char2vec, seq_length,
                                      make_valid=False)

result = sess.run(prediction, feed_dict={X: test_dataset.input_batch,
                                         Y: test_dataset.target_batch,
                                         Seqlen: test_dataset.seq_lens,
                                         keep_prob: 1})

for index, sq in enumerate(result):
    target_output, prediction_output = data.compare_sentence(output_char2vec, test_dataset.target_batch[index],
                                                             test_dataset.input_source[index], sq)
    print("target sentence:    ", target_output[1])
    print("prediction sentence:", prediction_output[1])


