import tensorflow as tf
import numpy as np
import utils
import data
import models.modelbase as base


def make_weight_mat(input_batch, seq_lens, seq_length):
    weight_mat = []
    for i in range(len(input_batch)):
        weight_list = []
        # Todo: need refactoring
        for j in range(seq_length):
            if j < seq_lens[i]:
                weight_list.append(1)
            else:
                weight_list.append(0)
        weight_mat.append(weight_list)
    return weight_mat

def compare_sentence(output_char2vec, target, input_source, prediction, printable):
    result_output = ([output_char2vec.r_char_dict[c] for c in prediction])
    target_output = ([output_char2vec.r_char_dict[c] for c in target])

    if printable == True:
        result_str = data.apply_punc("".join(input_source), result_output)
        target_str = data.apply_punc("".join(input_source), target_output)

        print("Target:    ", target_str)
        print("Prediction:", result_str)
    return result_output, target_output


class Seq2Seq(base.ModelBase):
    def __init__(self, modelconfig, char2vec=None, output_char2vec=None, input=None, seq_length=100, type="multi"):
        base.ModelBase.__init__(self, modelconfig, char2vec, output_char2vec, input)
        self.seq_length = seq_length
        self.type = type
        tf.reset_default_graph()
        self.prediction = None
        self.sess = None

        if type == "multi":
            print('****** MultiLayer LSTM Initialize ******')
        elif type == "bimul":
            print('****** Bidirectional LSTM Initialize ******')

    def run(self):
        training_dataset, valid_dataset = data.make_sequences(self.input, self.char2vec, self.output_char2vec, self.seq_length)

        input_batch = training_dataset.input_batch
        target_batch = training_dataset.target_batch
        seq_lens = training_dataset.seq_lens

        hidden_size = self.modelconfig.hidden_size

        X = tf.placeholder(tf.int32, [None, self.seq_length])  # X data
        Dec_Input = tf.placeholder(tf.int32, [None, self.seq_length])  # X data
        Y = tf.placeholder(tf.int32, [None, self.seq_length])  # Y label

        Seqlen = tf.placeholder(tf.int32, [None])
        Weight = tf.placeholder(tf.float32, [None, None])  # Weight

        X_one_hot = tf.one_hot(X, self.modelconfig.input_size)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
        Dec_Input_one_hot = tf.one_hot(Dec_Input, self.modelconfig.input_size)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

        keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('encode'):

            enc_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
            enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
            print("enc_input")
            outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, X_one_hot, dtype=tf.float32
                                                    ,
                                                    sequence_length=Seqlen)

        with tf.variable_scope('decode'):
            dec_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
            dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)

            outputs, enc_states = tf.nn.dynamic_rnn(dec_cell, Dec_Input_one_hot, initial_state=enc_states, dtype=tf.float32
                                                    ,
                                                    sequence_length=Seqlen)




        model = tf.layers.dense(outputs, self.modelconfig.output_size, activation=None)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
        self.prediction = tf.argmax(model, axis=2)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        weights = make_weight_mat(input_batch, seq_lens, self.seq_length)

        target = tf.cast(Y, tf.int64)

        self.sess = tf.Session()
        sess = self.sess
        sess.run(tf.global_variables_initializer())

        for epoch in range(self.modelconfig.epoch):
            l, loss = sess.run([optimizer, cost], feed_dict={X: input_batch,
                                                             Dec_Input: target_batch,
                                                             Y: target_batch,
                                                             Weight: weights,
                                                             Seqlen: seq_lens,
                                                             keep_prob: 0.8})
            if epoch % 25 == 24:
                print("epoch #", epoch)

                valid_weights = make_weight_mat(valid_dataset.input_batch, valid_dataset.seq_lens, self.seq_length)

                result = sess.run(self.prediction, feed_dict={X: valid_dataset.input_batch,
                                                              Dec_Input: valid_dataset.target_batch,

                                                              Y: valid_dataset.target_batch,
                                                         Weight: valid_weights,
                                                         Seqlen: valid_dataset.seq_lens,
                                                         keep_prob: 1})

                sess.run(target, feed_dict={Y: valid_dataset.target_batch})
                is_correct = tf.equal(self.prediction, target)

                accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
                print(sess.run(accuracy, feed_dict={X: valid_dataset.input_batch,
                                                    Dec_Input: valid_dataset.target_batch,

                                                    Y: valid_dataset.target_batch,
                                                    Weight: valid_weights,
                                                    Seqlen: valid_dataset.seq_lens,
                                                    keep_prob: 1}))


                result_output_mat = []
                target_output_mat = []

                for index, sq in enumerate(result):
                    result_output, target_output = compare_sentence(self.output_char2vec,
                        valid_dataset.target_batch[index], valid_dataset.input_source[index],
                        sq, True if epoch == (self.modelconfig.epoch - 1) else False)
                    result_output_mat.append(result_output)
                    target_output_mat.append(target_output)

                utils.print_pc_matrix(result_output_mat, target_output_mat)

        test_sentence = list("나는 사과, 참외, 배, 딸기를 샀다")

        test_dataset, _ = data.make_sequences(test_sentence, self.char2vec, self.output_char2vec, self.seq_length, make_valid=False)
        weights = make_weight_mat(test_dataset.input_batch, test_dataset.seq_lens, self.seq_length)

        result = sess.run(self.prediction, feed_dict={X: test_dataset.input_batch,
                                                      Dec_Input: test_dataset.target_batch,

                                                      Y: test_dataset.target_batch,
                                                Weight: weights,
                                                 Seqlen: test_dataset.seq_lens,
                                                 keep_prob: 1})

        for index, sq in enumerate(result):
            result_output, target_output = compare_sentence(self.output_char2vec, test_dataset.target_batch[index],
                                                                 test_dataset.input_source[index], sq, printable=True)


