import tensorflow as tf
import numpy as np
import data


def train_baiclstm():
    data_dim = len(char_set)
    rnn_hidden_size = hidden_size = len(char_set)
    num_classes = len(output_chars)
    seq_length = 30  # Any arbitrary number

    batch_size = len(dataX)

    # seq_length는 임의로 10의 크기로 잘랐다고 가정합니다. batch_size는 dataX의 전체 길이를 넣습니다. 나머지는 똑같습니다.


    X = tf.placeholder(tf.int32, [None, seq_length])  # X data
    Y = tf.placeholder(tf.int32, [None, seq_length])  # Y label
    Weight = tf.placeholder(tf.float32, [None, None])  # Y label

    X_one_hot = tf.one_hot(X, data_dim)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_classes, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(
        cell, X_one_hot, initial_state=initial_state, dtype=tf.float32, sequence_length=seqlens)

    weights = tf.ones([batch_size, seq_length])
    # print(weights)
    # for i in range(len(dataX)):
    #     x = seqlens[i]
    #     for j in range(30 - x):
    #         weights[i][j + 30 - x] = 0

    # sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y , weights=weights)
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=Weight)
    loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

    wei = []
    for i in range(len(dataX)):
        k = []
        for j in range(30):
            if j < seqlens[i]:
                k.append(1)
            else:
                k.append(0)
        wei.append(k)

    print(wei)
    prediction = tf.argmax(outputs, axis=2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3000):
            l, _ = sess.run([loss, train], feed_dict={X: dataX, Y: dataY, Weight: wei})
            result = sess.run(prediction, feed_dict={X: dataX, Weight: wei})
            # print char using dic
            print(np.squeeze(result))

            result_str = [output_chars[c] for c in np.squeeze(result)[0]]
            print(result_str)

            result2 = data.apply_punc("".join(org_data[0]), result_str)

            print(i, "loss:", l, "Prediction:", result2)
