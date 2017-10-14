from bs4 import BeautifulSoup
import operator
import glob
import os

class DataSet():
    def __init__(self, input_batch, input_source, target_batch, seq_lens):
        self.input_batch = input_batch
        self.input_source = input_source
        self.target_batch = target_batch
        self.seq_lens = seq_lens

def read_data(filename, max_para=None):
    doc = BeautifulSoup(open(filename,'rb'), 'html.parser')
    count = 0
    data = []
    doc = doc.find('body')
    for x in doc.findAll('p'):
        count = count + 1
        for y in x.getText():
            data.append(y)
        data.append(' ')
        count = count + 1

        if max_para != None:
            if count == max_para:
                break
    print("length of paragraph: ", count, " characters: ", len(data))
    return data

def read_large_data(path):
    filenames = glob.glob(os.path.join(path, '*.txt'))
    data = []
    for filename in filenames:
        data = data + read_data(filename)

    print("total characters: ", len(data))
    return data


def extract_punc(string_input, input_chars, output_chars):
    input_source = []
    output_source = []
    input_length = len(string_input)
    i = 0

    while i < input_length:
        char = string_input[i]

        if char in output_chars:
            output_source.append(char)
            if i < input_length - 1:
                input_source.append(string_input[i + 1])
            else:
                input_source.append(" ")
            i += 1

        if char not in output_chars:
            input_source.append(char)
            output_source.append("<nop>")

        i += 1
    return input_source, output_source

def apply_punc(text_input, punctuation):
    assert len(text_input) == len(punctuation), "input string has differnt length from punctuation list" + "".join(
        text_input) + str(punctuation) + str(len(text_input)) + ";" + str(len(punctuation))
    result = ""
    for char1, char2 in zip(text_input, punctuation):
        if char2 == "<cap>":
            result += char1.upper()
        elif char2 == "<nop>":
            result += char1
        else:
            result += char2 + char1
    return result

def get_sorted_char_map(data):
    char_map = {}
    for ch in data:
        if ch != ',' and ch != '.':
            if ch in char_map:
                char_map[ch] = char_map[ch] + 1
            else:
                char_map[ch] = 1

    sorted_char_map = sorted(char_map.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_char_map

def make_input_dic(data, dic_size = 50):
    sorted_char_map = get_sorted_char_map(data)

    input_chars = []
    for i in range(min(dic_size ,len(sorted_char_map))):
        input_chars.append((sorted_char_map[i])[0])

    return input_chars

def make_sequences(input_data, char2vec, output_char2vec, seq_length, make_valid=True, modeltype=None):
    total_frames = 1 if (len(input_data) < seq_length) else int(len(input_data) / seq_length) + 1

    training_dataset = None
    valid_dataset = None

    if make_valid == True:
        frames_training = int(total_frames * 0.9)
        test_case = ['training', 'validation']
    else:
        frames_training = total_frames
        test_case = ['training']

    for case in test_case:
        input_batch = []
        output_batch = []
        target_batch = []

        input_source = []

        seqlens = []

        if case == 'training':
            start = 0
            end = frames_training
        else:
            start = frames_training + 1
            end = total_frames

        for i in range(start, end):
            input_str, output_str = extract_punc(input_data[i * seq_length: (i + 1) * seq_length],
                                                      char2vec.char_dict, output_char2vec.char_dict)
            # print(i, input_str, '->', output_str)
            input_source.append(input_str)
            x = []
            for ch in input_str:
                if ch in char2vec.char_dict:
                    x.append(char2vec.char_dict[ch])
                else:
                    x.append(char2vec.char_dict['<unk>'])
            y = [output_char2vec.char_dict[c] for c in output_str]  # y str to index

            seqlens.append(len(x))

            if len(x) != seq_length:
                diff = seq_length - len(x)
                for _ in range(diff):
                    x.append(0)
            if len(y) != seq_length:
                diff = seq_length - len(y)
                for _ in range(diff):
                    y.append(0)
                    input_str.append(' ')

            input_batch.append(x)
            target_batch.append(y)

        if case == 'training':
            training_dataset = DataSet(input_batch, input_source, target_batch, seqlens)
        else:
            valid_dataset = DataSet(input_batch, input_source, target_batch, seqlens)
    return training_dataset, valid_dataset

def compare_sentence(output_char2vec, target, input_source, prediction):
    prediction_output = ([output_char2vec.r_char_dict[c] for c in prediction])
    target_output = ([output_char2vec.r_char_dict[c] for c in target])

    prediction_output_str = apply_punc("".join(input_source), prediction_output)
    target_output_str = apply_punc("".join(input_source), target_output)

    return (target_output, target_output_str), (prediction_output, prediction_output_str)

if __name__ == "__main__":
    data = list("5보다 작은 자연수는 1, 2, 3, 4이다.")
    print(data)
    input_chars, output_chars =  make_dic(data)
    print(input_chars, output_chars)
    i, o = extract_punc("5보다 작은 자연수는 1, 2, 3, 4이다.", input_chars, output_chars)
    print("Punc-less Text:\n========================\n", i)
    print("\nPunctuation Operators Extracted:\n========================\n", o)
    result = apply_punc("".join(i), o)
    print("\nVarify that it works by recovering the original string:\n========================\n", result)
