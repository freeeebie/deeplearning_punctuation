from bs4 import BeautifulSoup
import operator

def read_data(path):
    doc = BeautifulSoup(open(path,'rb'), 'html.parser')
    count = 0
    data = []
    for x in doc.findAll('p'):
        count = count + 1
        for y in x.getText():
            data.append(y)
        # if count == 20:
        #     break
    print("length of paragraph: ", count)
    return data

def extract_punc(string_input, input_chars, output_chars):
    input_source = []
    output_source = []
    input_length = len(string_input)
    i = 0
    # print(input_length)
    # print(string_input, input_chars, output_chars)
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

def make_dic(data):
    sorted_char_map = get_sorted_char_map(data)

    input_chars = []
    for i in range(min(50,len(sorted_char_map))):
        input_chars.append((sorted_char_map[i])[0])
    # print(input_chars)

    output_chars = ['<nop>', ',', '.']
    return input_chars, output_chars

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
