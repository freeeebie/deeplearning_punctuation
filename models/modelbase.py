class ModelConfiguration():
    def __init__(self, input_size, hidden_size, output_size, batch_size=1, epoch=400):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.epoch = epoch

class ModelBase():
    def __init__(self, modelconfig, char2vec=None, output_char2vec=None, input=None):
        self.modelconfig = modelconfig
        if char2vec is None:
            self.char2vec = Char2Vec()
        else:
            self.char2vec = char2vec

        if output_char2vec is None:
            self.output_char2vec = self.char2vec
        else:
            self.output_char2vec = output_char2vec

        self.input = input
