from termcolor import cprint, colored as c
import numpy as np
from sklearn import metrics

class Char2Vec():
    def __init__(self, size=None, chars=None, add_unknown=False):
        if chars is None:
            self.chars = CHARS
        else:
            self.chars = chars
        self.char_dict = {ch: i for i, ch in enumerate(self.chars)}
        self.r_char_dict = {i: ch for i, ch in enumerate(self.chars)}
        if size:
            self.size = size
        else:
            self.size = len(self.chars)
        if add_unknown:
            self.allow_unknown = True
            self.size += 1
            self.char_dict['<unk>'] = self.size - 1
        else:
            self.allow_unknown = False

class Evaluation():
    def __init__(self, model_name, max_epoch=1, epoch_unit=1, accuracy=[], cost=[], speed=[], precision=[], recall=[], fscore=[]):
        self.model_name = model_name
        self.epoch_unit = epoch_unit
        self.max_epoch = max_epoch
        self.accuracy = accuracy
        self.cost = cost
        self.speed = speed
        self.precision = precision
        self.recall = recall
        self.fscore = fscore
        self.totalcnt = self.max_epoch / self.epoch_unit
        self.total = {'acc': self.accuracy, 'cost': self.cost,
                      'speed': self.speed, 'prec': self.precision,
                      'recall': self.recall, 'fscore': self.fscore}
        self.average = {}

    def set(self, accuracy, cost, speed, precision, recall, fscore):
        self.accuracy.append(accuracy)
        self.cost.append(cost)
        self.speed.append(speed)
        self.precision.append(precision)
        self.recall.append(recall)
        self.fscore.append(fscore)

    def get_avg(self):
        for factor in self.total.keys():
            if self.total[factor] is not None:
                print(sum(self.total[factor]) / self.totalcnt)
                self.average[factor] = sum(self.total[factor]) / self.totalcnt
            else:
                self.average[factor] = 0

def get_pricision_recall_fscore(targets, predictions, label):
    total_cnt = len(targets)

    eval = np.zeros((4, 3))
    for target, prediction in zip(targets, predictions):
        eval = eval + np.array(metrics.precision_recall_fscore_support(target, prediction, labels=label, average=None))
    eval = eval / total_cnt

    precision = {k: eval[0][k] for k in label}
    recall = {k: eval[1][k] for k in label}
    fscore = {k: eval[2][k] for k in label}

    return precision, recall, fscore

def print_evaluation(targets, predictions, dict):
    p, r, f = get_pricision_recall_fscore(targets, predictions, list(dict.values()))

    avg_p = 0
    avg_r = 0
    avg_f = 0
    for k in list(dict.values()):
        key = [key for key, value in dict.items() if value == k][0]
        cprint("Key: " + c(("  " + key)[-5:], 'red') +
               "\tPrec: " + c("  {:.1f}".format(p[k] * 100)[-5:], 'green') + '%' +
               "\tRecall: " + c("  {:.1f}".format(r[k] * 100)[-5:], 'green')  + '%' +
               "\tF-Score: " + c("  {:.1f}".format(f[k] * 100)[-5:], 'green'))
        avg_p = avg_p + p[k]
        avg_r = avg_r + r[k]
        avg_f = avg_f + f[k]
    return avg_p / 3, avg_r / 3, avg_f / 3

if __name__ == "__main__":
    # TODO: Test code
    pass
