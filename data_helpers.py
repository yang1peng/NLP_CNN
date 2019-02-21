#-*-coding:utf-8-*-
import re
import numpy as np
import collections
def load_data_and_labels(positive_data_file, negative_data_file):
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # 清理数据替换掉无词义的符号
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def batch_iter(data, batch_size, num_epochs, shuffle=True): # shuffle=True洗牌
    """
    Generates a batch iterator for a dataset.
    """ # 每次只输出shuffled_data[start_index:end_index]这么多
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 # 每一个epoch有多少个batch_size
    for epoch in range(num_epochs): # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size)) # 洗牌
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size # 当前batch的索引开始
            end_index = min((batch_num + 1) * batch_size, data_size) # 判断下一个batch是不是超过最后一个数据了
            yield shuffled_data[start_index:end_index]


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary

    counter = collections.Counter(sentences)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))  # 负号代表降序排序，小括号代表元组,x[1],x[0]的顺序是有玄机的
    # 首先根据键值的降序排列，如果出现键值一样的，再根据键的顺序排

    words, _ = list(zip(*count_pairs))  # zip是压缩函数，将两个列表压成一个元组列表，对应压缩;‘*’代表解压缩，将元组解成两个列表,word代表键
    word_to_id = dict(zip(words, range(len(words))))  # dict将元组转化为字典，zip将word里的元素编号，弄成元组的形式

    return word_to_id  # 返回编好号的单词