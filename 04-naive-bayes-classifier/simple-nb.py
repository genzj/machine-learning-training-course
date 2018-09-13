# -*- encoding: utf-8 -*-
import functools

import numpy as np


# word list vector function
def loadDataSet():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'],
        ['i', 'want', 'to', 'hear', 'stories', 'about', 'your', 'dogs'],
        ['shut', 'up', 'you', 'idiot']
    ]
    class_vec = [0, 1, 0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def createVocabList(data_set):
    # vocab_set = set([])
    # for document in data_set:
    #     vocab_set = vocab_set | set(document)
    vocab_set = functools.reduce(lambda x, y: x | set(y), data_set, set([]))
    return list(vocab_set)


def setOfWords2Vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        assert word in vocab_list, "the word: %s is not in my Vocabulary!" % (word,)
        return_vec[vocab_list.index(word)] = 1
    return return_vec


def bagOfWords2VecMN(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


# naive bayes classifier training function
def trainNB0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category) / float(num_train_docs)
    p0_num = np.ones(num_words)
    p1_num = np.ones(num_words)  # Use Laplace methodology
    p0_denom = 2.0
    p1_denom = 2.0  # two classes (0/1) in total
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = np.log(p1_num / p1_denom)
    p0_vect = np.log(p0_num / p0_denom)
    return p0_vect, p1_vect, p_abusive


# Naive Bayes classify function
def classifyNB(vec2_classify, p0_vec, p1_vec, p_class1, classes=('normal', 'insulting')):
    p1 = sum(vec2_classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + np.log(1.0 - p_class1)
    if p1 > p0:
        return classes[1]
    else:
        return classes[0]


def testingNB():
    list_of_posts, list_classes = loadDataSet()
    my_vocab_list = createVocabList(list_of_posts)
    train_matrix = []
    for posting_doc in list_of_posts:
        train_matrix.append(bagOfWords2VecMN(my_vocab_list, posting_doc))
    p0_v, p1_v, p_ab = trainNB0(np.array(train_matrix), np.array(list_classes))

    test_set = [
        ['love', 'my', 'dalmation'],
        ['stupid', 'garbage'],
        ['stop', 'posting', 'about', 'your', 'idiot', 'dog']
    ]

    for test_entry in test_set:
        this_doc = np.array(bagOfWords2VecMN(my_vocab_list, test_entry))
        print(test_entry, 'classified as:', classifyNB(this_doc, p0_v, p1_v, p_ab))


if __name__ == '__main__':
    testingNB()

'''
['love', 'my', 'dalmation'] classified as: normal
['stupid', 'garbage'] classified as: insulting
['stop', 'posting', 'about', 'your', 'idiot', 'dog'] classified as: insulting

'''