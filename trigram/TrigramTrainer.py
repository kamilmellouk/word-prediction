#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import string
import nltk
import os
from collections import defaultdict
import codecs
import json
import requests
from tqdm import tqdm

"""
This file is a modified version of the BigramTrainer.py script from the course DD1418/DD2418 Language engineering at KTH.
Original file created in 2017 by Johan Boye and Patrik Jonell.
Modified in 2022 by Kamil Mellouk
"""
class TrigramTrainer(object):
    """
    This class constructs a trigram language model from a corpus.
    """

    def __init__(self, laplace=False, lowercase=False):
        """
        Constructor. Processes the file f and builds a language model from it.

        :param f: The training file.
        """
        # The mapping from words to identifiers.
        self.w2i = {}

        # The mapping from identifiers to words.
        self.i2w = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        # An array holding the bigram counts.
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # An array holding the trigram counts.
        self.trigram_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # The identifier of the previous i2w processed.
        self.last_index = -1

        # The identifier of the before-last i2w processed
        self.before_last_index = -1

        # Number of unique words (i2w forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # Indicates whether to apply Laplace smothing to the n-gram probabilities
        self.laplace_smoothing = laplace

        self.lower = lowercase

        self.CHARS = list(" abcdefghijklmnopqrstuvwxyz.'")
        self.CAP_CHARS = list(" abcdefghijklmnopqrstuvwxyz'ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.ALL_CHARS = [' ', '’', '—', '–', '“', '”', 'é', '‘'] + list(string.punctuation) + list(string.ascii_letters) + list(string.digits)


    def clean_line(self, line):
        dirty = line.split(" ")
        cleaned = []
        for w in dirty:
            if self.lower:
                if all([c in self.CHARS for c in w]):
                    cleaned.append(w)
            else:
                if all([c in self.CAP_CHARS for c in w]):
                    cleaned.append(w)

        return cleaned

    def process_files(self, f):
        """
        Processes the file f
        """
        for line in self.text_gen(f):
            for token in line:
                self.process_token(token)

    def text_gen(self, f):
        with open(f, encoding='utf8', errors='ignore') as f:
            for line in f:
                yield self.clean_line(line)

    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram,
        bigram and trigram counts.

        :param token: The current word to be processed.
        """
        # process new words
        if token in self.w2i.keys():
            w2i = self.w2i.get(token)

            self.total_words += 1
        else:
            w2i = self.unique_words
            self.w2i[token] = w2i
            self.i2w[w2i] = token

            self.unique_words += 1
            self.total_words += 1
        
        # set unigram count
        self.unigram_count[w2i] += 1

        # set bigram count
        if self.last_index != -1:
            self.bigram_count[self.last_index][w2i] += 1

        # set trigram count
        if self.last_index != -1 and self.before_last_index != -1:
            if w2i in self.trigram_count[self.before_last_index][self.last_index]:
                self.trigram_count[self.before_last_index][self.last_index][w2i] += 1
            else:
                self.trigram_count[self.before_last_index][self.last_index][w2i] = 1
        
        self.before_last_index = self.last_index
        self.last_index = w2i


    def stats(self):
        """
        Creates a list of rows to print from the language model.
        """
        rows, bigrams, trigrams = [], [], []
        
        rows.append(str(self.unique_words) + ' ' + str(self.total_words) + ' ' + str(self.laplace_smoothing)) # print metadata

        for i, w1 in tqdm(self.i2w.items(), desc="Saving model"):
            c1 = self.unigram_count[i]
            rows.append(str(i) + ' ' + w1 + ' ' + str(c1))

            for j, c2 in self.bigram_count[i].items():
                prob = (c2 + 1)/(c1 + self.unique_words) if self.laplace_smoothing else c2/c1
                bigrams.append(str(i) + ' ' + str(j) + ' ' + "%.15f" % math.log(prob))

                for k, c3 in self.trigram_count[i][j].items():
                    prob = (c3 + 1)/(c2 + self.unique_words) if self.laplace_smoothing else c3/c2
                    trigrams.append(str(i) + ' ' + str(j) + ' ' + str(k) + ' ' + "%.15f" % math.log(prob))
        rows.append("-1") # mark end of unigrams

        for row in bigrams:
            rows.append(row)
        rows.append("-2") # mark end of bigrams

        for row in trigrams:
            rows.append(row)
        rows.append("-3") # mark end of trigrams and file
        
        return rows
    

def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='TrigramTrainer')
    parser.add_argument('--file', '-f', type=str, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')
    parser.add_argument('--laplace', '-ls', action=argparse.BooleanOptionalAction, help='apply Laplace smoothing')
    parser.add_argument('--lowercase', '-lc', action=argparse.BooleanOptionalAction, help='only lowercase characters')

    arguments = parser.parse_args()

    trigram_trainer = TrigramTrainer(arguments.laplace, arguments.lowercase)

    if arguments.file:
        trigram_trainer.process_files(arguments.file)
    
    if arguments.destination:
        stats = trigram_trainer.stats()
        with codecs.open('./models/' + arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')


if __name__ == "__main__":
    main()
