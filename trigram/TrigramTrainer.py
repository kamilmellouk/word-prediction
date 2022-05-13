#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs
import json
import requests
import tqdm

"""
This file is a modified version of the BigramTrainer.py script from the course DD1418/DD2418 Language engineering at KTH.
Original file created in 2017 by Johan Boye and Patrik Jonell.
Modified in 2022 by Kamil Mellouk
"""


class TrigramTrainer(object):
    """
    This class constructs a trigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file @code{f}.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            # TODO: capitalization?
            text = reader = str(text_file.read()).lower()
        try :
            self.tokens = nltk.word_tokenize(text) # Important that it is named self.tokens for the --check flag to work
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)


    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram,
        bigram and trigram counts.

        :param token: The current word to be processed.
        """
        # process new words
        if token in self.index.keys():
            index = self.index.get(token)

            self.total_words += 1
        else:
            index = self.unique_words
            self.index[token] = index
            self.word[index] = token

            self.unique_words += 1
            self.total_words += 1
        
        # set unigram count
        self.unigram_count[index] += 1

        # set bigram count
        if self.last_index != -1:
            self.bigram_count[self.last_index][index] += 1

        # set trigram count
        if self.last_index != -1 and self.before_last_index != -1:
            if index in self.trigram_count[self.before_last_index][self.last_index]:
                self.trigram_count[self.before_last_index][self.last_index][index] += 1
            else:
                self.trigram_count[self.before_last_index][self.last_index][index] = 1
        
        self.before_last_index = self.last_index
        self.last_index = index


    def stats(self):
        """
        Creates a list of rows to print of the language model.

        """
        rows_to_print = []

        # print metadata
        rows_to_print.append(str(self.unique_words) + ' ' + str(self.total_words))

        # print unigram counts
        for index, word in self.word.items():
            rows_to_print.append(str(index) + ' ' + word + ' ' + str(self.unigram_count.get(index)))

        # print bigram counts
        for i in self.word.keys():
            for j in self.word.keys():
                if self.bigram_count[i][j] != 0:
                    rows_to_print.append(
                        str(i) + ' ' + str(j) + ' ' +
                        "%.15f" % math.log(self.bigram_count[i][j] / self.unigram_count[i])
                    )
        
        for i in self.word.keys():
            for j in self.trigram_count[i].keys():
                for k in self.trigram_count[j].keys():
                    if k in self.trigram_count[i][j] and self.trigram_count[i][j][k] != 0:
                        rows_to_print.append(
                            str(i) + ' ' + str(j) + ' ' + str(k) + ' ' +
                            "%.15f" % math.log(self.trigram_count[i][j][k] / self.unigram_count[i])
                        )
        
        rows_to_print.append("-1")

        return rows_to_print

    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        # An array holding the bigram counts.
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # An array holding the trigram counts.
        self.trigram_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        # The identifier of the previous word processed.
        self.last_index = -1

        # The identifier of the before-last word processed
        self.before_last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        self.laplace_smoothing = False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='TrigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    trigram_trainer = TrigramTrainer()

    print("processing text...")
    trigram_trainer.process_files(arguments.file)

    print("computing statistics...")
    stats = trigram_trainer.stats()

    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)
    
    print("model file saved as " + arguments.destination)


if __name__ == "__main__":
    main()
