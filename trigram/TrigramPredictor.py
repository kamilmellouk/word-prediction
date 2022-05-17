import argparse
import codecs
from collections import defaultdict

class TrigramPredictor:

    def __init__(self):
        
         # The mapping from words to identifiers.
        self.w2i = {}

        # The mapping from identifiers to words.
        self.i2w = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        # An array holding the bigram counts.
        self.bigram_prob = defaultdict(lambda: defaultdict(int))

        # An array holding the trigram counts.
        self.trigram_prob = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

         # Number of unique words (i2w forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0
    
    def read_model(self,filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: true if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                self.unique_words, self.total_words = map(int, f.readline().strip().split())

                print("reading unigram probabilities...")
                # read unigram probabilities
                for i in range(self.unique_words):
                    index, word, count = f.readline().strip().split()

                    self.w2i[word] = int(index)
                    self.i2w[i] = word
                    self.unigram_count[i] = int(count)
                
                self.unigram_count = sorted(self.unigram_count.items(), key=lambda x: x[1], reverse=True)
                # print(self.unigram_count)

                f.readline()

                print("reading bigram probabilities...")
                # read bigram probabilities
                for line in f:
                    if line.strip() == "-2":
                        break

                    i, j, log_p = line.strip().split()
                    self.bigram_prob[int(i)][int(j)] = float(log_p)

                for i in self.bigram_prob.keys():
                    # sort in reverse to get decreasing probabilities
                    self.bigram_prob[i] = sorted(self.bigram_prob[i].items(), key=lambda x: x[1], reverse=False)
                
                # print(self.bigram_prob)

                print("reading trigram probabilities...")
                # read trigram probabilities
                for line in f:
                    if line.strip() == "-3":
                        break

                    i, j, k, log_p = line.strip().split()
                    self.trigram_prob[int(i)][int(j)][int(k)] = float(log_p)

                for i, w1 in self.trigram_prob.items():
                    for j, w2 in w1.items():
                        # sort in reverse to get decreasing probabilities
                        self.trigram_prob[i][j] = sorted(self.trigram_prob[i][j].items(), key=lambda x: x[1], reverse=False)
                
                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False
    
    def predict(self, w0 = "", w1 = None, w2 = None):
        """
        """
        possible_words = []

        if w1 and w2:
            prev_options = self.trigram_prob.get(self.w2i[w2], None)
            if prev_options:
                options = prev_options.get(self.w2i[w1], None)
                if options:
                    possible_words = [(self.i2w[i], p) for (i, p) in options if self.i2w[i][:len(w0)] == w0][:3]
                    
        elif w1 and not w2:
            options = self.bigram_prob.get(self.w2i[w1], None)
            if options:
                possible_words = [(self.i2w[i], p) for (i, p) in options if self.i2w[i][:len(w0)] == w0][:3]

        else:
            possible_words = [(self.i2w[i], p) for (i, p) in self.unigram_count if self.i2w[i][:len(w0)] == w0][:3]
        
        return possible_words

    def interactive_word_predictor(self):
        print("Interactive word predictor session started...")

        while True:
            inp_string = input()
            if inp_string.strip().lower() == "exit":
                break
            elif inp_string == "":
                continue

            if inp_string.endswith(" "):
                last_word = ""
            else:
                last_word = inp_string.split()[-1]

            words = inp_string.strip().split()
            seq_size = len(words)

            words_pred = []

            if seq_size == 0:
                continue
            elif seq_size == 1:
                words_pred = self.predict(w0=last_word)
            elif seq_size == 2:
                words_pred = self.predict(w0=last_word, w1=words[0])
            else:
                words_pred = self.predict(w0=last_word, w1=words[seq_size-2], w2=words[seq_size-3])

            print(words_pred)
            # for w in words_pred:
            #     print(last_word + w, end=" | ")
            # print(last_word + words_pred[-1])

def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Word Predictor')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--stats', '-s', type=str, required=False, help='input a test file to run statistics on (how many keystrokes you would have saved)')

    arguments = parser.parse_args()

    predictor = TrigramPredictor()
    predictor.read_model(arguments.file)
    predictor.interactive_word_predictor()

if __name__ == "__main__":
    main()