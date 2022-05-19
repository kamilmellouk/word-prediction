import argparse
import codecs
from collections import defaultdict
from tqdm import tqdm

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
                self.unique_words, self.total_words = map(int, f.readline().strip().split()[:2])

                for i in tqdm(range(self.unique_words), desc="Reading unigrams"):
                    index, word, count = f.readline().strip().split()

                    self.w2i[word] = int(index)
                    self.i2w[i] = word
                    self.unigram_count[i] = int(count)
                
                self.unigram_count = sorted(self.unigram_count.items(), key=lambda x: x[1], reverse=True)

                f.readline()

                for line in tqdm(f, desc="Reading bigrams"):
                    if line.strip() == "-2":
                        break

                    i, j, log_p = line.strip().split()
                    self.bigram_prob[int(i)][int(j)] = float(log_p)

                for i in self.bigram_prob.keys():
                    # sort in reverse to get decreasing probabilities
                    self.bigram_prob[i] = sorted(self.bigram_prob[i].items(), key=lambda x: x[1], reverse=True)
                

                for line in tqdm(f, desc="Reading trigrams"):
                    if line.strip() == "-3":
                        break

                    i, j, k, log_p = line.strip().split()
                    self.trigram_prob[int(i)][int(j)][int(k)] = float(log_p)

                for i, w1 in self.trigram_prob.items():
                    for j, w2 in w1.items():
                        # sort in reverse to get decreasing probabilities
                        self.trigram_prob[i][j] = sorted(self.trigram_prob[i][j].items(), key=lambda x: x[1], reverse=True)
                
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
                    possible_words = [self.i2w[i] for (i, p) in options if self.i2w[i][:len(w0)] == w0][:3]
                    
        elif w1 and not w2:
            options = self.bigram_prob.get(self.w2i[w1], None)
            if options:
                possible_words = [self.i2w[i] for (i, p) in options if self.i2w[i][:len(w0)] == w0][:3]

        else:
            possible_words = [self.i2w[i] for (i, p) in self.unigram_count if self.i2w[i][:len(w0)] == w0][:3]
        
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
            if seq_size == 0:
                continue

            words_pred = []
            w1 = words[seq_size-2] if seq_size >= 2 else None
            w2 = words[seq_size-3] if seq_size >= 3 else None

            words_pred = self.predict(w0=last_word, w1=w1, w2=w2)


            l = len(words_pred)
            for i in range(l-1):
                print(words_pred[i], end=' | ')
            print(words_pred[l-1])

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