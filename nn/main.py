import argparse
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from sklearn.neighbors import NearestNeighbors
from operator import itemgetter
from tqdm import tqdm
import random
from random import shuffle
import string
from os.path import isfile

CHARS = list(" abcdefghijklmnopqrstuvwxyz.'")
CAP_CHARS = list(" abcdefghijklmnopqrstuvwxyz'ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALL_CHARS = [' ', '’', '—', '–', '“', '”', 'é', '‘'] + list(string.punctuation) + list(string.ascii_letters) + list(string.digits)

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

class NLM:
    def __init__(self, data_filename, learning_rate = 0.002, model_news=False, model_lower=False, fake_test=False, size_batch=128, hidden_size=256, char_emb_size=16):
        torch.manual_seed(99)
        np.random.seed(99)
        random.seed(99)
        self.filename = data_filename
        self.n_batch = size_batch
        self.n_hidden = hidden_size
        self.n_char_emb = char_emb_size
        self.model_lower = model_lower
        self.fake_test = fake_test
        model_folder = "./models/"
        if model_lower:
            self.CHARS = CHARS
            if model_news:
                self.model_files = [model_folder + f for f in ['news_emb_low.pt', 'news_lstm_low.pt', 'news_linear_low.pt']]
            else:
                self.model_files = [model_folder + f for f in ['blog_emb_low.pt', 'blog_lstm_low.pt', 'blog_linear_low.pt']]
        else:
            self.CHARS = ALL_CHARS
            if model_news:
                self.model_files = [model_folder + f for f in ['news_emb_all.pt', 'news_lstm_all.pt', 'news_linear_all.pt']]
            else:
                self.model_files = [model_folder + f for f in ['blog_emb_all.pt', 'blog_lstm_all.pt', 'blog_linear_all.pt']]

        self.n_classes = len(self.CHARS)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.seq_size = 129 # Real sequence size becomes [self.seq_size - 1]

        self.c2i = {c: i for i, c in enumerate(self.CHARS)}
        self.i2c = self.CHARS
        self.char_emb = nn.Embedding(len(self.CHARS), char_emb_size)
        
        self.model = nn.LSTM(input_size=char_emb_size, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.2).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.final_linear = nn.Linear(self.n_hidden, self.n_classes).to(self.device)
        self.optimizer = optim.Adam(self.get_params(), lr=learning_rate)

    def get_params(self):
        return list(self.char_emb.parameters()) + list(self.model.parameters()) + list(self.final_linear.parameters())

    def clean_line(self, line):
        if self.model_lower:
            line = ''.join([c for c in line.lower() if (c in self.CHARS)])
        else:
            line = ''.join([c for c in line if (c in self.CHARS)])
        return line

    def text_gen(self):
        with open(self.filename, encoding='utf8', errors='ignore') as f:
            for line in f:
                yield self.clean_line(line)

    def train_on_batch(self, data, epoch):
        batch_size = self.seq_size * self.n_batch
        pred_batch_size = (self.seq_size-1) * self.n_batch
        for i in tqdm(range(batch_size, len(data), batch_size), desc="Epoch %d" % epoch):
            batch = data[i - batch_size:i]

            batch_inds = torch.LongTensor(itemgetter(*batch)(self.c2i)).reshape((self.n_batch, self.seq_size))
            x_inds = batch_inds[:, :-1]

            self.optimizer.zero_grad()
            x = self.char_emb(x_inds).to(self.device)
            y = batch_inds[:, 1:].ravel().to(self.device)

            outputs, h = self.model(x)
            pred = self.final_linear(outputs)
            loss = self.loss_fn(pred.reshape((pred_batch_size, self.n_classes)), y)
            loss.backward()
            clip_grad_norm_(self.get_params(), 5)
            self.optimizer.step()

            if (self.iterations % 200) == 0:
                #print("Loss: %.6f" % loss.item())
                pass
            self.iterations += 1            


    def train(self, epochs):
        self.iterations = 0

        text = self.text_gen()
        dataset = []
        cur_seq = ""

        for s in text:
            cur_seq += s + ' '
            if len(cur_seq) >= self.seq_size:
                dataset.extend(list(cur_seq[:self.seq_size]))
                cur_seq = ""

        for epoch in range(epochs):
            self.train_on_batch(dataset, epoch+1)

        self.save_model()

    def save_model(self):
        print("Save model? (Y/n): ", end="")
        inp = input()
        if inp.lower().strip() == "y":
            torch.save(self.char_emb.state_dict(), self.model_files[0])
            torch.save(self.model.state_dict(), self.model_files[1])
            torch.save(self.final_linear.state_dict(), self.model_files[2])
            print("Model saved.")
        else:
            print("No save.")

    def load_model(self):
        self.char_emb.load_state_dict(torch.load(self.model_files[0]))
        self.model.load_state_dict(torch.load(self.model_files[1]))
        self.final_linear.load_state_dict(torch.load(self.model_files[2]))

    def sample_preds(self, preds, temperature=0.9):
        """
            Sample function from Keras. Higher temperature -> more random.
        """
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def eval_mode(self):
        self.char_emb.eval()
        self.model.eval()
        self.final_linear.eval()

    def interactive(self):
        self.eval_mode()
        print("Interactive session started...")
        n_predict = 200
        while True:
            inp = input()
            if inp.strip().lower() == "exit":
                break
            ok = True
            for c in inp:
                if not (c.lower() in self.CHARS):
                    print("Input not acceptable. List of acceptable input characters:")
                    print(self.CHARS)
                    ok = False
                    break
            if not ok:
                continue

            if self.model_lower:
                inp = list(inp.lower())
            else:
                inp = list(inp)
            seq_size = len(inp)
            x_preds = torch.LongTensor(n_predict)
            if seq_size == 0:
                continue
            elif seq_size == 1:
                x_inds = torch.LongTensor([itemgetter(*inp)(self.c2i)])
            else:
                x_inds = torch.LongTensor(itemgetter(*inp)(self.c2i))
            with torch.no_grad():
                x_emb = self.char_emb(x_inds).reshape((1, seq_size, self.n_char_emb)).to(self.device)
                outputs, h = self.model(x_emb)
                pred = self.final_linear(h[0][1, 0, :])
                probs = softmax(pred.cpu().tolist())
                x_preds[0] = self.sample_preds(probs)

                for i in range(1, n_predict):
                    x_emb = self.char_emb(torch.LongTensor([x_preds[i-1]])).reshape((1, 1, self.n_char_emb)).to(self.device)
                    outputs, h = self.model(x_emb, h)
                    pred = self.final_linear(h[0][1, 0, :])
                    probs = softmax(pred.cpu().tolist())
                    x_preds[i] = self.sample_preds(probs)

            char_preds = itemgetter(*x_preds.tolist())(self.i2c)
            string_pred = ''.join(inp) + ''.join(char_preds)

            print(string_pred)

    def get_nth_highest(self, vals, n): # 1 - highest, n - lowest
        """
            Returns tuple: (nth highest val in [vals], indx of nth highest val in [vals])
        """
        arr = np.array(vals)
        perm = arr.argsort()
        inds = np.arange(len(vals))

        arr = arr[perm]
        inds = inds[perm]

        return arr[-n], inds[-n]

    def get_k_probs(self, inp, seq_size, choices = None):
        max_len = 30
        t_probs = []
        if not choices:
            choices = {}

        t = 0
        count = 0
        x_preds = []

        with torch.no_grad():
            x_emb = self.char_emb(inp).reshape((1, seq_size, self.n_char_emb)).to(self.device)
            outputs, h = self.model(x_emb)
            pred = self.final_linear(h[0][1, 0, :])
            probs = softmax(pred.cpu().tolist())
            t_probs.append(probs)
            if t in choices:
                select = choices[t]
                val, ind = self.get_nth_highest(probs, select)
                x_preds.append(ind)
            else:
                x_preds.append(torch.argmax(pred))

            count += 1
            while (x_preds[-1] != self.c2i[' ']) and (count != max_len):
                t += 1
                x_emb = self.char_emb(torch.LongTensor([x_preds[-1]])).reshape((1, 1, self.n_char_emb)).to(self.device)
                outputs, h = self.model(x_emb, h)
                pred = self.final_linear(h[0][1, 0, :])
                probs = softmax(pred.cpu().tolist())
                t_probs.append(probs)
                if t in choices:
                    select = choices[t]
                    val, ind = self.get_nth_highest(probs, select)
                    x_preds.append(ind)
                else:
                    x_preds.append(torch.argmax(pred))
                count += 1

        return x_preds, t_probs

    def predictions(self, inp_string):
        if inp_string.endswith(" "):
            last_word = ""
        else:
            last_word = inp_string.split()[-1]

        if self.model_lower:
            inp = list(inp_string.lower())
        else:
            inp = list(inp_string)
        seq_size = len(inp)

        if seq_size == 0:
            return None
        else:
            x_inds = torch.LongTensor([self.c2i[c] for c in inp])

        words_pred = []
        with torch.no_grad():
            # Get most likely word
            x_preds, probs = self.get_k_probs(x_inds, seq_size)
            words_pred.append(''.join(itemgetter(*x_preds)(self.i2c))[:-1])

            # Get 2nd most likely word
            best2nd_rat = 0.0
            best2nd_ind = (0, 0)
            for t, p in enumerate(probs):
                high_val = max(p)
                secnd_val, secnd_ind = self.get_nth_highest(p, 2)
                ratio = secnd_val / high_val
                if (t == 0) or (ratio > best2nd_rat):
                    best2nd_rat = ratio
                    best2nd_ind = (t, 2)

            dic = {}
            dic[best2nd_ind[0]] = best2nd_ind[1] # Get 2nd largest prob
            x_preds, probs1 = self.get_k_probs(x_inds, seq_size, choices=dic)
            words_pred.append(''.join(itemgetter(*x_preds)(self.i2c))[:-1])

            # Get 3rd most likely word
            best3rd_rat = 0.0
            best3rd_ind = (None, None)
            secnd_winner = True
            for t, p in enumerate(probs1):
                if t <= best2nd_ind[0]:
                    continue
                high_val = max(p)
                third_val, third_ind = self.get_nth_highest(p, 2)
                ratio = third_val / high_val
                if (t == (best2nd_ind[0] + 1)) or (ratio > best3rd_rat):
                    best3rd_rat = ratio
                    best3rd_ind = (t, 2)

            best3rd_rat -= (1.0 - best2nd_rat)

            for t, p in enumerate(probs):
                kth = 2
                if t == best2nd_ind[0]:
                    kth = 3
                high_val = max(p)
                third_val, third_ind = self.get_nth_highest(p, kth)
                ratio = third_val / high_val
                if (ratio > best3rd_rat):
                    best3rd_rat = ratio
                    best3rd_ind = (t, kth)
                    secnd_winner = False

            if not secnd_winner:
                dic.clear()
                
            dic[best3rd_ind[0]] = best3rd_ind[1]
            x_preds, probs2 = self.get_k_probs(x_inds, seq_size, choices=dic)
            words_pred.append(''.join(itemgetter(*x_preds)(self.i2c))[:-1])

        final_preds = [last_word + w for w in words_pred]
        final_lengths = [len(w) for w in words_pred]
        return final_preds, final_lengths

    def interactive_word_predictor(self):
        self.eval_mode()
        print("Interactive word predictor session started...")

        while True:
            inp_string = input()
            if inp_string.strip().lower() == "exit":
                break

            pred_words, _ = self.predictions(inp_string)
            for w in pred_words[:-1]:
                print(w, end=" | ")
            print(pred_words[-1])

    def clean_line_test(self, line):
        if self.fake_test:
            return (''.join([c for c in line if (c in CAP_CHARS)])).split()[:20]
        else:
            return line.split()[:20]

    def create_datasets(self, new_train_filename, new_test_filename):
        num_lines_test = 20000

        with open(new_train_filename, 'w+', encoding='utf8') as train_file, open(new_test_filename, 'w+', encoding='utf8') as test_file, open(self.filename, encoding='utf8', errors='ignore') as data_file:
            all_lines = []
            for line in data_file:
                all_lines.append(line)

            shuffle(all_lines)
            for line in all_lines[:num_lines_test]:
                test_file.write(line)

            for line in all_lines[num_lines_test:]:
                train_file.write(line)

    def is_match(self, w, inp):
        """
            Checks if word is in predictions and if so returns length of predicted sequence.
        """
        pred_words, pred_lengths = self.predictions(inp)
        for pr in zip(pred_words, pred_lengths):
            if w == pr[0]:
                return pr[1]

        return 0

    def evaluate(self, test_filename):
        self.eval_mode()

        lines = []
        with open(test_filename, encoding='utf8', errors='ignore') as f:
            for line in f:
                lines.append(self.clean_line_test(line))

        shuffle(lines)
        lines = lines[:2000]

        tot_saved = 0
        tot_chars = 0

        for line in tqdm(lines, desc="Evaluating"):
            inp = ' '.join(line)
            pointer = 0
            start = 0

            for i, w in enumerate(line):
                tot_chars += len(w)

                if i > 0:
                    saved = self.is_match(w, inp[start:pointer])
                    if saved > 0:
                        tot_saved += saved
                        pointer += saved + 1
                        continue

                for j, char in enumerate(w):
                    pointer += 1
                    cur_char = char.lower() if self.model_lower else char
                    if not (cur_char in self.CHARS):
                        pointer += (len(w) - j - 1)
                        start = pointer
                        break
                    saved = self.is_match(w, inp[start:pointer])
                    if saved > 0:
                        tot_saved += saved
                        pointer += saved
                        break

                pointer += 1
        
        print("Proportion of saved keystrokes: %.6f" % (tot_saved / tot_chars))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word predictor')
    parser.add_argument('-lw', '--lower-case', action='store_true',
                        help='Use lower case model')
    parser.add_argument('-ld', '--load_model', action='store_true',
                        help='Load stored model')
    parser.add_argument('-tr', '--train', action='store_true',
                        help='Train model')
    parser.add_argument('-e', '--epochs', default=5, type=int,
                        help='Number of epochs to train model')
    parser.add_argument('-ev', '--evaluate', action='store_true',
                        help='Evaluate model')
    parser.add_argument('-wp', '--interactive-word-predictor', action='store_true')
    parser.add_argument('-ft', '--fake-test', action='store_true', help="Evaluate with cleaned test data")
    parser.add_argument('-news', '--train_news', action='store_true', help="Whether to use model trained on news")
    parser.add_argument('-tg', '--text-generation', action='store_true', help='Run interactive text generation mode')
    parser.add_argument('-lr', '--learning-rate', default=0.002, help='Learning rate')
    parser.add_argument('-f', '--train-file', default="./data/train.txt", help='Training data filename')
    parser.add_argument('-tf', '--test-file', default="./data/test_blog.txt", help='Test data filename')
    args = parser.parse_args()

    nlm = NLM(args.train_file, model_news=args.train_news, fake_test=args.fake_test, learning_rate=args.learning_rate, model_lower=args.lower_case)
    if args.load_model:
        nlm.load_model()
    if args.train:
        nlm.train(args.epochs)
    if args.text_generation:
        nlm.interactive()
    if args.interactive_word_predictor:
        nlm.interactive_word_predictor()
    if args.evaluate:
        nlm.evaluate(args.test_file)
