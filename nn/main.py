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
from random import shuffle
import string
from os.path import isfile

CHARS = list(" abcdefghijklmnopqrstuvwxyz.'")
CHARS1 = list(" abcdefghijklmnopqrstuvwxyz'")
CHARS_CAP = list(" abcdefghijklmnopqrstuvwxyz.'ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#CHARS = [' ', '’', '—', '–', '“', '”', 'é', '‘'] + list(string.punctuation) + list(string.ascii_letters) + list(string.digits)

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

class NLM:
    def __init__(self, data_filename, size_batch=128, hidden_size=256, char_emb_size=16, char_map=CHARS):
        #torch.manual_seed(99)
        #np.random.seed(99)
        self.filename = data_filename
        self.vocab_filename = "./data/vocab"
        self.test_filename = "./data/test"
        self.n_batch = size_batch
        self.n_hidden = hidden_size
        self.n_char_emb = char_emb_size
        self.n_classes = len(char_map)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.seq_size = 129 # Real sequence size becomes [self.seq_size - 1]
        self.n_train = 1200000 # Total n. of training samples to use
        self.n_mem = 300000 # Max n. of samples to load into memory at once        

        self.c2i = {c: i for i, c in enumerate(char_map)}
        self.i2c = char_map
        self.char_emb = nn.Embedding(len(char_map), char_emb_size)
        self.neigh_model = None
        
        self.model = nn.LSTM(input_size=char_emb_size, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.2).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.final_linear = nn.Linear(self.n_hidden, self.n_classes).to(self.device)
        params = list(self.char_emb.parameters()) + list(self.model.parameters()) + list(self.final_linear.parameters())
        self.optimizer = optim.Adam(params, lr=0.002)

    def clean_line(self, line):
        line = line.replace('!', '.')
        line = line.replace('?', '.')
        #line = ''.join([c for c in line if (c in CHARS)])
        line = ''.join([c for c in line.lower() if (c in CHARS)])
        return line

    def text_gen(self):
        with open(self.filename, encoding='utf8', errors='ignore') as f:
            for line in f:
                yield self.clean_line(line)

    def clean_line_vocab(self, line):
        line = ''.join([c for c in line if (c in CHARS_CAP)])
        return line

    def text_gen_vocab(self):
        with open(self.filename, encoding='utf8', errors='ignore') as f:
            for line in f:
                yield self.clean_line_vocab(line)

    def read_vocab(self):
        self.vocab = set()
        if isfile(self.vocab_filename):
            print("Reading vocab file...", end=" ")
            with open(self.vocab_filename) as f:
                for w in f:
                    self.vocab.add(w.strip())

            print("done.")
        else:
            print("Building vocab...", end=" ")
            text = self.text_gen_vocab()
            for s in text:
                for w in s.split():
                    _w = w
                    if _w.endswith("''"):
                        _w = _w[:-2]
                    if len(_w) >= 2:
                        if _w.endswith("'") and (_w[-2] != 's'):
                            _w = _w[:-1]                    
                    if _w.endswith('...'):
                        _w = _w[:-3]
                    if _w.endswith('..'):
                        _w = _w[:-2]
                    if _w.endswith('.'):
                        _w = _w[:-1]
                    if _w.endswith("'s"):
                        _w = _w[:-2]
                    if _w.startswith("'"):
                        _w = _w[1:]

                    self.vocab.add(_w)

            print("done.")
            print("Vocab length: %d" % len(self.vocab))
            print("Write vocab to file? (Y/n):", end=" ")
            ans = input().strip().lower()
            if ans == "y":
                print("Writing...", end=" ")
                self.write_vocab()
                print("done.")


    def write_vocab(self):
        with open(self.vocab_filename, 'w+', encoding='utf8') as f:
            for w in self.vocab:
                f.write('{}\n'.format(w))

    def train_on_batch(self, data, epoch, batch_ind):
        batch_size = self.seq_size * self.n_batch
        pred_batch_size = (self.seq_size-1) * self.n_batch
        for i in tqdm(range(batch_size, len(data), batch_size), desc="Epoch %d (%d/%d)" % (epoch, batch_ind, self.tot_batches)):
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
            clip_grad_norm_(list(self.char_emb.parameters()) + list(self.model.parameters()) + list(self.final_linear.parameters()), 5)
            self.optimizer.step()

            if (self.iterations % 200) == 0:
                #print("Loss: %.6f" % loss.item())
                pass
            self.iterations += 1            


    def train(self, epochs):
        self.tot_batches = self.n_train // self.n_mem
        self.iterations = 0

        for epoch in range(epochs):
            text = self.text_gen()
            count = 0
            cur_batch = 0
            tot_count = 0
            dataset = []
            cur_seq = ""
            sent_count = 0

            for s in text:
                sent_count += 1
                cur_seq += s + ' '
                if len(cur_seq) >= self.seq_size:
                    dataset.extend(list(cur_seq[:self.seq_size]))
                    cur_seq = ""
                    count += 1
                    if count == self.n_mem:
                        cur_batch += 1
                        self.train_on_batch(dataset, epoch+1, cur_batch)
                        dataset.clear()
                        tot_count += count
                        count = 0
                        if tot_count >= self.n_train:
                            break

            print("Num sentences: %d" % sent_count)

    def save_model(self):
        print("Save model? (Y/n): ", end="")
        inp = input()
        if inp.lower().strip() == "y":
            torch.save(self.char_emb.state_dict(), "models/emb_ext_1.pt")
            torch.save(self.model.state_dict(), "models/lstm_ext_1.pt")
            torch.save(self.final_linear.state_dict(), "models/linear_ext_1.pt")
            print("Model saved.")
        else:
            print("No save.")

    def load_model(self):
        self.char_emb.load_state_dict(torch.load("models/emb"))
        self.model.load_state_dict(torch.load("models/lstm"))
        self.final_linear.load_state_dict(torch.load("models/linear"))

    def sample_preds(self, preds, temperature=0.8):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def interactive(self):
        self.char_emb.eval()
        self.model.eval()
        self.final_linear.eval()

        print("Interactive session started...")
        n_predict = 200
        while True:
            inp = input()
            if inp.strip().lower() == "exit":
                break
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
                #x_preds[0] = torch.argmax(pred)
                x_preds[0] = self.sample_preds(probs)

                for i in range(1, n_predict):
                    x_emb = self.char_emb(torch.LongTensor([x_preds[i-1]])).reshape((1, 1, self.n_char_emb)).to(self.device)
                    outputs, h = self.model(x_emb, h)
                    pred = self.final_linear(h[0][1, 0, :])
                    probs = softmax(pred.cpu().tolist())
                    #x_preds[i] = torch.argmax(pred)
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
        inp = list(inp_string.lower())
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

        final_preds = []
        """
        for i, w in enumerate(words_pred):
            pred_word = last_word + w

            if not (pred_word[0] == pred_word[0].upper()):
                cap_pred_word = pred_word[0].upper() + pred_word[1:]
                rmv_end = 0
                if pred_word.endswith(".") or pred_word.endswith("'"):
                    rmv_end = 1
                if pred_word.endswith("'s") or pred_word.endswith("'."):
                    rmv_end = 2
                if pred_word.endswith("'s."):
                    rmv_end = 3

                if rmv_end > 0:
                    w_lower = pred_word[:-rmv_end]
                    w_cap = cap_pred_word[:-rmv_end]
                else:
                    w_lower = pred_word
                    w_cap = cap_pred_word

                capitalize = (w_cap in self.vocab) and not (w_lower in self.vocab)
                if capitalize:
                    pred_word = cap_pred_word

            final_preds.append(pred_word)
            """

        final_preds = [last_word + w for w in words_pred]
        final_lengths = [len(w) for w in words_pred]
        return final_preds, final_lengths

    def interactive_word_predictor(self):
        self.char_emb.eval()
        self.model.eval()
        self.final_linear.eval()

        #self.read_vocab()

        print("Interactive word predictor session started...")

        while True:
            inp_string = input()
            if inp_string.strip().lower() == "exit":
                break

            pred_words, _, _ = self.predictions(inp_string)
            for w in pred_words[:-1]:
                print(w, end=" | ")
            print(pred_words[-1])

    def clean_line_test(self, line):
        return (''.join([c for c in line.lower() if (c in CHARS1)])).split()

    def create_testset(self):
        start_line = 2101225
        num_lines = 200000

        with open(self.test_filename, 'w+', encoding='utf8') as test_file, open(self.filename, encoding='utf8', errors='ignore') as data_file:
            for _ in range(start_line):
                next(data_file)
            i = 0
            for line in data_file:
                test_file.write(line)
                i += 1
                if i == num_lines:
                    break

    def is_match(self, w, inp):
        """
            Checks if word is in predictions and if so returns length of predicted sequence.
        """
        pred_words, pred_lengths = self.predictions(inp)
        for pr in zip(pred_words, pred_lengths):
            if w == pr[0]:
                return pr[1]

        return 0

    def evaluate(self):
        lines = []

        with open(self.test_filename, encoding='utf8', errors='ignore') as f:
            for line in f:
                lines.append(self.clean_line_test(line))

        shuffle(lines)
        lines = lines[:600] # Only used 600 samples for now

        tot_saved = 0
        tot_chars = 0

        for line in tqdm(lines, desc="Evaluating"):
            inp = ' '.join(line)
            pointer = 0

            for i, w in enumerate(line):
                tot_chars += len(w)

                if i > 0:
                    saved = self.is_match(w, inp[:pointer])
                    if saved > 0:
                        tot_saved += saved
                        pointer += saved + 1
                        continue

                for char in w:
                    pointer += 1
                    saved = self.is_match(w, inp[:pointer])
                    if saved > 0:
                        tot_saved += saved
                        pointer += saved
                        break

                pointer += 1
        
        print("Proportion of saved keystrokes: %.6f" % (tot_saved / tot_chars))

filename = "data/news.2010.en.shuffled"

nlm = NLM(filename)
nlm.load_model()
#nlm.train(5)
#nlm.interactive()
#nlm.interactive_word_predictor()
#nlm.save_model()
nlm.evaluate()
