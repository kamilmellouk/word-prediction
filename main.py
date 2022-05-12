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
import re

CHARS = list(" abcdefghijklmnopqrstuvwxyz.'")

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

class NLM:
    def __init__(self, data_filename, size_batch=256, hidden_size=256, char_emb_size=16, char_map=CHARS):
        self.filename = data_filename
        self.n_batch = size_batch
        self.n_hidden = hidden_size
        self.n_char_emb = char_emb_size
        self.n_classes = len(char_map)
        self.device = "cuda:0"

        self.c2i = {c: i for i, c in enumerate(char_map)}
        self.i2c = char_map
        self.char_emb = nn.Embedding(len(char_map), char_emb_size)
        self.neigh_model = None
        
        self.model = nn.LSTM(input_size=char_emb_size, hidden_size=hidden_size, batch_first=True, num_layers=2, dropout=0.2).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.final_linear = nn.Linear(self.n_hidden, self.n_classes).to(self.device)
        params = list(self.char_emb.parameters()) + list(self.model.parameters()) + list(self.final_linear.parameters())
        self.optimizer = optim.Adam(params, lr=0.002)

    def clean_line(self, line, char_list):
        line = ''.join([c for c in line.lower() if (c in char_list)])
        line = line.replace('!', '.')
        line = line.replace('?', '.')

        return line.split()

    def text_gen(self):
        char_list = [c for c in CHARS]
        char_list.extend(['!', '?'])
        with open(self.filename, encoding='utf8', errors='ignore') as f:
            for line in f:
                yield self.clean_line(line, char_list)

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


    def train(self, epochs):
        self.seq_size = 129 # Real sequence size becomes [self.seq_size - 1]
        self.n_train = 1200000 # Total n. of training samples to use
        self.n_mem = 300000 # Max n. of samples to load into memory at once
        self.tot_batches = self.n_train // self.n_mem

        for epoch in range(epochs):
            text = self.text_gen()
            count = 0
            cur_batch = 0
            tot_count = 0
            dataset = []
            cur_seq = ""

            for s in text:
                cur_seq += ' '.join(s) + ' '
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

    def save_model(self):
        print("Save model? (Y/n): ", end="")
        inp = input()
        if inp.lower().strip() == "y":
            torch.save(self.char_emb.state_dict(), "models/emb")
            torch.save(self.model.state_dict(), "models/lstm")
            torch.save(self.final_linear.state_dict(), "models/linear")
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
        n_predict = 100
        while True:
            inp = input().lower()
            if inp == "exit":
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
        t_probs = []
        if not choices:
            choices = {}

        t = 0
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

            while x_preds[-1] != self.c2i[' ']:
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

        return x_preds, t_probs


    def interactive_word_predictor(self):
        self.char_emb.eval()
        self.model.eval()
        self.final_linear.eval()

        print("Interactive word predictor session started...")
        n_words = 3

        while True:
            inp_string = input()
            if inp_string.strip().lower() == "exit":
                break
            if inp_string.endswith(" "):
                last_word = ""
            else:
                last_word = inp_string.split()[-1]
            inp = list(inp_string.lower())
            seq_size = len(inp)

            if seq_size == 0:
                continue
            elif seq_size == 1:
                x_inds = torch.LongTensor([itemgetter(*inp)(self.c2i)])
            else:
                x_inds = torch.LongTensor(itemgetter(*inp)(self.c2i))

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
                x_preds, probs1 = self.get_k_probs(x_inds, seq_size, dic)
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
                x_preds, probs2 = self.get_k_probs(x_inds, seq_size, dic)
                words_pred.append(''.join(itemgetter(*x_preds)(self.i2c))[:-1])


            for w in words_pred[:-1]:
                print(last_word + w, end=" | ")
            print(last_word + words_pred[-1])

filename = "data/news.2010.en.shuffled"

nlm = NLM(filename)
#nlm.train(5)
nlm.load_model()
nlm.interactive_word_predictor()
#nlm.interactive()
#nlm.save_model()