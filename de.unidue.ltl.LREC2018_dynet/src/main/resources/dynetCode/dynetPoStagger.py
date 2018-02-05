from sys import argv
from collections import Counter, defaultdict
from itertools import count
import random
import argparse

import dynet as dy
import numpy as np

if  __name__ =='__main__':

	parser = argparse.ArgumentParser(description="LREC Keras")
	parser.add_argument("--trainData", nargs=1, required=True)
	parser.add_argument("--trainOutcome", nargs=1, required=True)
	parser.add_argument("--testData", nargs=1, required=True)
	parser.add_argument("--testOutcome", nargs=1, required=True)    
	parser.add_argument("--embedding", nargs=1, required=True)    
	parser.add_argument("--maxLen", nargs=1, required=True)
	parser.add_argument("--predictionOut", nargs=1, required=True)
	parser.add_argument("--seed", nargs=1, required=True)    
	
	# DyNet parameters must be declared otherwise an exception is thrown
	parser.add_argument("--dynet-seed", nargs=1, required=False)    
	parser.add_argument("--dynet-mem", nargs=1, required=False)    
	parser.add_argument("--dynet-devices", nargs=1, required=False)    
	parser.add_argument("--dynet-autobatch", nargs=1, required=False)    
    
	args = parser.parse_args()
    
	trainSeq = args.trainData[0]
	trainLabel = args.trainOutcome[0]
	testSeq = args.testData[0]
	testLabel = args.testOutcome[0]
	embedding = args.embedding[0]
	predictionOut = args.predictionOut[0]
	seed = args.seed[0]
	
	np.random.seed(int(seed))

class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).next)
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.items()}
    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).__next__)
        for sent in corpus:
            [w2i[word] for word in sent]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())

def load_embeddings_file(file_name, sep=" ",lower=False):
    """
    load embeddings file
    """
    emb={}
    for line in open(file_name, errors='ignore', encoding='utf-8'):
        try:
            fields = line.strip().split(sep)
            vec = [float(x) for x in fields[1:]]
            word = fields[0]
            if lower:
                word = word.lower()
            emb[word] = vec
        except ValueError:
            print("Error converting: {}".format(line))

    print("loaded pre-trained embeddings (word->emb_vec) size: {} (lower: {})".format(len(emb.keys()), lower))
    return emb, len(emb[word])

def read(data, labels):
    """
    Read a POS-tagged file where each line is of the form "word1/tag2 word2/tag2 ..."
    Yields lists of the form [(word1,tag1), (word2,tag2), ...]
    """
    sents = []
    
    f = open(data)
    sentences=f.readlines()
    f.close()
    
    f = open(labels)
    labels=f.readlines()
    f.close()
    
    for seq, labels in zip(sentences,labels):
        s = seq.strip().split()
        l = labels.strip().split()
        
        sent=[]
        for se, le in zip(s,l):
        	sent.append((se,le))
        sents.append(sent)
    
    return sents


train=list(read(trainSeq, trainLabel))
dev=list(read(testSeq, testLabel))
words=[]
tags=[]
for sent in (train+dev):
    for w,p in sent:
        words.append(w)
        tags.append(p)
words.append("_UNK_")

vw = Vocab.from_corpus([words]) 
vt = Vocab.from_corpus([tags])

nwords = vw.size()
ntags  = vt.size()

# DyNet Starts

model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)

NUM_LAYERS = 1

embeddings, emb_dim = load_embeddings_file(embedding)
# init model parameters and initialize them
WORDS_LOOKUP = model.add_lookup_parameters((nwords, emb_dim), init=dy.GlorotInitializer())
init = 0
UNK_vec = np.random.rand(emb_dim)

for word in vw.w2i.keys():
    # for those words we have already in w2i, update vector, otherwise add to w2i (since we keep data as integers)
    if word in embeddings.keys():
        #print("found ["+word+"] in w2i")
        WORDS_LOOKUP.init_row(vw.w2i[word], embeddings[word])
    else:
        WORDS_LOOKUP.init_row(vw.w2i[word], UNK_vec)


p_t1  = model.add_lookup_parameters((ntags, 30))

# MLP on top of biLSTM outputs 100 -> 32 -> ntags
pH = model.add_parameters((75, 50+50))
pO = model.add_parameters((ntags, 75))

# word-level LSTMs
# input dimension is word-vector+fwd.CharVector+bckwd.CharVector-Length
fwdRNN = dy.LSTMBuilder(1, emb_dim, 50, model)
bwdRNN = dy.LSTMBuilder(1, emb_dim, 50, model)

def build_tagging_graph(words):
    #dy.renew_cg()
    # parameters -> expressions
    H = dy.parameter(pH)
    O = dy.parameter(pO)

    # initialize the RNNs
    f_init = fwdRNN.initial_state()
    b_init = bwdRNN.initial_state()

    wembs = [WORDS_LOOKUP[vw.w2i[w]] for w in words]
    #wembs = [dy.dropout(w,0.2) for w in wembs]

    # feed word vectors into biLSTM
    fw_exps = f_init.transduce(wembs)
    bw_exps = b_init.transduce(reversed(wembs))

    # biLSTM states
    bi_exps = [dy.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]

    # feed each biLSTM state to an MLP
    exps = []
    for x in bi_exps:
        r_t = O*(dy.tanh(H * x))
        exps.append(r_t)

    return exps

def sent_loss(words, tags):
    vecs = build_tagging_graph(words)
    errs = []
    for v,t in zip(vecs,tags):
        tid = vt.w2i[t]
        err = dy.pickneglogsoftmax(v, tid)
        errs.append(err)
    return dy.esum(errs)

def tag_sent(words):
    vecs = build_tagging_graph(words)
    vecs = [dy.softmax(v) for v in vecs]
    probs = [v.npvalue() for v in vecs]
    tags = []
    for prb in probs:
        tag = np.argmax(prb)
        tags.append(vt.i2w[tag])
    return zip(words, tags)

def evaluate():
    good = bad = 0.0
    gold_out = []
    pred_out = []
    words_out = []
    for sent in dev:
        words = [w for w, t in sent]
        golds = [t for w, t in sent]
        tags = [t for w, t in tag_sent(words)]

        gold_out.append(golds)
        pred_out.append(tags)
        words_out.append(words)
        
        for go, gu in zip(golds, tags):
            if go == gu:
                good += 1
            else:
                bad += 1
    print("Test-Accuracy: ", good / (good + bad) * 100)
    return words_out, gold_out, pred_out


batch=[]
for ITER in range(30):
    total_loss=0.0
    random.shuffle(train)
    for i,s in enumerate(train,1):
        
        words = [w for w,t in s]
        golds = [t for w,t in s]
        
        loss =  sent_loss(words, golds)
        batch.append(loss)
        if len(batch) == 1:
            l = dy.esum(batch)
            total_loss+=l.value()
            l.backward()
            trainer.update()
            dy.renew_cg()
            batch=[]

    if len(batch) > 0:
        l = dy.esum(batch)
        total_loss+=l.value()
        l.backward()
        trainer.update()
        dy.renew_cg()
        batch=[]
    print ("epoch %r finished" % ITER)
    evaluate()
print("Finish")
w,g,p = evaluate()

with open(predictionOut, mode="w") as out:
    out.write("#Gold\tPrediction\n")
    for w_sent, g_sent,p_sent in zip(w,g,p):
        assert(len(p_sent) == len(g_sent) == len(w_sent))
        for i in range(0, len(p_sent)):
            out.write(g_sent[i] + "\t" + p_sent[i]+"\n")
        out.write("\n")




