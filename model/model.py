### Model 


import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np
import torch.nn.functional as F 
import tqdm

from vocab import *



class NBOW(nn.Module):
    def __init__(self, VOCAB_SIZE, DIM_EMB=300, NUM_CLASSES=5):
        super(NBOW, self).__init__()
        self.device = False
        self.NUM_CLASSES=NUM_CLASSES

        self.embedded = nn.Embedding(VOCAB_SIZE, DIM_EMB)
        self.convolution = nn.Conv1d(1, 1, kernel_size=3)
        self.pool = nn.MaxPool1d(1)
        self.dropout = nn.Dropout()

        self.linear = nn.Linear(298, NUM_CLASSES)
        self.relu = nn.ReLU()
        #self.logSoftmax = nn.LogSoftmax(dim=0)



    def forward(self, X):
        embedded = self.embedded(X)
        embedded = embedded.unsqueeze(0)
        conv = self.convolution(embedded)
        pool = self.pool(conv)
        dropout = self.dropout(pool)

        linear = self.linear(dropout)
        relu = self.relu(linear)
        #output = self.logSoftmax(relu)
        
        return relu


def EvalNet(data, net):
    num_correct = 0
    Y = (data.Y + 1.0) / 5.0
    X = data.XwordList
    for i in range(len(X)):
        logProbs = net.forward(X[i])
        pred = torch.argmax(logProbs)
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))

def SavePredictions(data, outFile, net):
    fOut = open(outFile, 'w')
    for i in range(len(data.XwordList)):
        logProbs = net.forward(data.XwordList[i])
        pred = torch.argmax(logProbs)
        fOut.write(f"{data.XfileList[i]}\t{pred}\n")

def Train(net, X, Y, n_iter, dev):
    print("Start Training!")
    #TODO: initialize optimizer.
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    loss_func = nn.CrossEntropyLoss()


    num_classes = 5

    for epoch in range(n_iter):
        num_correct = 0
        total_loss = 0.0
        net.train()   #Put the network into training mode
        for i in tqdm(range(len(X))):
            
            x_ = X[i]
            Y_ = int(Y[i])


            net.zero_grad()
            logProbs = net.forward(X[i])

            input_ = logProbs.view(1,-1)
            target = torch.LongTensor([Y_])

            loss = loss_func(input_, target)
            total_loss += loss 

            loss.backward()
            optimizer.step()


        net.eval()    #Switch to eval mode
        print(f"loss on epoch {epoch} = {total_loss}")
        EvalNet(dev, net)


if __name__ == "__main__":
    train = WSBdata()
    dev = WSBdata(train=False)
    nbow = NBOW(train.vocab.get_vocab_size())
    Train(nbow, train.XwordList, (train.Y + 1.0) / 2.0, 5, dev)










