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
        self.NUM_CLASSES= NUM_CLASSES
        self.vocab_size = VOCAB_SIZE
        self.embedding = nn.Embedding(VOCAB_SIZE, DIM_EMB)
        self.conv1 = nn.Conv1d(DIM_EMB, 700, 3)
        self.conv2 = nn.Conv1d(DIM_EMB, 700, 4)
        self.conv3 = nn.Conv1d(DIM_EMB, 700, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.55)
        self.linear1 = nn.Linear(2100, NUM_CLASSES)
        # self.linear2 = nn.Linear(10, NUM_CLASSES)
        # self.log_softmax = nn.LogSoftmax(dim=1)
        #self.logSoftmax = nn.LogSoftmax(dim=0)

    def forward(self, X):
        out1  = self.embedding(X)
        out1 = self.dropout(out1)
        out1 = out1.reshape((out1.shape[0], out1.shape[2], out1.shape[1]))

        conv_out_1 = self.relu(self.conv1(out1))
        conv_out_2 = self.relu(self.conv2(out1))
        conv_out_3 = self.relu(self.conv3(out1))

        max_1 = torch.max(conv_out_1, dim=2)[0]
        max_2 = torch.max(conv_out_2, dim=2)[0]
        max_3 = torch.max(conv_out_3, dim=2)[0]
        lin_input = torch.cat((max_1, max_2, max_3), dim=1)
        drop_out = self.dropout(lin_input)
        lin_out_1 = self.linear1(drop_out)
        output = lin_out_1
        # lin_out_2 = self.linear2(lin_out_1)
        # output = self.log_softmax(lin_out_1)
        #return raw logits since we are using nn.CrossEntropyLoss
        return output
        # embedded = self.embedded(X)
        # embedded = embedded.unsqueeze(0)
        # conv = self.convolution(embedded)
        # pool = self.pool(conv)
        # dropout = self.dropout(pool)
        #
        # linear = self.linear(dropout)
        # relu = self.relu(linear)
        # #output = self.logSoftmax(relu)

        # return relu


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



def main():
    train = WSBdata()
    dev = WSBdata(train=False)
    nbow = NBOW(train.vocab.get_vocab_size())
    Train(nbow, train.XwordList, (train.Y + 1.0) / 2.0, 5, dev)


if __name__ == "__main__":
    main()
