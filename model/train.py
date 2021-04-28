import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np
import torch.nn.functional as F
import tqdm
import argparse
from model import NBOW
from vocab import Vocab, WSBdata, load_csv, create_vocab


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wsb_csv_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
    )
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    args = parser.parse_args()
    return args


def EvalNet(data, net, use_cuda=False):
    num_correct = 0
    X = data.XwordList
    Y = data.Y
    for i in range(len(X)):
        if use_cuda:
            input = X[i].cuda()
        else:
            input = X[i]
        logProbs = net.forward(input)
        pred = torch.argmax(logProbs).item()
        if pred == Y[i]:
            num_correct += 1
    print("Accuracy: %s" % (float(num_correct) / float(len(X))))

def SavePredictions(data, outFile, net):
    fOut = open(outFile, 'w')
    for i in range(len(data.XwordList)):
        logProbs = net.forward(data.XwordList[i])
        pred = torch.argmax(logProbs)
        fOut.write(f"{data.XfileList[i]}\t{pred}\n")

def train_model(net, X, Y, n_iter, dev, num_classes=5, use_cuda=False):
    print("Start Training!")
    #TODO: initialize optimizer.
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(n_iter):
        num_correct = 0
        total_loss = 0.0
        net.train()   #Put the network into training mode
        for i in tqdm(range(len(X))):
            x_sample = X[i]
            if use_cuda:
                x_sample = x_sample.cuda()

            Y_sample = int(Y[i])

            net.zero_grad()
            logProbs = net.forward(x_sample)

            input_ = logProbs.view(1,-1)
            if use_cuda:
                target = torch.LongTensor(np.array([Y_sample])).cuda()
            else:
                target = torch.LongTensor(np.array([Y_sample]))

            loss = loss_func(input_, target)
            loss.backward()
            total_loss += float(loss.detach().item())
            optimizer.step()


        net.eval()    #Switch to eval mode
        print(f"loss on epoch {epoch} = {total_loss}")
        EvalNet(dev, net)

def main():
    args = parse_args()
    wsb_file_path = args.wsb_csv_file
    device = args.device
    save_model = args.save_model
    wsb_data = load_csv(wsb_file_path)
    vocab = create_vocab(wsb_data['title'].values)

    split_point = int(len(wsb_data)*0.9)
    train_df = wsb_data[0:split_point]
    dev_df = wsb_data[split_point:]


    train_data = WSBdata(wsb_file_path, dataframe=train_df, vocab=vocab train=True)
    dev_data = WSBdata(wsb_file_path, dataframe=dev_df, vocab=vocab, train=False)
    if device == "gpu":
        nbow_model = NBOW(train.vocab.get_vocab_size(), DIM_EMB=350).cuda()
        X = train_data.XwordList
        Y = train_data.Y
        train_model(nbow_model, X, Y, 10, dev_data, use_cuda=True)
    else:
        nbow_model = NBOW(train.vocab.get_vocab_size(), DIM_EMB=350)
        X = train_data.XwordList
        Y = train_data.Y
        train_model(nbow_model, X, Y, 10, dev_data, use_cuda=False)

    if save_model:
        torch.save(nbow_model.state_dict(), "saved_models/nbow.pth")
