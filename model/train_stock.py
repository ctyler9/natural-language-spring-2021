import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model import NBOW
from model_attention import AttentionModel
from vocab import Vocab, WSBData, WSBDataLarge, load_csv, create_vocab
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--wsb_csv_file",
    #     type=str,
    #     required=True,
    # )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
    )
    parser.add_argument("--model_type", type=str, choices=["nbow", "attention"], default="nbow")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    args = parser.parse_args()
    return args

def eval_network(data, net, use_gpu=False, batch_size=25, device=torch.device('cpu')):
    print("Evaluation Set")
    num_correct = 0
    # Y = (data.labels + 1.0) / 2.0
    X = data[0]
    Y = data[1]
    # X = data.XwordList
    # Y = data.Y
    batch_predictions = []
    for batch in tqdm(range(0, len(X), batch_size), leave=False):
        batch_x = pad_batch_input(X[batch:batch + batch_size], device=device)
        batch_y = torch.tensor(Y[batch:batch + batch_size], device=device)
        batch_y_hat = net.forward(batch_x)
        # if isinstance(net, AttentionModel):
        #     batch_y_hat = batch_y_hat[0]
        predictions = batch_y_hat.argmax(dim=1)
        batch_predictions.append(predictions)
        # num_correct = float((predictions == batch_y).sum())
        # accuracy = num_correct/float(batch_size)
        # batch_accuracies.append(accuracy)
    predictions = torch.cat(batch_predictions)
    predictions = predictions.type(torch.float64)
    Y_tensor = torch.tensor(Y, device=device)
    num_correct = float((predictions == Y_tensor).sum())
    accuracy = num_correct/len(Y)

    print("Eval Accuracy: %s" % accuracy)
    return accuracy
    # return min_accuracy, accuracy, max_accuracy


def SavePredictions(data, outFile, net):
    fOut = open(outFile, 'w')
    for i in range(len(data.XwordList)):
        logProbs = net.forward(data.XwordList[i])
        pred = torch.argmax(logProbs)
        fOut.write(f"{data.XfileList[i]}\t{pred}\n")

def convert_to_onehot(Y_list, NUM_CLASSES=2, device=torch.device('cpu')):
    Y_onehot = torch.zeros((len(Y_list), NUM_CLASSES), device=device)
    # Y_onehot = [torch.zeros(len(l), NUM_CLASSES) for l in Y_list]
    for i in range(len(Y_list)):
        Y_onehot[i, int(Y_list[i])]= 1.0
    return Y_onehot


def pad_batch_input(X_list, device=torch.device('cpu')):
    X_padded = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(l) for l in X_list], batch_first=True).type(torch.LongTensor).to(device)
    # X_mask   = torch.nn.utils.rnn.pad_sequence([torch.as_tensor([1.0] * len(l)) for l in X_list], batch_first=True).type(torch.FloatTensor)
    return X_padded


def train_network(net, X, Y, num_epochs, dev, lr=0.001, batchSize=50, use_gpu=False, num_classes=5, device=torch.device('cpu')):

    print("Start Training!")
    #TODO: initialize optimizer.
    optimizer = optim.Adam(net.parameters(), lr=lr)
    epoch_losses = []
    eval_accuracy = []
    for epoch in range(num_epochs):
        num_correct = 0
        total_loss = 0.0
        net.train()   #Put the network into training model
        for batch in tqdm(range(0, len(X), batchSize), leave=False):
            batch_tweets = X[batch:batch + batchSize]
            batch_labels = Y[batch:batch + batchSize]
            batch_tweets = pad_batch_input(batch_tweets, device=device)
            batch_onehot_labels = convert_to_onehot(batch_labels, NUM_CLASSES=num_classes, device=device)
            optimizer.zero_grad()
            batch_y_hat = net.forward(batch_tweets)
            # if isinstance(net, AttentionModel): # if its a tuple
            #     batch_y_hat = batch_y_hat[0]
            batch_losses = torch.neg(batch_y_hat)*batch_onehot_labels #cross entropy loss
            loss = batch_losses.mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())

        epoch_losses.append(total_loss)
        net.eval()    #Switch to eval mode
        print(f"loss on epoch {epoch} = {total_loss}")
        accuracy = eval_network(dev, net, use_gpu=use_gpu, batch_size=batchSize, device=device)
        eval_accuracy.append(accuracy)


    print("Finished Training")
    return epoch_losses, eval_accuracy

def plot_accuracy(accuracy_results, model_name):
    # min_accs, accs, max_accs = accuracy_results
    plt.figure()
    plt.plot(accuracy_results, 'ro-')
    # plt.plot(min_accs, 'bo-', label="min_accuracy")
    # plt.plot(max_accs, 'go-', label="max_accuracy")
    plt.title("Reddit WSB+Stock Sentiment Accuracy: " + model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    # plt.legend()
    plt.show()

def attention(device_type, save_model):
    reddit_path = "../data/reddit_wsb.csv"
    twitter_path = "../data/twitter_data.csv"

    path = reddit_path
    device = "gpu"
    save_model = True

    if path == "../data/reddit_wsb.csv":
      data = load_csv(path, "reddit")
      vocab = create_vocab(data['title'].values)
    if path == "../data/twitter_data.csv":
      data = load_csv(path, "twitter")
      vocab = create_vocab(data['Text'].values)

    split_point = int(len(data)*0.9)
    train_df = data[0:split_point]
    dev_df = data[split_point:]

    print("load train data")
    if path == "../data/reddit_wsb.csv":
      n_classes = 2
      train_data = WSBDataLarge(path, dataframe=train_df, vocab=vocab, train=True)
      print("load dev data")
      dev_data = WSBDataLarge(path, dataframe=dev_df, vocab=vocab, train=False)
      print(train_data.vocab.get_vocab_size())
    if path == "../data/twitter_data.csv":
      n_classes = 2
      train_data = TwitterData(path, dataframe=train_df, vocab=vocab, train=True)
      print("load dev data")
      dev_data = TwitterData(path, dataframe=dev_df, vocab=vocab, train=False)
      print(train_data.vocab.get_vocab_size())


    if device_type == "gpu":
        device = torch.device('cuda:0')
        attn_model = AttentionModel(train_data.vocab.get_vocab_size(), DIM_EMB=350, NUM_CLASSES=n_classes).cuda()
        X = train_data.XwordList
        Y = train_data.Y
        losses, accuracies = train_network(attn_model, X, Y, 2, dev_data, batchSize=50, device = device, num_classes=n_classes)
        print(accuracies)
        # train_model(attn_model, X, Y, 1, dev_data, use_cuda=True)
    else:
        device = torch.device('cpu')
        attn_model = AttentionModel(train_data.vocab.get_vocab_size(), DIM_EMB=350, NUM_CLASSES=n_classes)
        X = train_data.XwordList
        Y = train_data.Y
        losses, accuracies = train_network(attn_model, X, Y, 2, dev_data, batchSize=50, device = device, num_classes=n_classes)
        print(accuracies)
        # train_model(attn_model, X, Y, 1, dev_data, use_cuda=False)

    if save_model:
        torch.save(attn_model.state_dict(), "attention.pth")

def main():
    args = parse_args()
    model = args.model_type
    device_type = args.device
    save_model = args.save_model
    if model == "attention":
        attention(device_type, save_model)


if __name__ == '__main__':
    main()
