import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model import NBOW
from vocab import Vocab, WSBData, load_csv, create_vocab


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


# def EvalNet(data, net, use_cuda=False):
#     num_correct = 0
#     X = data.XwordList
#     Y = data.Y
#     for i in range(len(X)):
#         if use_cuda:
#             input = X[i].cuda()
#         else:
#             input = X[i]
#         logProbs = net.forward(input)
#         pred = torch.argmax(logProbs).item()
#         if pred == Y[i]:
#             num_correct += 1
#     print("Accuracy: %s" % (float(num_correct) / float(len(X))))


def eval_network(data, net, use_gpu=False, batch_size=25, device=torch.device('cpu')):
    print("Evaluation Set")
    num_correct = 0
    # Y = (data.labels + 1.0) / 2.0
    X = data.XwordList
    Y = data.Y
    batch_predictions = []
    for batch in tqdm(range(0, len(X), batch_size), leave=False):
        batch_x = pad_batch_input(X[batch:batch + batch_size], device=device)
        batch_y = torch.tensor(Y[batch:batch + batch_size], device=device)
        batch_y_hat = net.forward(batch_x)
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

# def train_model(net, X, Y, n_iter, dev, batch_size=50, num_classes=5, use_cuda=False):
#     print("Start Training!")
#     #TODO: initialize optimizer.
#     optimizer = optim.Adam(net.parameters(), lr=0.001)
#     loss_func = nn.CrossEntropyLoss()
#
#     for epoch in range(n_iter):
#         num_correct = 0
#         total_loss = 0.0
#         net.train()   #Put the network into training mode
#         for i in tqdm(range(len(X))):
#             x_sample = X[i]
#             if use_cuda:
#                 x_sample = x_sample.cuda()
#
#             Y_sample = int(Y[i])
#
#             net.zero_grad()
#             logProbs = net.forward(x_sample)
#
#             input_ = logProbs.view(1,-1)
#             if use_cuda:
#                 target = torch.LongTensor(np.array([Y_sample])).cuda()
#             else:
#                 target = torch.LongTensor(np.array([Y_sample]))
#
#             loss = loss_func(input_, target)
#             loss.backward()
#             total_loss += float(loss.detach().item())
#             optimizer.step()
#
#
#         net.eval()    #Switch to eval mode
#         print(f"loss on epoch {epoch} = {total_loss}")
#         EvalNet(dev, net)

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

    print("load train data")
    train_data = WSBData(wsb_file_path, dataframe=train_df, vocab=vocab, train=True)
    print("load dev data")
    dev_data = WSBData(wsb_file_path, dataframe=dev_df, vocab=vocab, train=False)
    print(train_data.vocab.get_vocab_size())
    if device == "gpu":
        device = torch.device('cuda:0')
        nbow_model = NBOW(train_data.vocab.get_vocab_size(), DIM_EMB=350).cuda()
        X = train_data.XwordList
        Y = train_data.Y
        losses, accuracies = train_network(nbow_model, X, Y, 2, dev_data, batchSize=50, device = device)
        print(accuracies)
        # train_model(nbow_model, X, Y, 1, dev_data, use_cuda=True)
    else:
        device = torch.device('cpu')
        nbow_model = NBOW(train_data.vocab.get_vocab_size(), DIM_EMB=350)
        X = train_data.XwordList
        Y = train_data.Y
        losses, accuracies = train_network(nbow_model, X, Y, 2, dev_data, batchSize=50, device = device)
        print(accuracies)
        # train_model(nbow_model, X, Y, 1, dev_data, use_cuda=False)

    if save_model:
        torch.save(nbow_model.state_dict(), "saved_models/nbow.pth")
if __name__ == '__main__':
    main()
