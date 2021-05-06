import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model_cnn import NBOW
# from model_attention import HierarchicalAttentionNetwork
from model_attention_alt import  WordAttention
from vocab import Vocab, WSBDataStock, load_csv, create_vocab
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef

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
    parser.add_argument("--label_type", type=str, choices=["up_down", "volitility"])
    args = parser.parse_args()
    return args

pad_idx = 0

def eval_network(data, net, use_gpu=False, batch_size=25, num_classes=2, device=torch.device('cpu')):
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
        # if isinstance(net, HierarchicalAttentionNetwork):
        #     # doc_lengths = torch.ones(batchSize).type(torch.LongTensor).to(device)
        #     # sentence_lengths = torch.sum(batch_x != pad_idx, axis=1).type(torch.LongTensor).to(device)
        #     # sentence_lengths = sentence_lengths.reshape((sentence_lengths.shape[0], 1))
        #     # batch_x = batch_x.reshape((batch_x.shape[0], 1, batch_x.shape[1]))
        #     batch_y_hat = net.forward(batch_x)
        # else:
        #     batch_y_hat = net.forward(batch_x)
        predictions = batch_y_hat.argmax(dim=1)
        batch_predictions.append(predictions)

    predictions = torch.cat(batch_predictions)
    predictions = predictions.type(torch.float64)
    Y_tensor = torch.tensor(Y, device=device)
    num_correct = float((predictions == Y_tensor).sum())
    accuracy = num_correct/len(Y)

    predictions = predictions.cpu().numpy()
    if num_classes == 2:
        f1 = f1_score(Y, predictions)
    else:
        f1 = f1_score(Y, predictions, average="weighted")
    m_coef = matthews_corrcoef(Y, predictions)


    print("Eval Accuracy: %s" % accuracy)
    print("Eval F1 Score: %s" % f1)
    print("Eval Matthew Correlation %s" % m_coef)
    return accuracy, f1, m_coef
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


def train_network(net, X, Y, num_epochs, dev, lr=0.001, batchSize=50, use_gpu=False, num_classes=2, device=torch.device('cpu')):

    print("Start Training!")
    #TODO: initialize optimizer.
    optimizer = optim.Adam(net.parameters(), lr=lr)
    epoch_losses = []
    eval_accuracy = []
    f1_scores = []
    m_coefs_scores = []
    for epoch in range(num_epochs):
        num_correct = 0
        total_loss = 0.0
        net.train()   #Put the network into training model
        for batch in tqdm(range(0, len(X), batchSize), leave=False):
            batch_x = X[batch:batch + batchSize]
            batch_labels = Y[batch:batch + batchSize]
            batch_x = pad_batch_input(batch_x, device=device)
            batch_onehot_labels = convert_to_onehot(batch_labels, NUM_CLASSES=num_classes, device=device)
            optimizer.zero_grad()
            batch_y_hat = net.forward(batch_x)
            # if isinstance(net, HierarchicalAttentionNetwork):
            #     # doc_lengths = torch.ones(batchSize).type(torch.LongTensor).to(device)
            #     # sentence_lengths = torch.sum(batch_x != pad_idx, axis=1).type(torch.LongTensor).to(device)
            #     # sentence_lengths = sentence_lengths.reshape((sentence_lengths.shape[0], 1))
            #     # batch_x = batch_x.reshape((batch_x.shape[0], 1, batch_x.shape[1]))
            #     batch_y_hat = net.forward(batch_x)
            # else:
            #     batch_y_hat = net.forward(batch_x)
            batch_losses = torch.neg(batch_y_hat)*batch_onehot_labels #cross entropy loss
            loss = batch_losses.mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())

        epoch_losses.append(total_loss)
        net.eval()    #Switch to eval mode
        print(f"loss on epoch {epoch} = {total_loss}")
        accuracy, f1, m_coef = eval_network(dev, net, use_gpu=use_gpu, batch_size=batchSize, num_classes=num_classes, device=device)
        eval_accuracy.append(accuracy)
        f1_scores.append(f1)
        m_coefs_scores.append(m_coef)

    print("Finished Training")
    return epoch_losses, eval_accuracy, f1_scores, m_coefs_scores

def plot_accuracy(accuracy_results, model_name):
    plt.figure()
    plt.plot(accuracy_results, 'ro-')
    plt.title(model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.show()


def plot_scores(scores, model_name, label_name):
    plt.figure()
    plt.plot(scores, 'ro-')
    plt.title(model_name)
    plt.xlabel("Epochs")
    plt.ylabel(label_name)
    plt.show()



def nbow(device_type, save_model, label_type):
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
    n_classes = 2
    if path == "../data/reddit_wsb.csv":
        train_data = WSBDataStock(path, dataframe=train_df, vocab=vocab, label_type=label_type, train=True)
        print("load dev data")
        dev_data = WSBDataStock(path, dataframe=dev_df, vocab=vocab, label_type=label_type, train=False)
        if label_type == "up_down":
            n_classes = 2
        else:
            n_classes = 3
        print(train_data.vocab.get_vocab_size())
    # if path == "../data/twitter_data.csv":
    #   train_data = TwitterData(path, dataframe=train_df, vocab=vocab, train=True)
    #   print("load dev data")
    #   dev_data = TwitterData(path, dataframe=dev_df, vocab=vocab, train=False)
    #   print(train_data.vocab.get_vocab_size())
    print("num classes")
    print(n_classes)
    if device_type == "gpu":
        device = torch.device('cuda:0')
        nbow_model = NBOW(train_data.vocab.get_vocab_size(), DIM_EMB=350, NUM_CLASSES=n_classes).cuda()
        X = train_data.XwordList
        Y = train_data.Y
        dev_data = (dev_data.XwordList, dev_data.Y)
        losses, accuracies, f1_scores, m_coef_scores = train_network(nbow_model, X, Y, 10, dev_data, batchSize=100, lr=0.002, num_classes=n_classes, device = device)
        print(accuracies)
        print(f1_scores)
        print(m_coef_scores)
        # train_model(nbow_model, X, Y, 1, dev_data, use_cuda=True)
    else:
        device = torch.device('cpu')
        nbow_model = NBOW(train_data.vocab.get_vocab_size(), DIM_EMB=350, NUM_CLASSES=n_classes)
        X = train_data.XwordList
        Y = train_data.Y
        dev_data = (dev_data.XwordList, dev_data.Y)
        losses, accuracies, f1_scores, m_coef_scores = train_network(nbow_model, X, Y, 10, dev_data, batchSize=100, num_classes=n_classes, device = device)
        print(accuracies)
        print(f1_scores)
        print(m_coef_scores)
        # train_model(nbow_model, X, Y, 1, dev_data, use_cuda=False)

    if save_model:
        torch.save(nbow_model.state_dict(), "nbow.pth")
    plot_accuracy(accuracies, "CNN WSB Stock Data Accuracy, Stock " + label_type)
    plot_scores(f1_scores, "CNN WSB Stock Data F1 Scores, Stock " + label_type, "F1 Scores")
    plot_scores(m_coef_scores, "CNN WSB Stock Data M Coef, Stock " + label_type, "Matthew Coef.")
    np.save("results/cnn-sentiment-accuracy-wsb-stock-" + label_type + ".npy", np.array(accuracies))
    np.save("results/cnn-sentiment-f1-wsb-stock-" + label_type + ".npy", np.array(f1_scores))
    np.save("results/cnn-sentiment-m_coef-wsb-stock-" + label_type + ".npy", np.array(m_coef_scores))


def attention(device_type, save_model, label_type):
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
    n_classes = 2
    if path == "../data/reddit_wsb.csv":
        train_data = WSBDataStock(path, dataframe=train_df, label_type=label_type, vocab=vocab, train=True)
        print("load dev data")
        dev_data = WSBDataStock(path, dataframe=dev_df, label_type=label_type, vocab=vocab, train=False)
        print("vocab size")
        print(train_data.vocab.get_vocab_size())
        if label_type == "up_down":
            n_classes = 2
        else:
            n_classes = 3
    # if path == "../data/twitter_data.csv":
    #     n_classes = 2
    #     train_data = TwitterData(path, dataframe=train_df, vocab=vocab, train=True)
    #     print("load dev data")
    #     dev_data = TwitterData(path, dataframe=dev_df, vocab=vocab, train=False)
    #     print("vocab size")
    #     print(train_data.vocab.get_vocab_size())

    print("num classes")
    print(n_classes)
    if device_type == "gpu":
        device = torch.device('cuda:0')
        # attn_model = HierarchicalAttentionNetwork(vocab_size=train_data.vocab.get_vocab_size(), batch_size=100, num_classes=n_classes).cuda()
        attn_model = WordAttention(vocab_size=train_data.vocab.get_vocab_size(), hidden_size=350, atten_size=150, num_classes=n_classes).cuda()
        X = train_data.XwordList
        Y = train_data.Y
        dev_data = (dev_data.XwordList, dev_data.Y)
        losses, accuracies, f1_scores, m_coef_scores = train_network(attn_model, X, Y, 10, dev_data, batchSize=100, lr=0.002, device = device, num_classes=n_classes)
        print(accuracies)
        print(f1_scores)
        print(m_coef_scores)
        # train_model(attn_model, X, Y, 1, dev_data, use_cuda=True)
    else:
        device = torch.device('cpu')
        # attn_model = HierarchicalAttentionNetwork(vocab_size=train_data.vocab.get_vocab_size(), batch_size=100, num_classes=n_classes)
        attn_model = WordAttention(vocab_size=train_data.vocab.get_vocab_size(), num_classes=n_classes)
        X = train_data.XwordList
        Y = train_data.Y
        dev_data = (dev_data.XwordList, dev_data.Y)
        losses, accuracies, f1_scores, m_coef_scores = train_network(attn_model, X, Y, 10, dev_data, batchSize=100, device = device, num_classes=n_classes)
        print(accuracies)
        print(f1_scores)
        print(m_coef_scores)
        # train_model(attn_model, X, Y, 1, dev_data, use_cuda=False)

    if save_model:
        torch.save(attn_model.state_dict(), "hierarchattention.pth")

    plot_accuracy(accuracies, "Hierarchical Attention WSB Stock Data Accuracy, Stock " + label_type)
    plot_scores(f1_scores, "Hierarchical Attention WSB Stock Data F1 Score, Stock " + label_type, "F1 Score")
    plot_scores(m_coef_scores, "Hierarchical Attention WSB Stock Data M Coef, Stock " + label_type, "Matthew Coef")
    np.save("results/attention-sentiment-accuracy-wsb-stock-" + label_type + ".npy", np.array(accuracies))
    np.save("results/attention-sentiment-f1-wsb-stock-" + label_type + ".npy", np.array(f1_scores))
    np.save("results/attention-sentiment-m_coef-wsb-stock-" + label_type + ".npy", np.array(m_coef_scores))

def main():
    args = parse_args()
    model = args.model_type
    device_type = args.device
    save_model = args.save_model
    label_type = args.label_type
    if model == "attention":
        attention(device_type, save_model, label_type)
    elif model == "nbow":
        nbow(device_type, save_model, label_type)


if __name__ == '__main__':
    main()
