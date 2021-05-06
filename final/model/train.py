import torch
import torch.nn as nn
from torch import optim
import random
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model_cnn import NBOW
# from model_attention import AttentionModel
from model_attention_alt import WordAttention
from vocab import Vocab, WSBDataStock, WSBDataScore, load_csv, create_vocab
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef
torch.autograd.set_detect_anomaly(True)

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
    parser.add_argument("--model_type", type=str, choices=["nbow", "attention"], default="nbow")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    args = parser.parse_args()
    return args

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
    f1_scores = []
    m_coefs = []
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
        accuracy, f1, m_coef = eval_network(dev, net, use_gpu=use_gpu, num_classes=num_classes, batch_size=batchSize, device=device)
        eval_accuracy.append(accuracy)
        f1_scores.append(f1)
        m_coefs.append(m_coef)


    print("Finished Training")
    return epoch_losses, eval_accuracy, f1_scores, m_coefs

def plot_accuracy(accuracy_results, model_name):
    # min_accs, accs, max_accs = accuracy_results
    plt.figure()
    plt.plot(accuracy_results, 'ro-')
    # plt.plot(min_accs, 'bo-', label="min_accuracy")
    # plt.plot(max_accs, 'go-', label="max_accuracy")
    plt.title("Reddit WSB Sentiment Accuracy: " + model_name)
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    # plt.legend()
    plt.show()

def main():
    args = parse_args()
    wsb_file_path = args.wsb_csv_file
    device = args.device
    save_model = args.save_model
    model_type = args.model_type
    wsb_data = load_csv(wsb_file_path)
    vocab = create_vocab(wsb_data['title'].values)
    data = WSBDataScore(wsb_file_path, dataframe=wsb_data, vocab=vocab)
    print(vocab.get_vocab_size())

    if device == "gpu":
        device = torch.device('cuda:0')
        split_point = int(len(wsb_data)*0.9)
        X_train = data.XwordList[0:split_point]
        Y_train = data.Y[0:split_point]
        X_dev = data.XwordList[split_point:]
        Y_dev = data.Y[split_point:]

        n_classes = len(set(Y_dev))
        print(n_classes)

        dev_data = (X_dev, Y_dev)

        if model_type == "nbow":
            model = NBOW(vocab.get_vocab_size(), DIM_EMB=310, NUM_CLASSES=n_classes).cuda()
        elif model_type == "attention":
            model = WordAttention(vocab_size=vocab.get_vocab_size(), hidden_size=350, atten_size=150, num_classes=n_classes).cuda()

        # X = train_data.XwordList
        # Y = train_data.Y
        losses, accuracies, f1_scores, m_coefs = train_network(model, X_train, Y_train, 10, dev_data, lr=0.0025, batchSize=50, num_classes=n_classes, device = device)
        print(accuracies)

    else:
        split_point = int(len(wsb_data)*0.9)
        X_train = data.XwordList[0:split_point]
        Y_train = data.Y[0:split_point]
        X_dev = data.XwordList[split_point:]
        Y_dev = data.Y[split_point:]
        dev_data = (X_dev, Y_dev)
        n_classes = len(set(Y_dev))
        print(n_classes)
        device = torch.device('cpu')
        # nbow_model = NBOW(train_data.vocab.get_vocab_size(), DIM_EMB=300)
        if model_type == "nbow":
            model = NBOW(vocab.get_vocab_size(), DIM_EMB=300, NUM_CLASSES=n_classes)
        elif model_type == "attention":
            model = WordAttention(vocab_size=vocab.get_vocab_size(), hidden_size=350, atten_size=150, num_classes=n_classes)


        # X = train_data.XwordList
        # Y = train_data.Y
        losses, accuracies, f1_scores, m_coefs = train_network(model, X_train, Y_train, 10, dev_data, batchSize=150, num_classes=n_classes, device = device)
        print(accuracies)

    # plot_accuracy(accuracies, model_type + "-Sentiment WSB")
    np.save("results/"+ model_type  +"-sentiment-accuracy-wsb.npy", np.array(accuracies))
    np.save("results/" + model_type + "-sentiment-f1-wsb.npy", np.array(f1_scores))
    np.save("results/" + model_type + "-sentiment-m_coef-wsb.npy", np.array(m_coefs))
    if save_model:
        torch.save(model.state_dict(), "saved_models/" + model_type + ".pth")


if __name__ == '__main__':
    main()
