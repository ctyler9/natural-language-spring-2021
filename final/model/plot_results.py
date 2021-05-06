import numpy as np
import matplotlib.pyplot as plt

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

def plot_results(results_list, labels, colors, title, xlabel, ylabel):
    plt.figure()
    for i in range(len(results_list)):
        result_i = np.load(results_list[i])
        label_i = labels[i]
        color_i = colors[i]
        plt.plot(result_i, color_i, label=label_i)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def main():
    #accuracies
    accuracy_results_sent = ["results/nbow-sentiment-accuracy-wsb.npy", "results/attention-sentiment-accuracy-wsb.npy"]
    f1_results_sent = ["results/nbow-sentiment-f1-wsb.npy", "results/attention-sentiment-f1-wsb.npy"]
    colors_1 = ['ro-', 'bo-']
    labels_1 = ["CNN Model", "Word Attention"]
    title = "Model Accuracy Comparison (Upvote Sentiment)"
    xlabel = "epochs"
    ylabel = "Validation Accuracy"
    plot_results(accuracy_results_sent, labels_1, colors_1, title, xlabel, ylabel)
    plot_results(f1_results_sent, labels_1, colors_1, "Model F1 Comparison (Upvote Sentiment)", xlabel, "F1 Score on Validation Set")


    accuracy_results_ud = ["results/cnn-sentiment-accuracy-wsb-stock-up_down.npy", "results/attention-sentiment-accuracy-wsb-stock-up_down.npy", "results/bert-sentiment-accuracy-wsb-stock-up_down.npy"]
    accuracy_results_vol = ["results/cnn-sentiment-accuracy-wsb-stock-volitility.npy", "results/attention-sentiment-accuracy-wsb-stock-volitility.npy"]
    labels = ["CNN Model", "Word Attention", "Pre-Trained Bert"]
    colors = ['ro-', 'bo-', 'go-']
    acc_title_ud = "Model Accuracy Comparison (Stock Up/Down)"
    acc_title_vol = "Model Accuracy Comparison (Stock Volatility)"
    xlabel = "epochs"
    acc_y = "Validation Accuracy"
    plot_results(accuracy_results_ud, labels, colors, acc_title_ud, xlabel, acc_y)
    plot_results(accuracy_results_vol, labels, colors, acc_title_vol, xlabel, acc_y)

    #f1_scores
    f1_results_ud = ["results/cnn-sentiment-f1-wsb-stock-up_down.npy", "results/attention-sentiment-f1-wsb-stock-up_down.npy",  "results/bert-sentiment-f1-wsb-stock-up_down.npy"]
    f1_results_vol = ["results/cnn-sentiment-f1-wsb-stock-volitility.npy", "results/attention-sentiment-f1-wsb-stock-volitility.npy"]
    f1_title_ud = "Model F1 Comparison (Stock Up/Down)"
    f1_title_vol = "Model F1 Comparison (Stock Volatility)"
    f1_y = "F1 Score on Validation Set"
    plot_results(f1_results_ud, labels, colors, f1_title_ud, xlabel, f1_y)
    plot_results(f1_results_vol, labels, colors, f1_title_vol, xlabel, f1_y)

    #m coef
    m_results_ud = ["results/cnn-sentiment-m_coef-wsb-stock-up_down.npy", "results/attention-sentiment-m_coef-wsb-stock-up_down.npy",  "results/bert-sentiment-m_coef-wsb-stock-up_down.npy"]
    m_results_vol = ["results/cnn-sentiment-m_coef-wsb-stock-volitility.npy", "results/attention-sentiment-m_coef-wsb-stock-volitility.npy"]
    m_title_ud = "Model Matthews Coefficient Comparison (Stock Up/Down)"
    m_title_vol = "Model Matthews Coefficient Comparison (Stock Volatility)"
    m_coef_y = "M. Coefficient on Validation Set"
    plot_results(m_results_ud, labels, colors, m_title_ud, xlabel, m_coef_y)
    plot_results(m_results_vol, labels, colors, m_title_vol, xlabel, m_coef_y)



if __name__ == '__main__':
    main()
