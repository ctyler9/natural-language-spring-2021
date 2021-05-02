import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab import *



class AttentionModel(nn.Module):
	def __init__(self, vocab_size, DIM_EMB=300, HID_DIM=300,
				NUM_CLASSES=5, CONT_DIM=400, dropout=0.5, pad_idx=0):

		super().__init__()

		self.HID_DIM = HID_DIM

		# core
		self.embedding = nn.Embedding(vocab_size, DIM_EMB, padding_idx=pad_idx)
		self.dropout = nn.Dropout(p=dropout)
		self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=DIM_EMB,
			hidden_size=HID_DIM, batch_first=True, dropout=dropout)

		# attention
		self.ui = nn.Linear(2*HID_DIM, CONT_DIM)
		self.uw = nn.Parameter(torch.randn(CONT_DIM))

		# output
		self.fc = nn.Linear(2*HID_DIM, NUM_CLASSES)
		self.log_softmax = nn.LogSoftmax(dim=1)


	def forward(self, x, seq_lens=None):

		embeds = self.dropout(self.embedding(x))

		# if seq_lens is not None:
		# 	packed_embeds = nn.utils.rnn.packed_embeds(embeds, seq_lens, batch_first=True)
		# 	embeds = packed_embeds.clone()


		enc, (h_n, c_n) = self.lstm(embeds)

		# if seq_lens is not None:
		# 	enc, _ = nn.utils.rnn.packed_sequence(enc, batch_first=True)

		u_it = torch.tanh(self.ui(enc))
		weights = torch.softmax(u_it.matmul(self.uw), dim=1).unsqueeze(1)
		sent = torch.sum(weights.matmul(enc), dim=1)

		logits = self.fc(sent)
		logits = self.log_softmax(logits)
		# preds = logits.argmax(-1)

		return logits, weights


def main():
	pass

if __name__ == '__main__':
	main()
