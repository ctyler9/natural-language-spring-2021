import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab import *



class AttentionModel(nn.Module):
	def __init__(self, vocab_size, DIM_EMB=300, HID_DIM=300,
				NUM_CLASSES=5, CONT_DIM=350, dropout=0.5, pad_idx=0):

		super().__init__()

		self.HID_DIM = HID_DIM

		# core
		self.pad_idx = pad_idx
		self.embedding = nn.Embedding(vocab_size, DIM_EMB, padding_idx=pad_idx)
		self.dropout = nn.Dropout(p=dropout)
		self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=DIM_EMB,
			hidden_size=HID_DIM, batch_first=True, dropout=dropout)

		# attention
		self.ui = nn.Linear(2*HID_DIM, CONT_DIM)
		self.uw = nn.Parameter(torch.randn(CONT_DIM))

		self.tan_h = nn.Tanh()
		self.relu = nn.ReLU()

		# output
		self.fc1 = nn.Linear(2*HID_DIM, HID_DIM)
		self.fc = nn.Linear(HID_DIM, NUM_CLASSES)
		self.log_softmax = nn.LogSoftmax(dim=1)


	def forward(self, x, seq_lens=None):
		source_lengths = torch.sum(x != self.pad_idx, axis=1).cpu()
		seq_lens = source_lengths
		embeds = self.dropout(self.embedding(x))

		# if seq_lens is not None:
		packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, seq_lens, enforce_sorted=False, batch_first=True)
			# embeds = packed_embeds.clone()
		enc_packed, (h_n, c_n) = self.lstm(packed_embeds)
		# enc, (h_n, c_n) = self.lstm(embeds)

		# if seq_lens is not None:
		enc, _ = nn.utils.rnn.pad_packed_sequence(enc_packed, batch_first=True)
		# u_it = self.relu(self.ui(enc))
		u_it = self.tan_h(self.ui(enc))

		
		weights = torch.softmax(u_it.matmul(self.uw), dim=1).unsqueeze(1)
		sent = torch.sum(weights.matmul(enc), dim=1)



		fc1_out = self.fc1(sent)
		logits = self.fc(fc1_out)
		# logits = self.fc(sent)
		logits = self.log_softmax(logits)
		# preds = logits.argmax(-1)

		return logits


def main():
	pass

if __name__ == '__main__':
	main()
