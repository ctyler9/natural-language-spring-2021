import torch
import torch.nn as nn
import torch.nn.functional as F
from vocab import *



class WordAttention(nn.Module):
	def __init__(self, vocab_size, hidden_size=150, atten_size=200, embed_size=300, num_classes=2, dropout=0.5, pad_idx=0):
		super(WordAttention, self).__init__()

		self.pad_idx = pad_idx
		self.word_attention = nn.Linear(2*hidden_size, atten_size)
		self.context_weight = nn.Linear(atten_size, 1, bias=False)
		# self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
		# self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
		# self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
		self.dropout = nn.Dropout(p=dropout)
		self.layer_norm = nn.LayerNorm(2 * hidden_size, elementwise_affine=True)
		self.embedding = nn.Embedding(vocab_size, embed_size)
		self.gru = nn.GRU(embed_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout)
		# self._create_weights(mean=0.0, std=0.05)
		self.fc = nn.Linear(2 * hidden_size, num_classes)
		# self.softmax = nn.Softmax()
		self.log_softmax = nn.LogSoftmax(dim=1)
		self.tan_h = nn.Tanh()

	# def _create_weights(self, mean=0.0, std=0.05):
	#
	# 	self.word_weight.data.normal_(mean, std)
	# 	self.context_weight.data.normal_(mean, std)

	def forward(self, input):
		source_lengths = torch.sum(input != self.pad_idx, axis=1).cpu()
		seq_lens = source_lengths
		embeds = self.embedding(input)
		embeds = self.dropout(embeds)
		packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, seq_lens, enforce_sorted=False, batch_first=True)
		f_output, h_output = self.gru(packed_embeds)  # feature output and hidden state output
		f_output, _ = nn.utils.rnn.pad_packed_sequence(f_output, batch_first=True)
		normed_out = self.layer_norm(f_output)
		atten_out = self.tan_h(self.word_attention(f_output))

		context_vec = self.context_weight(atten_out)
		atten_out = F.softmax(context_vec, dim=1)
		weighted_out = f_output*atten_out
		weighted_out = weighted_out.sum(dim=1)
		lin_out = self.fc(weighted_out)
		log_probs = self.log_softmax(lin_out)
		# output = matrix_mul(f_output, self.word_weight, self.word_bias)
		# output = matrix_mul(output, self.context_weight).permute(1,0)
		# output = F.softmax(output)
		# output = element_wise_mul(f_output,output.permute(1,0))

		return log_probs


# class HierarchicalAttentionNetwork(nn.Module):
# 	def __init__(self, vocab_size, word_hidden_size=300, sent_hidden_size=350, batch_size=50, num_classes=3):
#
# 		super(HierarchicalAttentionNetwork, self).__init__()
# 		self.batch_size = batch_size
# 		self.word_hidden_size = word_hidden_size
# 		self.sent_hidden_size = sent_hidden_size
# 		# self.max_sent_length = max_sent_length
# 		# self.max_word_length = max_word_length
# 		self.word_att_net = WordAttNet(vocab_size, word_hidden_size)
# 		self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
# 		self._init_hidden_state()
#
# 	def _init_hidden_state(self, last_batch_size=None):
# 		if last_batch_size:
# 			batch_size = last_batch_size
# 		else:
# 			batch_size = self.batch_size
# 			self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
# 			self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
# 		if torch.cuda.is_available():
# 			self.word_hidden_state = self.word_hidden_state.cuda()
# 			self.sent_hidden_state = self.sent_hidden_state.cuda()
#
# 	def forward(self, input):
# 		output_list = []
# 		input = input.unsqueeze(-1)
# 		input = input.permute(1, 0, 2)
# 		print(input.shape)
# 		for i in input:
# 			print(i.shape)
# 			output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
# 			output_list.append(output)
# 		output = torch.cat(output_list, 0)
# 		output, self.sent_hidden_state = self.sent_att_net(output, self.sent_hidden_state)
#
# 		return output
#
#
# class WordAttNet(nn.Module):
# 	def __init__(self, vocab_size, hidden_size=50, embed_size=300):
# 		super(WordAttNet, self).__init__()
#
# 		# dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
# 		# dict_len, embed_size = dict.shape
# 		# dict_len += 1
# 		# unknown_word = np.zeros((1, embed_size))
# 		# dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float))
#
# 		self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
# 		self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
# 		self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
#
# 		self.lookup = nn.Embedding(vocab_size, embed_size)
# 		self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)
# 		self._create_weights(mean=0.0, std=0.05)
# 		# self.fc = nn.Linear(2 * hidden_size, num_classes)
#
# 	def _create_weights(self, mean=0.0, std=0.05):
#
# 		self.word_weight.data.normal_(mean, std)
# 		self.context_weight.data.normal_(mean, std)
#
# 	def forward(self, input, hidden_state):
#
# 		output = self.lookup(input)
# 		f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
# 		output = matrix_mul(f_output, self.word_weight, self.word_bias)
# 		output = matrix_mul(output, self.context_weight).permute(1,0)
# 		output = F.softmax(output)
# 		output = element_wise_mul(f_output,output.permute(1,0))
#
# 		return output, h_output
#
#
# class SentAttNet(nn.Module):
# 	def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=3):
# 		super(SentAttNet, self).__init__()
#
# 		self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
# 		self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
# 		self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))
#
# 		self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
# 		self.fc = nn.Linear(2 * sent_hidden_size, num_classes)
# 		# self.sent_softmax = nn.Softmax()
# 		# self.fc_softmax = nn.Softmax()
# 		self._create_weights(mean=0.0, std=0.05)
#
# 	def _create_weights(self, mean=0.0, std=0.05):
# 		self.sent_weight.data.normal_(mean, std)
# 		self.context_weight.data.normal_(mean, std)
#
# 	def forward(self, input, hidden_state):
# 		f_output, h_output = self.gru(input, hidden_state)
# 		output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
# 		output = matrix_mul(output, self.context_weight).permute(1, 0)
# 		output = F.softmax(output)
# 		output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
# 		output = self.fc(output)
#
# 		return output, h_output
#
#
# def matrix_mul(input, weight, bias=False):
# 	feature_list = []
# 	for feature in input:
# 		feature = torch.mm(feature, weight)
# 		if isinstance(bias, torch.nn.parameter.Parameter):
# 			feature = feature + bias.expand(feature.size()[0], bias.size()[1])
# 		feature = torch.tanh(feature).unsqueeze(0)
# 		feature_list.append(feature)
#
# 	return torch.cat(feature_list, 0).squeeze()
#
# def element_wise_mul(input1, input2):
#
# 	feature_list = []
# 	for feature_1, feature_2 in zip(input1, input2):
# 		feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
# 		feature = feature_1 * feature_2
# 		feature_list.append(feature.unsqueeze(0))
# 	output = torch.cat(feature_list, 0)
#
# 	return torch.sum(output, 0).unsqueeze(0)


def main():
	pass

if __name__ == '__main__':
	main()
