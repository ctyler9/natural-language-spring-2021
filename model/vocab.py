## Vocab Primer
import os
import sys
import nltk
from nltk import word_tokenize
nltk.download('punkt')
from tqdm import tqdm
import torch
import pandas as pd



from scipy.sparse import csr_matrix
import numpy as np
from collections import Counter

from data import DataLoader


class Vocab():
	def __init__(self):
		self.locked = False
		self.nextID = 0
		self.word2id = {}
		self.id2word = {}

	def get_id(self, word):
		if not word in self.word2id:
			if self.locked:
				return -1 	# UNK token
			else:
				self.word2id[word] = self.nextID
				self.id2word[self.word2id[word]] = word
				self.nextID += 1
		return self.word2id[word]

	def has_word(self, word):
		return self.word2id.has_key(word)

	def has_id(self, wid):
		return self.word2.has_key(wid)

	def get_word(self, wid):
		return self.id2word[wid]

	def save_vocab(self, vocabFile):
		fOut = open(vocabFile, 'w')
		for word in self.word2id.keys():
			fOut.write("%s\t%s\n" % (word, self.word2id[word]))

	def get_vocab_size(self):
		#return self.nextId-1
		return self.nextID

	def get_words(self):
		return self.word2id.keys()

	def lock(self):
		self.locked = True

def create_vocab(wsb_data):
    vocab = Vocab()
    for item in wsb_data:
        tokenized_item = word_tokenize(item)
        for word in tokenized_item:
            id = vocab.GetID(word.lower())
    vocab.Lock()
    return vocab

def load_csv(csv_file_path):
	data = pd.read_csv(path, delimiter=",")
	title = data['title'].values
	upvotes = data['score'].values
	comment_num = data['comms_num'].values
	wsb_data = pd.DataFrame({'title':title, 'score':upvotes, 'comms_num':comment_num})
	return wsb_data


class WSBdata(DataLoader):
	def __init__(self, csv_file_path, dataframe=None, vocab=None, train=True):
		""" Reads in data into sparse matrix format """
		super(WSBdata, self).__init__(csv_file_path)
		self.get_stats()

		if not vocab:
			self.vocab = Vocab()
		else:
			self.vocab = vocab

		if not dataframe:
			dataframe = self.wsb_data

		rows = dataframe.shape[0]
		
		# if train:
		# 	dataframe = dataframe.iloc[rows//4:, :]
		# else:
		# 	dataframe = dataframe.iloc[:rows//4, :]

		#For csr_matrix (see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
		X_values = []
		X_row_indices = []
		X_col_indices = []
		Y = []

		XwordList = []
		XfileList = []

		#Read entries
		for i in tqdm(range(len(dataframe))):
			row = dataframe.iloc[i, :]
			title = row[0]
			wordlist = []
			tokenized_title = word_tokenize(title)
			for w in tokenized_title:
				id = self.vocab.get_id(w.lower())
				if id >= 0:
					wordlist.append(id)

			# wordList = [self.vocab.get_id(w.lower()) for w in word_tokenize(title) if self.vocab.get_id(w.lower()) >= 0]
			if len(wordList) == 0:
				continue
			XwordList.append(wordList)
			XfileList.append(row[0])
			wordCounts = Counter(wordList)
			for (wordId, count) in wordCounts.items():
				if wordId >= 0:
					X_row_indices.append(len(row[0])+i)
					X_col_indices.append(wordId)
					X_values.append(count)

			sentiment_value = self.sentiment_function(row)

			if sentiment_value == "very-bearish":
				Y.append(0.0)
			elif sentiment_value == "bearish":
				Y.append(1.0)
			elif sentiment_value == "neutral":
				Y.append(2.0)
			elif sentiment_value == "bullish":
				Y.append(3.0)
			elif sentiment_value == "very-bullish":
				Y.append(4.0)
			# for line in row[0]:
			# 	wordList = [self.vocab.get_id(w.lower()) for w in word_tokenize(line) if self.vocab.get_id(w.lower()) >= 0]
			# 	if len(wordList) == 0:
			# 		continue
			# 	XwordList.append(wordList)
			# 	XfileList.append(row[0])
			# 	wordCounts = Counter(wordList)
			# 	for (wordId, count) in wordCounts.items():
			# 		if wordId >= 0:
			# 			X_row_indices.append(len(row[0])+i)
			# 			X_col_indices.append(wordId)
			# 			X_values.append(count)
			#
			# 	sentiment_value = self.sentiment_function(row)
			#
			# 	if sentiment_value == "very-bearish":
			# 		Y.append(0.0)
			# 	elif sentiment_value == "bearish":
			# 		Y.append(1.0)
			# 	elif sentiment_value == "neutral":
			# 		Y.append(2.0)
			# 	elif sentiment_value == "bullish":
			# 		Y.append(3.0)
			# 	elif sentiment_value == "very-bullish":
			# 		Y.append(4.0)


		self.vocab.lock()

		#Create a sparse matrix in csr format
		self.X = csr_matrix((X_values, (X_row_indices, X_col_indices)), shape=(max(X_row_indices)+1, self.vocab.get_vocab_size()))
		self.Y = np.asarray(Y)

		#Randomly shuffle
		index = np.arange(self.X.shape[0])
		np.random.shuffle(index)
		self.X = self.X[index,:]
		self.XwordList = [torch.LongTensor(XwordList[i]) for i in index]  #Two different sparse formats, csr and lists of IDs (XwordList).
		self.XfileList = [XfileList[i] for i in index]
		self.Y = self.Y[index]


	def sentiment_function(self, df_row):
		score = df_row[1]
		comments = df_row[2]

		theta1 = 0.5
		theta2 = 0.5

		sentiment = theta1*score + theta2*comments

		bound1, bound2, bound3, bound4, bound5 = tuple(self.func_percentiles)

		# know i could have just made it return a number, but thought
		# i'd keep it string match to get a general concept across
		if sentiment >= 0 and sentiment < bound1:
			return 'very-bearish'
		elif sentiment >= bound1 and sentiment < bound2:
			return 'bearish'
		elif sentiment >= bound2 and sentiment < bound3:
			return 'neutral'
		elif sentiment >= bound3 and sentiment < bound4:
			return 'bullish'
		elif sentiment >= bound4 and sentiment < bound5:
			return 'very-bullish'


if __name__ == '__main__':
	test = WSBdata()
