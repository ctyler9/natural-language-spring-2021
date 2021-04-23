## Vocab Primer 

import os 
import sys
import nltk
from nltk import word_tokenize
nltk.download('unkt')
#import torch
import importlib



from scipy.sparse import csr_matrix
import numpy as np 
from collections import Counter 

import data_crawler
importlib.reload(data_crawler)
from data_crawler import DataLoader


class Vocab: 
	def __init__(self): 
		self.locked = False 
		self.nextId = 0 
		self.word2id = {}
		self.id2word = {}

	
	def get_id(self, word):
		if not word in self.word2id:
			if self.locked: 
				return -1 	# UNK token
			else:
				self.word2id[word] = self.nextId
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
		return self.nextId

	def get_words(self):
		return self.word2id.keys()

	def lock(self):
		self.locked = True



class WSBdata:
	def __init__(self, dataframe, vocab=None):
		""" Reads in data into sparse matrix format """

		if not vocab:
			self.vocab = Vocab()
		else:
			self.vocab = vocab

		#For csr_matrix (see http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
		X_values = []
		X_row_indices = []
		X_col_indices = []
		Y = []

		XwordList = []
		XfileList = []

		#Read positive files
		for i in range(len(dataframe)):
			row = dataframe[i, :]
			for line in row[0]:
				wordList = [self.vocab.GetID(w.lower()) for w in word_tokenize(line) if self.vocab.GetID(w.lower()) >= 0]
				XwordList.append(wordList)
				XfileList.append(row[0])
				wordCounts = Counter(wordList)
				for (wordId, count) in wordCounts.items():
					if wordId >= 0:
						X_row_indices.append(len(pFiles)+i)
						X_col_indices.append(wordId)
						X_values.append(count)

				sentiment_value = self.sentiment_function(row)

				if sentiment_value == 'bearish':
					Y.append(-1.0)
				elif sentiment_value == 'bullish':
					Y.append(1.0)
			
		self.vocab.Lock()

		#Create a sparse matrix in csr format
		self.X = csr_matrix((X_values, (X_row_indices, X_col_indices)), shape=(max(X_row_indices)+1, self.vocab.GetVocabSize()))
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
		comments = df_row[4]

		theta1 = 0.5 
		theta2 = 0.5 

		sentiment = theta1*score + theta2*comments


		if sentiment >= 500:
			return 'bullish'
		else:
			return 'bearish'


if __name__ == '__main__':
	test = WSBdata(DataLoader().get_wsb_data('../data/reddit_wsb.csv'))


