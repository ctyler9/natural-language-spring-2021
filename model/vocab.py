## Vocab Primer
import os
import sys
import nltk
from nltk import word_tokenize
nltk.download('punkt')
from tqdm import tqdm
import torch
import pandas as pd
from pandas.tseries.offsets import BDay
from datetime import datetime, timedelta, date



from scipy.sparse import csr_matrix
import numpy as np
from collections import Counter



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
            id = vocab.get_id(word.lower())
    vocab.lock()
    return vocab

def load_csv(csv_file_path, type_=None):
	if type_ == "reddit" or type_ == None:
		data = pd.read_csv(csv_file_path, delimiter=",")
		data = data[["title", "score", "comms_num", "timestamp"]]
	
	if type_ == "twitter": 
		data = pd.read_csv(csv_file_path, delimiter=",")

	return data


class WSBData():
	def __init__(self, csv_file_path, dataframe=None, vocab=None, train=True):
		""" Reads in data into sparse matrix format """
		if not vocab:
			self.vocab = Vocab()
		else:
			self.vocab = vocab

		if dataframe is not None:
			self.dataframe = dataframe 
		else:
			self.dataframe = pd.read_csv(csv_file_path)
		
		rows = self.dataframe.shape[0]

		self.get_stats_wsb()

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
			if len(wordlist) == 0:
				continue
			XwordList.append(wordlist)
			XfileList.append(row[0])
			wordCounts = Counter(wordlist)
			for (wordId, count) in wordCounts.items():
				if wordId >= 0:
					X_row_indices.append(len(row[0])+i)
					X_col_indices.append(wordId)
					X_values.append(count)

			sentiment_value = self.sentiment_function(row)

			if sentiment_value == "very-bearish":
				Y.append(0)
			elif sentiment_value == "bearish":
				Y.append(1)
			elif sentiment_value == "neutral":
				Y.append(2)
			elif sentiment_value == "bullish":
				Y.append(3)
			elif sentiment_value == "very-bullish":
				Y.append(4)


		self.vocab.lock()

		#Create a sparse matrix in csr format
		# self.X = csr_matrix((X_values, (X_row_indices, X_col_indices)), shape=(max(X_row_indices)+1, self.vocab.get_vocab_size()))
		self.Y = np.asarray(Y)
		print(self.Y.shape)
		print(len(XwordList))
		#Randomly shuffle
		index = np.arange(len(XwordList))
		# print(self.X.shape)
		# index = np.arange(self.X.shape[0])
		np.random.shuffle(index)
		# self.X = self.X[index,:]
		self.XwordList = [torch.LongTensor(XwordList[i]) for i in index]  #Two different sparse formats, csr and lists of IDs (XwordList).
		self.XfileList = [XfileList[i] for i in index]
		self.Y = self.Y[index]


	def sentiment_function(self, df_row):
		# import pdb; pdb.set_trace()
		score = df_row[1]
		comments = df_row[2]

		theta1 = 0.75
		theta2 = 0.25

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
		else:
			return "very-bullish"

	def get_stats_wsb(self, plot=False):
		score = self.dataframe.iloc[:, 1].to_numpy()
		comments = self.dataframe.iloc[:, 2].to_numpy()

		score = np.log(score)
		comments = np.log(comments)


		theta1 = .5
		theta2 = .5
		func = theta1 * score + theta2 * comments


		## all pareto distributions which really shouldn't come as too
		## much of a surprise --
		if plot:
			hist1 = plt.hist(score, bins=100, range=(0, 500))
			hist2 = plt.hist(comments, bins=100, range=(0, 500))
			plt.show()

		score_percentiles = []
		comment_percentiles = []
		func_percentiles = []
		for perc in range(0, 100, 00):
			s_perc = np.percentile(score, perc)
			c_perc = np.percentile(comments, perc)
			f_perc = np.percentile(func, perc)
			score_percentiles.append(s_perc)
			comment_percentiles.append(c_perc)
			func_percentiles.append(f_perc)

		self.score_percentiles = score_percentiles
		self.comment_percentiles = comment_percentiles
		self.func_percentiles = func_percentiles


class TwitterData(): 
	def __init__(self, csv_file_path, dataframe=None, vocab=None, train=True):
		""" Reads in data into sparse matrix format """
		if not vocab:
			self.vocab = Vocab()
		else:
			self.vocab = vocab

		if dataframe is not None:
			self.dataframe = dataframe
		else:
			self.dataframe = pd.read_csv(csv_file_path)


		rows = self.dataframe.shape[0]

		X_values = []
		X_row_indices = []
		X_col_indices = []
		Y = []

		XwordList = []
		XfileList = []

		#Read entries
		for i in tqdm(range(len(self.dataframe))):
			row = self.dataframe.iloc[i, :]
			title = row[0]
			wordlist = []
			tokenized_title = word_tokenize(title)
			for w in tokenized_title:
				id = self.vocab.get_id(w.lower())
				if id >= 0:
					wordlist.append(id)

			if len(wordlist) == 0:
				continue
			XwordList.append(wordlist)
			XfileList.append(row[0])
			wordCounts = Counter(wordlist)
			for (wordId, count) in wordCounts.items():
				if wordId >= 0:
					X_row_indices.append(len(row[0])+i)
					X_col_indices.append(wordId)
					X_values.append(count)

			Y.append(row[1])


		self.vocab.lock()

		#Create a sparse matrix in csr format
		# self.X = csr_matrix((X_values, (X_row_indices, X_col_indices)), shape=(max(X_row_indices)+1, self.vocab.get_vocab_size()))
		self.Y = np.asarray(Y)
		print(self.Y.shape)
		print(len(XwordList))
		#Randomly shuffle
		index = np.arange(len(XwordList))
		# print(self.X.shape)
		# index = np.arange(self.X.shape[0])
		np.random.shuffle(index)
		# self.X = self.X[index,:]
		self.XwordList = [torch.LongTensor(XwordList[i]) for i in index]  #Two different sparse formats, csr and lists of IDs (XwordList).
		self.XfileList = [XfileList[i] for i in index]
		self.Y = self.Y[index]


class WSBDataLarge():
	def __init__(self, csv_file_path, dataframe=None, vocab=None, train=True):
		""" Reads in data into sparse matrix format """
		if not vocab:
			self.vocab = Vocab()
		else:
			self.vocab = vocab

		if dataframe is not None:
			self.dataframe = dataframe 
		else:
			self.dataframe = pd.read_csv(csv_file_path)
		
		rows = self.dataframe.shape[0]

		self.stock_price(pd.read_csv("../data/GME.csv"))


		#self.dataframe["Date"] = pd.to_datetime(dataframe["Date"], format='%Y-%m-%d %H:%M:%S')

		
		self.dataframe["timestamp"] = pd.to_datetime(self.dataframe["timestamp"], format='%Y-%m-%d %H:%M:%S')
		
		isBusinessday = BDay().onOffset
		match_series = self.dataframe["timestamp"].map(isBusinessday)
		self.dataframe = self.dataframe[match_series]

		X_values = []
		X_row_indices = []
		X_col_indices = []
		Y = []

		XwordList = []
		XfileList = []

		#Read entries
		for i in tqdm(range(len(self.dataframe))):
			row = self.dataframe.iloc[i, :]
			title = row[0]
			wordlist = []
			tokenized_title = word_tokenize(title)
			for w in tokenized_title:
				id = self.vocab.get_id(w.lower())
				if id >= 0:
					wordlist.append(id)

			# wordList = [self.vocab.get_id(w.lower()) for w in word_tokenize(title) if self.vocab.get_id(w.lower()) >= 0]
			if len(wordlist) == 0:
				continue
			XwordList.append(wordlist)
			XfileList.append(row[0])
			wordCounts = Counter(wordlist)
			for (wordId, count) in wordCounts.items():
				if wordId >= 0:
					X_row_indices.append(len(row[0])+i)
					X_col_indices.append(wordId)
					X_values.append(count)

			### Add Y logic 

			reddit_date = row[-1] + timedelta(days=1)
			#reddit_date_mon = row[-1] + timedelta(days=2)
			str_time = reddit_date.strftime('%m') + '-' + reddit_date.strftime('%d') 
			#str_time_mon = reddit_date_mon.strftime('%m') + '-' + reddit_date_mon.strftime('%d') 

				
			# need to figure out the fix for Friday to Saturday
			try:
				Y.append(self.gme_stock_dict[str_time])	
			except:
				#Y.append(self.gme_stock_dict[str_time_mon])
				Y.append(0)


		self.vocab.lock()

		#Create a sparse matrix in csr format
		# self.X = csr_matrix((X_values, (X_row_indices, X_col_indices)), shape=(max(X_row_indices)+1, self.vocab.get_vocab_size()))
		self.Y = np.asarray(Y)
		print(self.Y.shape)
		print(len(XwordList))
		#Randomly shuffle
		index = np.arange(len(XwordList))
		# print(self.X.shape)
		# index = np.arange(self.X.shape[0])
		np.random.shuffle(index)
		# self.X = self.X[index,:]
		self.XwordList = [torch.LongTensor(XwordList[i]) for i in index]  #Two different sparse formats, csr and lists of IDs (XwordList).
		self.XfileList = [XfileList[i] for i in index]
		self.Y = self.Y[index]


	def stock_price(self, dataframe):
		dataframe["Date"] = pd.to_datetime(dataframe["Date"], format='%Y-%m-%d %H:%M:%S')
		# start_date = min(self.dataframe['timestamp']) not working for some reason
		start_date = "2021-01-28 00:00:00"
		df = dataframe[["Date", "Open", "Close"]]
		df = df[df["Date"] >= start_date]
		df["Date_str"] = df["Date"].dt.strftime('%m') + '-' + df["Date"].dt.strftime('%d')

		df['Up_Down'] = np.where((df["Close"] - df["Open"]) > 0, 1, -1)

		print(df.head)

		self.gme_stock_dict = pd.Series(df.Up_Down, index=df.Date_str)


		return None 



if __name__ == '__main__':
	test = WSBDataLarge("../data/reddit_data.csv")
