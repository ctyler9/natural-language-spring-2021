### Data Exploration



import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt



class DataLoader():
	def __init__(self, csv_file_path, dataframe=None):
		self.wsb_data = None
		self.stock_data = None
		self.score_percentiles = None
		self.comment_percentiles = None
		self.func_percentiles = None
		if dataframe is None:
			self.get_wsb_data(csv_file_path)
		else:
			self.wsb_data = dataframe


	def get_wsb_data(self, path):
		data = pd.read_csv(path, delimiter=",")
		title = data['title'].values
		upvotes = data['score'].values
		comment_num = data['comms_num'].values
		wsb_data = pd.DataFrame({'title':title, 'score':upvotes, 'comms_num':comment_num})
		self.wsb_data = wsb_data

		return data

	def get_stock_data(self, path):
		files = os.listdir(path)

		for file in files:
			data = pd.read_csv(file, delimiter=",")
			## will continue this if decide to add


	def reddit_api_call(self, json='api.json'):
		### read json and load api
		pass

	def get_live_wsb(self):
		self.reddit_api_call()

		### call data in
		pass

	def get_stats(self, plot=False):
		score = self.wsb_data.iloc[:, 1].to_numpy()
		comments = self.wsb_data.iloc[:, 2].to_numpy()


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
		for perc in range(0, 100, 20):
			s_perc = np.percentile(score, perc)
			c_perc = np.percentile(comments, perc)
			f_perc = np.percentile(func, perc)
			score_percentiles.append(s_perc)
			comment_percentiles.append(c_perc)
			func_percentiles.append(f_perc)

		self.score_percentiles = score_percentiles
		self.comment_percentiles = comment_percentiles
		self.func_percentiles = func_percentiles




if __name__ == '__main__':
	DataLoader().get_stats()
