### Data Exploration



import numpy as np
import os 



class DataLoader():
	def __init__(self):
		self.wsb_data = None 
		self.stock_data = None 



	def get_wsb_data(self, path):
		data = np.readtxt(path, delimiter=",")
		self.wsb_data = data[:, 1:] 

		return data 


	def get_stock_data(self, path):
		files = os.listdir(path)

		for file in files: 
			data = np.readtxt(file, delimiter=",")
			## will continue this if decide to add 


	def reddit_api_call(self, json='api.json'):
		### read json and load api 
		pass 

	def get_live_wsb(self, ): 
		self.reddit_api_call()

		### call data in 


