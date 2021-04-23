### Data Exploration



import numpy as np
import pandas as pd
import os 



class DataLoader():
	def __init__(self):
		self.wsb_data = None 
		self.stock_data = None 

		#self.get_wsb()


	def get_wsb_data(self, path):
		data = pd.read_csv(path, delimiter=",", skiprows=1)
		self.wsb_data = data

		return data

	def get_stock_data(self, path):
		files = os.listdir(path)

		for file in files: 
			data = pd.read_csv(file, delimiter=",")
			## will continue this if decide to add 


	def reddit_api_call(self, json='api.json'):
		### read json and load api 
		pass 

	def get_live_wsb(self, ): 
		self.reddit_api_call()

		### call data in 


