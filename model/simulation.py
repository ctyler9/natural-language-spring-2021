### Simulation backend 


import os 
import pandas as pd 
import numpy as np 
import random 
import copy
from collections import deque 


from model import ActorModel, CriticModel




class CustomEnvironment(): 
	def __init__(self, df, initial_balance=1000, lookback_window_size=50): 
		self.df = df.dropna().reset_index()
		self.df_total_steps = self.df.shape[0] - 1
		self.initial_balance = initial_balance 
		self.lookback_window_size = lookback_window_size

		# hold, buy, sell
		self.action_space = np.array([0,1,2])

		# history 
		self.orders_history = deque(maxlen=self.lookback_window_size)
		self.market_history = deque(maxlen=self.lookback_window_size)
		self.state_size = (self.lookback_window_size, 10)


		# hyperparameters 	
		self.lr = 0.001 
		self.epochs = 1
		self.normalize_value = 1000 
		self.optimizer = Adam 

		# call in actor critic model 
		self.Actor = ActorModel(input_shape=self.state_size, action_space=self.action_space.shape[0], lr=self.lr, optimizer=self.optimizer)
		self.Critic = CriticModel(input_shape=self.state_size, action_space=self.action_space.shape[0], lr=self.lr, optimizer=self.optimizer)



	def writer():
		pass 


	def reset(self, env_steps_size=0): 
		# initial state/conditions 
		self.balance = self.initial_balance
		self.net_worth = self.initial_balance
		self.stock_held = 0 
		self.stock_sold = 0 
		self.stock_bought = 0 
		self.episode_orders = 0 
		self.env_steps_size = env_steps_size
		if env_steps_size > 0: 
			self.star_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
			self.end_step = self.start_step + env_steps_size
		else: 
			self.start_step = self.lookback_window_size
			self.end_step = self.df_total_steps

		self.current_step = self.start_step

		for i in reversed(range(self.lookback_window_size)):
			self.orders_history.append([self.balance, self.net_worth, self.stock_bought, self.stock_sold, self.stock_held])
			self.market_history.append([self.df.loc[current_step, "open"],
										self.df.loc[current_step, "high"],
										self.df.loc[current_step, "low"],
										self.df.loc[current_step, "close"]
										])
		
		state = np.concatenate((self.market_history, self.orders_history), axis=1)

		return state 


	def _next_observation(self):
		self.market_history.append([self.df.loc[current_step, "open"],
										self.df.loc[current_step, "high"],
										self.df.loc[current_step, "low"],
										self.df.loc[current_step, "close"]
										])

		obs = np.concatenate((self.market_history, self.orders_history), axis=1)

		return obs 


	def step(self, action): 
		# go one timestep in the future 

		self.stock_bought = 0 
		self.stock_sold = 0 
		self.current_step += 1


		open_ = self.df.loc[self.current_step, "open"]
		close = self.df.loc[self.current_step, "close"]
		current_price = random.uniform(open_, close)

		if action == 0:
			pass # Hodl
		elif action == 1 and self.balance > self.initial_balance/100:
			# buy 
			self.stock_bought = self.balance / current_price 
			self.balance -= self.stock_sold * current_price 
			self.stock_held += self.stock_bought 
			self.episode_orders += 1 
		elif action == 2 and self.stock_held > 0: 
			# sell 
			self.stock_sold = self.stock_held
			self.balance += self.stock_sold * current_price
			self.episode_orders += 1

		self.prev_net_worth = self.net_worth
		self.net_worth = self.balance + self.stock_held * current_price

		self.orders_history.append([self.balance, self.net_worth, self.stock_bought, self.stock_sold, self.stock_held])


		reward = self.net_worth - self.prev_net_worth

		if self.net_worth <= self.initial_balance/2: 
			done = True 
		else:
			done = False 

		obs = self._next_observation() / self.normalize_value


		return obs, reward, done 

	def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True): 
		deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
		deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Compute discounted rewards
        #discounted_r = np.vstack(self.discount_rewards(rewards))

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)
        # Compute advantages
        #advantages = discounted_r - values
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True)

        self.replay_count += 1
        
    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction
