from keras import Input
from keras import Model
from keras.layers import Dense, Activation,Reshape,LSTM
from keras.models import load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import tensorflow as tf

class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=4,
                 layer1_size=16, layer2_size=16, input_dims=128,
                 fname='reinforce.h5'):
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy, self.predict = self.build_policy_network()
        self.action_space = [i for i in range(n_actions)]

        self.model_file = fname

    def build_policy_network(self):
        input = Input(shape=(self.input_dims,))
        advantages = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        reshape_layer = Reshape((1, self.fc2_dims))(dense2) 
        lstm_layer = LSTM(units=64, activation='tanh', return_sequences=False)(reshape_layer)

        probs = Dense(self.n_actions, activation='softmax')(lstm_layer)

        def custom_loss(y_true, y_pred):
            x = tf.math.squared_difference(y_true, y_pred)
            loss = tf.reduce_sum(x)
            return loss

        policy = Model([input,advantages], [probs])

        policy.compile(optimizer=Adam(self.lr), loss=custom_loss)

        predict = Model(input, probs)

        return policy, predict

    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)

        actions = np.zeros([len(action_memory), self.n_actions])
        actions[np.arange(len(action_memory)), action_memory] = 1

        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        self.G = (G - mean) / std

        cost = self.policy.train_on_batch([state_memory, self.G], actions)

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        return cost

    def save_model(self):
        self.policy.save(self.model_file)

    def load_model(self):
        self.policy = load_model(self.model_file)
