import numpy as np
from statistics import mean


class QLearning:
    """
    QLearning reinforcement learning agent

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      discount - (float) The discount factor. Controls the perceived value of
        future reward relative to short-term reward.
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
    """

    def __init__(self, epsilon=0.2, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment

        args:
          env - (Env) OpenAI Gym environment with discrete actions and observations
          steps - (int) The number of actions to perform during training

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. shape is
            (number of environment states x number of possible actions)
            rewards - (np.array) 1D sequence of averaged rewards of length 100
        """
        A = env.action_space.n
        S = env.observation_space.n

        state_action_values = np.zeros((S, A))
        sa_counts = np.zeros((S, A))
        s = steps // 100
        
        rewards = np.zeros(0)
        total = 0
        
        s0 = env.reset()
        
        for i in range(steps):
          larger = np.random.uniform() >= self._get_epsilon(i/steps)
          if larger:
            curr_sav = state_action_values[s0]
            if np.any(curr_sav) == False:
              action = env.action_space.sample()
            else:
              action = np.argmax(curr_sav)
          else:
            action = env.action_space.sample()
          
          s1, reward, done, debug_info = env.step(action)

          sa_counts[s0,action] += 1
          alpha = 1/sa_counts[s0,action]
          curr_Q = state_action_values[s0][action]
          next_Q = np.amax(state_action_values[s1])
          state_action_values[s0,action] += alpha * (reward + self.discount * next_Q - curr_Q)
              
          total += reward
          multiple = (i + 1) % s

          if done:
            s0 = env.reset()
          else:
            s0 = s1

          if multiple == 0:
            reward_term = total/s
            rewards = np.append(rewards, reward_term)
            total = 0
                
        return state_action_values, rewards

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode.

        Arguments:
          env - (Env) OpenAI Gym environment with discrete actions and observations
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. shape is
            (number of environment states x number of possible actions)

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """
        
        states = np.zeros(0)
        actions = np.zeros(0)
        rewards = np.zeros(0)
        
        curr_state = env.reset()
        done = False
        while not done:
          curr_sav = state_action_values[curr_state]
          action = np.argmax(curr_sav)
          next_state, reward, done, debug_info = env.step(action)
          states = np.append(states, next_state)
          actions = np.append(actions, action)
          rewards = np.append(rewards, reward)
          curr_state = next_state
              
        return states, actions, rewards

    def _get_epsilon(self, progress):
        """
        Retrieves the current value of epsilon. 

        Arguments:
            progress - (float) value between 0 and 1 that indicates the
                training progess. calculated as current_step / steps.
        """
        return self._adaptive_epsilon(progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        """
        Modifies epsilon such that it shrinks with learner progress, controls how 
        much exploratory behaior there is after model has identified tennable strategies

        Arguments:
            progress - (float) value between 0 and 1 that indicates the
                training progess. calculated as current_step / steps.
        """
        progress = (1 - progress) * self.epsilon
        return progress