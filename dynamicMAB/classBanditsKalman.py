import numpy as np
from numpy import random

class BanditsKalman:
    def __init__(self,total_time):
        self.options = 4            # Number of options
        self.sigma = 4              # Variance of sampled payoff from option
        self.decay = 0.9836         # Decay constant of expected payoff RW
        self.decay_centre = 50      # Decay centre of expected payoff RW
        self.decay_noise = 2.8      # Variance of expected payoff RW
        self.time = total_time      # Total simulation time

    def sample(self,mu):
        # Sample from input mean and constant variance
        return random.normal(mu,self.sigma)
    
    def genMeans(self,starting_var):
        # Generate the mean payoff according to a decaying Gaussian random walk (what is decaying?)
        noise = random.normal(0,self.decay_noise,(self.options,self.time))   # Preallocating Gaussian noise for payoff diffusion
        payoff0 = np.round(random.normal(50,starting_var,(self.options,1)))            # Defining expected payoffs for step 1
        payoff = np.zeros([self.options,self.time])                          # Preallocating payoff walker
        payoff[:,0] = np.transpose(payoff0)
        for t in range(1,self.time):        # Loop over time and generate expected payoff RW
            payoff[:,t] = self.decay*payoff[:,t-1] + (1-self.decay) * self.decay_centre + noise[:,t]
        return payoff

    def findRegret(self,payoff,result):
        optimal_scores = np.zeros(self.time)
        optimal_choices = np.zeros(self.time)
        for t in range(self.time):
            chosen = np.argmax(payoff[:,t])
            # optimal_scores[t] = self.sample(payoff[chosen,t])
            optimal_scores[t] = payoff[chosen,t]
            optimal_choices[t] = chosen
        regret = 1 - result/sum(optimal_scores)
        return regret, optimal_scores, optimal_choices

    def exploit(self,payoff,trials):
        scores = np.zeros([trials,self.time])                               # Initialise score history
        scores[:,range(self.options)] = [self.sample(payoff[range(self.options),0]) for _ in range(trials)]     # Sample from each option once
        chosen = np.argmax(scores,1)
        for t in range(self.options,self.time):
            scores[:,t] = self.sample(payoff[chosen,t])
        return np.sum(scores,1)

    def softmax(self,payoff,temp,trials):
        ''' Implement softmax algorithm under Kalman filter. Assume all parameters are known.'''
        scores = np.zeros([trials,self.time])
        # history = {idx:np.zeros([trials,self.time]) for idx in range(self.options)}       # History of priors: need this to be a single vector
        history = np.zeros([trials,self.time,self.options])
        history_var = np.zeros([trials,self.time,self.options])
        for option in range(self.options):
            scores[:,option] = self.sample(np.zeros(trials) + payoff[option,0])
            history[:,0,option] = scores[:,option]
            history_var[:,0,option] = np.zeros(trials) + self.sigma
        for t in range(1, self.time-self.options):
            payoff_priors = history[:,t-1,:]/temp
            weights = np.exp(payoff_priors) / np.sum(np.exp(payoff_priors),1).reshape([trials,1])
            chosen = (np.random.rand(len(weights),1) < weights.cumsum(axis=1)).argmax(axis=1)
            scores[:,t+self.options] = self.sample(payoff[chosen,t])
            
            # Compute posterior mean and variance
            pred_err = scores[:,t+self.options-1] - history[:,t-1,chosen]
            gain = np.sqrt(history_var[:,t-1,chosen]**2 / (history_var[:,t-1,chosen]**2 + 4**2))
            post_mean = history_var[:,t-1,chosen] + gain*pred_err
            post_var = (1-gain)*history_var[:,t-1,chosen]

            # Compute new prior mean and variance
            prior_mean = self.decay*post_mean + (1-self.decay)*self.decay_centre
            prior_var = np.sqrt(self.decay**2 * post_var**2 + self.decay_noise)

            history[:,t,:] = prior_mean
            history_var[:,t,:] = prior_var
        return np.sum(scores,1),history


