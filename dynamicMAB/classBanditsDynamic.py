import numpy as np
from numpy import random

class Bandits:
    def __init__(self,total_time):
        self.options = 4            # Number of options
        self.sigma = 4              # Variance of payoff distribution for each option
        self.decay = 0.9836         # Decay constant of expected payoff RW
        self.decay_centre = 50      # Decay centre of expected payoff RW
        self.decay_noise = 2.8      # Variance of expected payoff RW
        self.time = total_time      # Total simulation time

    def sample(self,mu):
        ''' Sample from Gaussian function with input mean and set variance '''
        return random.normal(mu,self.sigma)
    
    def genMeans(self,starting_var):
        ''' Generate the moving payoff according to a decaying Gaussian random walk. 
            Starting_var = how spread apart the options may begin. '''
        # Generate the mean payoff according to a decaying Gaussian random walk (what is decaying?)
        noise = random.normal(0,self.decay_noise,(self.options,self.time))          # Preallocating Gaussian noise for payoff diffusion
        payoff0 = np.round(random.normal(50,starting_var,(self.options,1)))         # Defining expected payoffs for step 1
        payoff = np.zeros([self.options,self.time])                                 # Preallocating payoff walker
        payoff[:,0] = np.transpose(payoff0)
        for t in range(1,self.time):                            # Loop over time and generate payoff RW
            payoff[:,t] = self.decay*payoff[:,t-1] + (1-self.decay) * self.decay_centre + noise[:,t]
        return payoff

    def findRegret(self,payoff,result):
        ''' Find the final regret of a trial given the input payoff function and summed reward. 
            Outputs: regret + scores nad choices of optimal algorithm'''
        optimal_scores = np.zeros(self.time)
        optimal_choices = np.zeros(self.time)
        for t in range(self.time):              # Find the optimal solution by picking highest payoff option
            chosen = np.argmax(payoff[:,t])
            # optimal_scores[t] = self.sample(payoff[chosen,t])         # Sample from option
            optimal_scores[t] = payoff[chosen,t]                        # Take mean of option: minimising randomness
            optimal_choices[t] = chosen
        regret = 1 - result/sum(optimal_scores)
        return regret, optimal_scores, optimal_choices

    def exploit(self,payoff,trials):
        ''' Pure exploitation algorithm. '''
        scores = np.zeros([trials,self.time])                               # Initialise score history
        scores[:,range(self.options)] = [self.sample(payoff[range(self.options),0]) for _ in range(trials)]     # Sample from each option once
        chosen = np.argmax(scores,1)
        for t in range(self.options,self.time):
            scores[:,t] = self.sample(payoff[chosen,t])
        return np.sum(scores,1)

    def softmax(self,payoff,temp,rec,trials):
        ''' Discounted softmax algorithm, taking input temperature and discount factor.'''
        scores = np.zeros([trials,self.time])
        history = {idx: np.zeros([trials,self.time]) for idx in range(self.options)}
        history_choices = np.zeros([trials,self.time])

        for option in history:      # Sample each option once and record
            scores[:,option] = self.sample(np.zeros(trials) + payoff[option,0])
            history[option][:,option] = scores[:,option]
            history_choices[:,option] = np.zeros(trials) + option
        
        for t in range(self.options,self.time):
            # Calculate historical mean of each option, then apply softmax activation function
            expected_rewards = np.transpose([np.sum(history[option],1)/np.count_nonzero(history[option],1) for option in history])/temp
            weights = np.exp(expected_rewards) / np.sum(np.exp(expected_rewards),1).reshape([trials,1])
            chosen = (np.random.rand(len(weights),1) < weights.cumsum(axis=1)).argmax(axis=1)      # Selecting from given weights
            scores[:,t] = self.sample(payoff[chosen,t])
            for i,c in enumerate(chosen):                   # Updating history with scores
                history[c][i,:] = history[c][i,:] * rec     # Multiply in recent information bias for chosen option
                history[c][i,t] = scores[i,t]
            history_choices[:,t] = chosen
        return scores,history_choices, history

    def dUCB1(self,payoff,n,rec,trials):
        ''' Discounted UCB algorithm with added softmax, taking uncertainty weight, temperature, and discount factor. NO PRIOR IMPLEMENTED.''' 
        # Initialising scores and recording structures
        scores = np.zeros([trials,self.time])
        history = {idx: np.zeros([trials,self.time]) for idx in range(self.options)}
        history_choices = np.zeros([trials,self.time])
        for option in history:              # Sample from each option once
            scores[:,option] = self.sample(np.zeros(trials) + payoff[option,0])
            history[option][:,option] = scores[:,option]
            history_choices[:,option] = np.zeros(trials) + option

        # Actual loop starting here.
        for t in range(self.options,self.time):
            mu_rewards = [np.sum(history[option],1)/np.count_nonzero(history[option],1) for option in history]     # Mean history per option
            IB =  [n*np.sqrt(2*np.log(t) / np.count_nonzero(history[option],1)) for option in history]
            UCB_rewards = np.transpose(np.add(mu_rewards,IB))
            chosen = np.argmax(UCB_rewards,1)
            
            scores[:,t] = self.sample(payoff[chosen,t])
            for i,c in enumerate(chosen):
                history[c][i,:] = history[c][i,:] * rec     # Multiply in recent information bias for chosen option
                history[c][i,t] = scores[i,t]
            history_choices[:,t] = chosen
        return scores,history_choices,history

    def softdUCB1(self,payoff,n,temp,rec,trials):
        ''' Discounted UCB algorithm with added softmax, taking uncertainty weight, temperature, and discount factor. NO PRIOR IMPLEMENTED.''' 
        # Initialising scores and recording structures
        scores = np.zeros([trials,self.time])
        history = {idx: np.zeros([trials,self.time]) for idx in range(self.options)}
        history_choices = np.zeros([trials,self.time])
        for option in history:              # Sample from each option once
            scores[:,option] = self.sample(np.zeros(trials) + payoff[option,0])
            history[option][:,option] = scores[:,option]
            history_choices[:,option] = np.zeros(trials) + option

        # Actual loop starting here.
        for t in range(self.options,self.time):
            mu_rewards = [np.sum(history[option],1)/np.count_nonzero(history[option],1) for option in history]     # Mean history per option
            # std_rewards = [np.sqrt((np.sum((history[option] - mu_rewards[:,option].reshape(trials,1))**2,1) - mu_rewards[:,option]**2 
            #         * (t-np.count_nonzero(history[option],1)))/np.count_nonzero(history[option],1)) for option in history]
            # UCB_rewards = (mu_rewards + n*np.transpose(std_rewards))
            IB =  [n*np.sqrt(2*np.log(t) / np.count_nonzero(history[option],1)) for option in history]
            UCB_rewards = np.transpose(np.add(mu_rewards,IB))/temp
            weights = np.exp(UCB_rewards)/ np.sum(np.exp(UCB_rewards),1).reshape([trials,1]) 
            chosen = (np.random.rand(len(weights),1) < weights.cumsum(axis=1)).argmax(axis=1)      # Selecting from probabilities
            
            # Need to add uncertainty bonus to ^: just + n*SD[option], where n is some number of SDs.
            # chosen = np.argmax(UCB_rewards,1)
            scores[:,t] = self.sample(payoff[chosen,t])
            for i,c in enumerate(chosen):
                history[c][i,:] = history[c][i,:] * rec     # Multiply in recent information bias for chosen option
                history[c][i,t] = scores[i,t]
            history_choices[:,t] = chosen
        return scores,history_choices,history
