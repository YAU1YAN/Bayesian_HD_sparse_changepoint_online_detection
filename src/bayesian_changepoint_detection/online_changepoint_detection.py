from __future__ import division
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from math import log, pow

def online_changepoint_detection(data, hazard_func, observation_likelihood): # oncd.online_changepoint_detection(data, partial(oncd.constant_hazard, lamda_gap), oncd.StudentT(0.1, 0.01, 1, 0))
    maxes = np.zeros(len(data) + 1)
    tracker = np.zeros(len(data) +1)
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
        
    for t, x in enumerate(data):
        
        maxes[t] = observation_likelihood.mu[t]
        scale=np.sqrt(observation_likelihood.beta[t] * (observation_likelihood.kappa[t]+1) / (observation_likelihood.alpha[t] *
                            observation_likelihood.kappa[t]))
        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        
        predprobs = observation_likelihood.pdf(x)

        # Evaluate the hazard function for this interval
        H = hazard_func(np.array(range(t+1)))
        
        if t<1:
            H = H*0
            
        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at r = 0.
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)


        # Renormalize the run length probabilities for improved numerical
        # stability.
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        
        #if t<10:
        #     print "R[:, t+1] (normalized): ", R[:t+3, t+1]

        # Update the parameter sets for each possible run length.
        observation_likelihood.update_theta(x)
        #maxes[t] = R[:, t].argmax()

        tracker[t] = scale
        #if t<10:
        #    print mean
       

    return R, maxes, tracker

def HDSonline_changepoint_detection(data, hazard_func, observation_likelihood): # oncd.online_changepoint_detection(data, partial(oncd.constant_hazard, lamda_gap), oncd.StudentT(0.1, 0.01, 1, 0))
    maxes = np.zeros(len(data) + 1)
    tracker = np.zeros(len(data) +1)
    R = np.zeros((len(data) + 1, len(data) + 1))
    R[0, 0] = 1
    declare=[]

    for t, x in enumerate(data):
        predprobs = observation_likelihood.pdf(x)
        H = hazard_func(np.array(range(t+1)))
        
        R[1:t+2, t+1] = R[0:t+1, t] * predprobs * (1-H)
        R[0, t+1] = np.sum( R[0:t+1, t] * predprobs * H)
        
        R[:, t+1] = R[:, t+1] / np.sum(R[:, t+1])
        
       
        if t==50000:
            observation_likelihood.reset_theta(x)
        else:
            observation_likelihood.update_theta(x)
        maxes[t] = observation_likelihood.mu[t]
        
        tracker[t] = 1-R[t,t]
        if tracker[t]>0.9 and len(declare)==0:
            declare.append(t)
    if len(declare)==0:
        declare.append(len(data))
    return R, maxes, tracker, declare[0]  #R, mean, cProb, declare
    
def constant_hazard(lam, r):
    return 1/lam * np.ones(r.shape)


class StudentT: 
    def __init__(self, alpha, beta, kappa, mu):   #so far we have initiated this with oncd.StudentT(0.1, 0.01, 1, 0))
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):  
        return stats.t.pdf(x=data,  
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) / (self.alpha *
                               self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))
            
        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0
    
    def reset_theta(self, data):
        muT0 = np.concatenate((self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, 1))
        alphaT0 = np.concatenate((self.alpha0, 0.1))
        betaT0 = np.concatenate((self.beta0, self.beta + (self.kappa * (data -
            self.mu)**2) / (2. * (self.kappa + 1.))))
            
        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0
        
class Gauss: 
    def __init__(self, sigma, mu):   #so far we have initiated this with oncd.StudentT(0.1, 0.01, 1, 0))
        self.sigma0 = self.sigma = np.array([sigma])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        return stats.t.pdf(x=data,
                           loc=self.mu,
                           scale=self.sigma)

    def update_theta(self, data):
        muT0 = np.concatenate((self.mu0, ))
        kappaT0 = np.concatenate((self.kappa0, ))

            
        self.mu = muT0
        self.kappa = kappaT0
