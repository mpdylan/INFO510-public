'''
A couple of simple implementations of Hidden Markov models.
'''

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class MultinomialHMM:
    '''
    A discrete state HMM with multinomial emissions. Supports filtering, smoothing, and fitting via the Baum-Welch algorithm.
    '''
    def __init__(self, A, B, π):
        '''
        Initialize the model.
        Parameters:
            A: transition matrix
            B: observation matrix
            π: initial state distribution
        Returns: None
        '''
        self.A = A
        self.B = B
        self.π = π
        self.N = A.shape[0]
    
    def forward(self, O, scale=True):
        '''
        Runs the forward pass algorithm, used in filtering and smoothing.
        Parameters:
            O: a sequence of observations
            scale: boolean, determines whether to rescale each step. Should be True (default) for longer sequences to avoid underflow.
        Returns:
            α: array
            c: array, if scale is True
        '''
        T = len(O)
        α = np.zeros((self.N, T))
        if scale:
            c = np.zeros(T)
        for i in range(self.N):
            α[i, 0] = self.π[i] * self.B[i, O[0]]
        if scale:
            c[0] = 1 / np.sum(α[:, 0])
            α[:, 0] *= c[0]
        t = 1
        while t < T:
            for i in range(self.N):
                α[i, t] = np.sum(α[:, t-1] * self.A[:, i] * self.B[i, O[t]])
            if scale:
                c[t] = 1 / np.sum(α[:, t])
                α[:, t] *= c[t]
            t += 1
        if scale:
            return α, c
        else:
            return α
    
    def backward(self, O, c=None):
        '''
        Runs the backward pass algorithm, used in smoothing. Rescales at each step if an array of scaling parameters (from the forward pass) is passed.
        Parameters:
            O: a sequence of observations
            c: an array of scaling parameters
        Returns:
            β: backward pass parameters
        '''
        T = len(O)
        β = np.zeros((self.N, T))
        for i in range(self.N):
            β[i, T-1] = 1
            if c is not None:
                β[i, T-1] *= c[T-1]
        t = T-1
        while t > 0:
            for i in range(self.N):
                β[i, t-1] = np.sum(β[:, t] * self.A[i, :] * self.B[:, O[t]])
                if c is not None:
                    β[i, t-1] *= c[t-1]
            t -= 1
        return β
    
    def filter(self, O, scale = True):
        '''
        Returns the estimated distribution of states at time T, given observations O.
        Parameters:
            O: sequence of observations
            scale: boolean, determines whether to rescale each step. Should be True (default) for longer sequences to avoid underflow.
        Returns:
            Array of probabilities
        '''
        T = len(O)
        α, c = self.forward(O, scale)
        return α[:, T - 1] / np.sum(α[:, T - 1])
    
    def smooth(self, O, scale = True):
        '''
        Returns the estimated distribution of states at all times, given observations O.
        Parameters:
            O: sequence of observations
            scale: boolean, determines whether to rescale each step. Should be True (default) for longer sequences to avoid underflow.
        Returns:
            Array of probabilities
        '''
        T = len(O)
        if scale:
            α, c = self.forward(O, scale)
        else:
            α = self.forward(O, scale)
            c = None
        β = self.backward(O, c)
        if scale:
            return α * β / c
        else:
            norm = np.sum(α[:, T-1])
            return α * β / norm
    
    def generate(self, T):
        '''
        Generate simulated data given the current model parameters.
        Parameters:
            T: length of sequence to generate
        Returns:
            states: sequence of hidden states
            obs: sequence of observations
        '''
        states = np.zeros(T, dtype = np.int64)
        obs = np.zeros(T, dtype = np.int64)
        states[0] = np.random.choice(a=self.N, p=self.π)
        for t in range(1, T):
            states[t] = np.random.choice(a=self.N, p=self.A[states[t-1], :])
        for t in range(T):
            obs[t] = np.random.choice(a=self.B.shape[1], p=self.B[states[t], :])
        return states, obs
    
    def baum_welch_step(self, O):
        '''
        Performs a single step of the Baum-Welch EM algorithm for fitting HMM parameters.
        Parameters:
            O: sequence of observations
        Returns:
            newA: new transition matrix
            newB: new observation matrix
            newπ: new initial state distribution
            logP: log-probability of the observations under the new model
        '''
        T = len(O)
        α, c = self.forward(O)
        β = self.backward(O, c)
        
        ϝ = np.zeros((self.N, self.N, T-1))
        Y = np.zeros((T, self.B.shape[1]))
        γ = α * β / c
        
        newA = np.zeros_like(self.A)
        newB = np.zeros_like(self.B)
        newπ = np.zeros_like(self.π)
        
        for j in range(self.N):
            ϝ[:, j, :] = (self.A[:, j].reshape(self.N,1) * α[:, :T-1])
            ϝ[:, j, :] *= β[j,1:]
            ϝ[:, j, :] *= self.B[j, O[1:]]
            
        for i in range(self.N):
            for j in range(self.N):
                newA[i, j] = np.sum(ϝ[i, j, :T-1]) / np.sum(γ[i, :T-1])
        
        for i in range(self.N):
            for j in range(self.B.shape[1]):
                for t in range(T):
                    if O[t] == j:
                        newB[i, j] += γ[i, t]
                newB[i,j] /= np.sum(γ[i, :])
        newπ = γ[:,0]

        logP = -np.sum(np.log(c))
        return newA, newB, newπ, logP
        
    def fit(self, O, max_iter=500, tol = 1e-5, verbose = False):
        '''
        Runs the Baum-Welch EM algorithm to fit the HMM to an observed sequence.
        Parameters:
            O: sequence of observations
            steps: maximum number of iterations to run
            tol: tolerance; stop early if delta log-probability falls below this threshold
            verbose: if True, print the delta log-probability after each step
        Returns: None (modifies model parameters in-place)
        '''
        _, c = self.forward(O)
        oldlogP = -np.sum(np.log(c))
        print('Initial log-probability:', oldlogP)
        for i in range(max_iter):
            newA, newB, newπ, logP = self.baum_welch_step(O)
            if verbose:
                print(('\rStep ' + str(i) + ';').ljust(10)
                      + ('delta log-probability: ' + str(round(logP - oldlogP, 6))).rjust(35),
                      end='')
            if i > 0 and (logP - oldlogP < tol):
                print('\nTolerance reached.')
                break
            self.A = newA
            self.B = newB
            self.π = newπ
            oldlogP = logP
        else:
            print('\nMaximum iterations reached.')
        print('Final log-probability:', logP)
    
    @staticmethod
    def init_matrix(n, m, scale = 0.15):
        '''
        Static method for initializing matrices for fitting.
        '''
        A = np.ones((n, m)) + scale * np.random.randn(n, m)
        for i in range(n):
            A[i,:] /= np.sum(A[i,:])
        return A
    
class GaussianHMM:
    '''
    A modification of the previous class for Gaussian emissions. Implements filtering and smoothing, but not yet fitting.
    '''
    def __init__(self, A, means, covs, π):
        '''
        Initialize the model.
        Parameters:
            A: transition matrix
            means: sequence of mean vectors for observations
            covs: sequence of covariance matrices for observations
            π: initial state distribution
        Returns: None
        '''
        self.A = A
        self.means = means
        self.covs = covs
        self.π = π
        self.N = len(π)
    
    def forward(self, O):
        '''
        Runs the forward pass algorithm, used in filtering and smoothing.
        Parameters:
            O: a sequence of observations
            scale: boolean, determines whether to rescale each step. Should be True (default) for longer sequences to avoid underflow.
        Returns:
            α: array
            c: array, if scale is True
        '''
        T = len(O)
        α = np.zeros((self.N, T))
        c = np.zeros(T)
        for i in range(self.N):
            α[i, 0] = self.π[i] * sp.stats.multivariate_normal(mean=self.means[i], cov=self.covs[i]).pdf(O[0])
        c[0] = 1 / np.sum(α[:,0])
        α[:, 0] *= c[0]
        t = 1
        while t < T:
            for i in range(self.N):
                α[i, t] = np.sum(α[:, t-1] * self.A[:, i] * sp.stats.multivariate_normal(mean=self.means[i], cov=self.covs[i]).pdf(O[t]))
            c[t] = 1 / np.sum(α[:,t])
            α[:, t] *= c[t]
            t += 1
        return α, c
    
    def backward(self, O, c):
        '''
        Runs the backward pass algorithm, used in smoothing. Rescales at each step if an array of scaling parameters (from the forward pass) is passed.
        Parameters:
            O: a sequence of observations
            c: an array of scaling parameters
        Returns:
            β: backward pass parameters
        '''
        T = len(O)
        β = np.zeros((self.N, T))
        for i in range(self.N):
            β[i, T-1] = 1
        β[:, T - 1] *= c[T-1]
        t = T-1
        while t > 0:
            for i in range(self.N):
                v = np.array([sp.stats.multivariate_normal(mean=self.means[j], cov=self.covs[j]).pdf(O[t]) for j in range(self.N)])
                β[i, t-1] = np.sum(β[:, t] * self.A[i, :] * v[:])
            β[:, t-1] *= c[t-1]
            t -= 1
        return β
    
    def filter(self, O):
        '''
        Returns the estimated distribution of states at time T, given observations O.
        Parameters:
            O: sequence of observations
            scale: boolean, determines whether to rescale each step. Should be True (default) for longer sequences to avoid underflow.
        Returns:
            Array of probabilities
        '''
        T = len(O)
        α = self.forward(O)
        return α[:, T - 1] / np.sum(α[:, T - 1])
    
    def smooth(self, O):
        '''
        Returns the estimated distribution of states at all times, given observations O.
        Parameters:
            O: sequence of observations
            scale: boolean, determines whether to rescale each step. Should be True (default) for longer sequences to avoid underflow.
        Returns:
            Array of probabilities
        '''
        T = len(O)
        α, c = self.forward(O)
        β = self.backward(O, c)
        norm = sum(α[:, T-1])
        return α * β / c
    
    def generate(self, T):
        '''
        Generate simulated data given the current model parameters.
        Parameters:
            T: length of sequence to generate
        Returns:
            states: sequence of hidden states
            obs: sequence of observations
        '''
        states = np.zeros(T, dtype = np.int64)
        obs = []
        states[0] = np.random.choice(a=self.N, p=self.π)
        for t in range(1, T):
            states[t] = np.random.choice(a=self.N, p=self.A[states[t-1]])
        for t in range(T):
            obs.append(sp.stats.multivariate_normal(mean=self.means[states[t]], cov=self.covs[states[t]]).rvs())
        return states, np.array(obs)
