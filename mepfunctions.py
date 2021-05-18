# %%
import os
import time
import numpy as np
from numpy import savetxt
from math import factorial, exp, log, gamma, pow, sqrt 
import scipy.sparse as scs
import scipy.stats as sts
from scipy.special import digamma, gamma, loggamma
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import copy


# %%
def L2norm(x):
    return sqrt(np.sum(([x[i]**2 for i in range(len(x))])))

def L2dist(x,y):
    return sqrt(np.sum([(x[i]-y[i])**2 for i in range(len(x))]))

def logsumexp(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x-c))) 

def update_phi(lam,gam,i,j):
    exponent = np.zeros(K)
    #d = digamma(np.sum(gam[i]))
    for k in range(K): 
        exponent[k] = digamma(lam[k][j]) - digamma(np.sum(lam[k])) + digamma(gam[i][k])
    return np.exp(exponent - logsumexp(exponent))

def update_gam(phi,i):
    vec = np.sum(phi[i], axis = 0)
    #return alpha_vec + vec
    return alpha + vec
    
def update_lam(phi,k): #nonsparse
    vec = np.sum(np.multiply(x,phi[:,:,k]), axis=0) 
    #return np.add(eta_vec,vec) # V-vector
    return np.add(eta,vec)


def ELBO(x, phi,gam,lam):
    rows,cols = x.nonzero()
    term1 = 0
    for k in range(K):
        term1 += np.sum(digamma(lam[k])) - V*digamma(np.sum(lam[k])) 
    term1 = (eta-1)*term1  
    term2 = 0
    for i in range(D):
        term2 += (np.sum(digamma(gam[i])) - K*digamma(np.sum(gam[i])))  
    term2 = (alpha - 1)*term2
    
    term3 = 0 
    for i, j in zip(rows,cols):
        thing = digamma(np.sum(gam[i]))
        for k in range(K):
            term3 += x[i][j]*phi[i][j][k] * (digamma(gam[i][k]) - thing + digamma(lam[k][j])- digamma(np.sum(lam[k])) )  
    logjoint = term1+term2+term3
    term4 = 0
    for k in range(K):
        dot = np.dot(lam[k]-np.ones(V), digamma(lam[k]) - np.ones(V)*digamma(np.sum(lam[k]))) 
        term4 += -loggamma(np.sum(lam[k])) + np.sum(loggamma(lam[k])) - dot     
    term5 = 0
    for i in range(D):   
        dot5 = np.dot(gam[i]-np.ones(K), digamma(gam[i]) - np.ones(K)*digamma(np.sum(gam[i])))
        term5 += -loggamma(np.sum(gam[i])) + np.sum(loggamma(gam[i])) - dot5
    term6=0
    #phi = phi.clip(min=1e-200)
    for (i,j) in zip(rows,cols):
        p = phi[i][j]
        term6 -= x[i][j]*np.dot(p, np.log(p))  #for some reason term6 is inf after the 1st update
    #assert term6 < np.inf, 'term6 is inf' 
    
    negentropy = term4 + term5 + term6
    
    return logjoint + negentropy

def plot_cavi(x,gam, lam, phi, T): #cavi with information printed 

    rows,cols = x.nonzero()
    y=[]
    print('init elbo', ELBO(x, phi, gam,lam))
    for t in range(T):
        for i, j in zip(rows,cols):
            phi[i][j] = update_phi(lam,gam,i,j)
        for i in range(D):
            gam[i] = update_gam(phi,i)
        for k in range(K):
            lam[k] = update_lam(phi,k)
            
        elbo=ELBO(x, phi,gam,lam)
        y.append(elbo)
    
        if t>0 and t%50==0:
            print([y[i]-y[i-1] for i in range(t-5,t)], 'differences in last few elbos')
            print(y[t-1], 'elbo after', t, 'iters')
            
            plt.scatter(range(t-50,t), y[t-50:])
            plt.title('CAVI half')
            plt.xlabel('Iterations')
            plt.ylabel('ELBO')
            plt.show()
                
    return lam, gam, phi

def cavi(x, gam, lam, phi, T):
    rows,cols = x.nonzero()
    for t in range(T):
        for i, j in zip(rows,cols):
            phi[i][j] = update_phi(lam,gam,i,j)
        for i in range(D):
            gam[i] = update_gam(phi,i)
        for k in range(K):
            lam[k] = update_lam(phi,k)
    return lam, gam, phi



# %%
def l_ascent(bead,grad,stepsize,phi): #"bead" is lam for one bead
    return np.add(bead, [stepsize*val for val in grad(bead,phi)])
                       
def g_ascent(bead,grad,stepsize,phi):
    return np.add(bead, [stepsize*val for val in grad(bead,phi)])
                       
def p_ascent(bead,grad,stepsize, lam, gam): 
    return np.add(bead, [stepsize*val for val in grad(lam, gam, bead)])


def L2_dist(x,y):
    return sqrt(np.sum(np.square(np.subtract(x,y))))

def param_curve(beads, lengths, l): #returns position on curve at length l in [0,L]
    N=len(beads)
    for i in range(N):
        if i== N-1:
            return beads[N-1]
        if l > lengths[i]:
            l-= lengths[i]
        elif i < N-1:
            return beads[i] + l/lengths[i]*np.subtract(beads[i+1],beads[i])
    
def reparam_with_length(beads):
    N = len(beads)
    lengths = [L2_dist(beads[i],beads[i+1]) for i in range(N-1)] #so lengths[i] = length of segment(bead[i], bead[i+1])
    L = np.sum(lengths)
    
    return [param_curve(beads, lengths, i*L/(N-1)) for i in range(N)], L


def reparam(beads):
    N = len(beads)
    lengths = [L2_dist(beads[i],beads[i+1]) for i in range(N-1)] #so lengths[i] = length of segment(bead[i], bead[i+1])
    L = np.sum(lengths)
    
    return [param_curve(beads, lengths, i*L/(N-1)) for i in range(N)]

def cavi_string(lam_beads,gam_beads,phi_beads,iters):
    N = len(lam_beads)
    lengths = [0]
    
    for t in range(iters):
        for i in range(N):
            lam_beads[i], gam_beads[i], phi_beads[i] = cavi(x, gam_beads[i], lam_beads[i], phi_beads[i], 1)
            
        lam_beads, l = reparam_with_length(lam_beads)
        gam_beads = reparam(gam_beads)
        phi_beads = reparam(phi_beads)

        lengths.append(l)
        rat = (lengths[-1]-lengths[-2])/lengths[-1]
        print(rat)
        if rat < .01: #if change in length < 1%
            break
        
        print('iter', t)
        
    return lam_beads, gam_beads, phi_beads

# %%
