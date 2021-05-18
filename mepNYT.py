{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/Edith/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:59: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "import numpy as np\n",
    "import scipy.io\n",
    "import time\n",
    "import numpy as np\n",
    "from math import factorial, exp, log, gamma, pow, sqrt \n",
    "from scipy.special import digamma, gamma, loggamma\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import pickle\n",
    "import copy\n",
    "from scipy import sparse\n",
    "from collections import Counter\n",
    "\n",
    "\n",
    "with open('/Users/Edith/Downloads/nyt/vocab.pkl', 'rb') as f:\n",
    "    words = pickle.load(f) #212,237 unique words\n",
    "\n",
    "\n",
    "tr_counts = scipy.io.loadmat('/Users/Edith/Downloads/nyt/bow_tr_counts.mat')\n",
    "tr_tokens = scipy.io.loadmat('/Users/Edith/Downloads/nyt/bow_tr_tokens.mat')\n",
    "\n",
    "counts = tr_counts.get('counts')[0] #1,368,205 total documents\n",
    "tok = tr_tokens.get('tokens')[0]#nonzero vocab indices for each document\n",
    "\n",
    "#new corpus size: number of documents and vocab size\n",
    "D=500\n",
    "V=1000\n",
    "\n",
    "#turn into 2d array\n",
    "x= [counts[i][0] for i in range(D)]\n",
    "tok = [tok[i][0] for i in range(D)]\n",
    "\n",
    "#flatten x and tok\n",
    "flatx = [item for sublist in x for item in sublist] \n",
    "flattok = [item for sublist in tok for item in sublist]\n",
    "\n",
    "#sort words by frequency \n",
    "all = []\n",
    "for i in range(len(flatx)):\n",
    "    for j in range(flatx[i]):\n",
    "        all.append(flattok[i])\n",
    "pairs = Counter(all)\n",
    "\n",
    "#trim only most V most common words\n",
    "mostcommon = pairs.most_common(V)\n",
    "vocab = [p[0] for p in mostcommon]\n",
    "newtok = [[v if v in vocab else 0 for v in tok[i]] for i in range(D)]\n",
    "\n",
    "for i in range(D):\n",
    "    x[i] = [x[i][j] for j in np.array(newtok[i]).nonzero()]\n",
    "\n",
    "#final data: x is counts, newtok is new tokens vector \n",
    "x = np.array(x)\n",
    "x = [x[i][0] for i in range(D)]\n",
    "\n",
    "tok = [[vocab.index(v) for v in newtok[i] if v!=0] for i in range(D)]\n",
    "\n",
    "#for non-ragged arrays\n",
    "c = np.zeros((D,V))\n",
    "for i in range(D):\n",
    "    uniq = 0\n",
    "    for j in tok[i]:\n",
    "        c[i][j] = x[i][uniq]\n",
    "        uniq +=1 \n",
    "\n",
    "x = np.array(c)\n",
    "\n",
    "\n",
    "#delete unneeded stuff\n",
    "del tr_counts\n",
    "del tr_tokens\n",
    "del counts\n",
    "del newtok\n",
    "\n",
    "del c\n",
    "del tok\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "exec(open('/Users/Edith/Downloads/mepfunctions.py').read())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "K=25\n",
    "eta = .1 #np.random.uniform(.01,.1)\n",
    "alpha = 5/K \n",
    "\n",
    "eta_vec = np.ones(V)*eta\n",
    "alpha_vec = np.ones(K)*alpha\n",
    "\n",
    "lam1 = np.random.rand(K,V)\n",
    "gam1 = np.random.rand(D,K)\n",
    "\n",
    "phi1 = np.zeros((D,V,K))\n",
    "for i in range(D):\n",
    "    for j in x[i].nonzero():\n",
    "        phi1[i][j] = np.ones(K)*1/K\n",
    "\n",
    "lam2 = np.random.rand(K,V)\n",
    "gam2 = np.random.rand(D,K)\n",
    "phi2 = copy.deepcopy(phi1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "821.4071779251099\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "start = time.time()\n",
    "l_1,g_1,p_1 = cavi(x, gam1, lam1,phi1, 51)\n",
    "\n",
    "print(time.time()-start)\n",
    "\n",
    "l_2,g_2,p_2 = cavi(x, gam2, lam2,phi2, 51)\n",
    "\n",
    "l_diff = l_2 - l_1\n",
    "g_diff = g_2 - g_1\n",
    "p_diff = p_2 - p_1\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "N=15\n",
    "\n",
    "lam_beads = [l_1 + t*l_diff for t in np.linspace(0,1,N)] #initial beads along line segment\n",
    "phi_beads = [p_1 + t*p_diff for t in np.linspace(0,1,N)]\n",
    "gam_beads = [g_1 + t*g_diff for t in np.linspace(0,1,N)]\n",
    "\n",
    "start = time.time()\n",
    "lcmep, gcmep, pcmep = cavi_string(lam_beads,gam_beads,phi_beads,21)  \n",
    "print(time.time()-start, 'time')\n",
    "\n",
    "lcmep, gcmep, pcmep = cavi_string(lcmep, gcmep, pcmep,11)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "\n",
    "y_1 = []\n",
    "\n",
    "M=len(lcmep)\n",
    "iter_range = range(M)\n",
    "for d in iter_range: #straight line between \n",
    "    y_1.append(ELBO(x, p_1+d/(M-1)*p_diff, g_1+d/(M-1)*g_diff, l_1+d/(M-1)*l_diff))\n",
    "\n",
    "\n",
    "z=[]\n",
    "for i in range(M):\n",
    "    z.append(ELBO(x, pcmep[i], gcmep[i], lcmep[i]))\n",
    "\n",
    "cmep = plt.scatter(range(len(z)),z,color = 'g')\n",
    "cmepline = plt.plot(range(len(z)),z, color = 'g')\n",
    "\n",
    "cs = plt.scatter(iter_range, y_1, color='b')\n",
    "csline = plt.plot(iter_range,y_1, color = 'b')\n",
    "\n",
    "plt.title('Paths between maxima')\n",
    "plt.ylabel('ELBO')\n",
    "plt.xlabel('Position along path')\n",
    "plt.legend((cs, cmep),\n",
    "           ('cs', 'cmep'),\n",
    "           scatterpoints=1,\n",
    "           loc='best',\n",
    "           fontsize=8)\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(max(z),max(z[0],z[-1]), min(z))\n",
    "print(max(z)/max(z[0],z[-1])) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "'''file = open('lcmep' + str(D) + ',' + str(V) + ',' + str(K) + '.txt', 'w') #shape of lcmep is NxKxV\n",
    "for row in lcmep:\n",
    "    np.savetxt(file,row)\n",
    "    \n",
    "file = open('gcmep' + str(D)+ ','  + str(V) + ',' + str(K) + ',' + '.txt', 'w') #shape of gcmep is NxDxK\n",
    "for row in gcmep:\n",
    "    np.savetxt(file,row)\n",
    "    \n",
    "file = open('pcmep' + str(D) + ',' + str(V) + ',' + str(K) + '.txt', 'w') #shape of pcmep is NxDxVxK\n",
    "for row in pcmep:\n",
    "    for val in row:\n",
    "        np.savetxt(file,val)\n",
    "    '''"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "mininds1 = [i for i in range(len(l_1[0])) if np.round(l_1[0][i],1) ==eta ] #indices of l_1 topic 1 that are near eta\n",
    "mininds2 = [i for i in range(len(l_2[0])) if np.round(l_2[0][i],1) ==eta ]\n",
    "\n",
    "print(len(mininds1), len(mininds2))\n",
    "overlap = list(set(mininds1) & set(mininds2)) # components that are near eta in both l_1 and l_2 topic 1\n",
    "print(len(overlap)/min(len(mininds1), len(mininds2)), 'percent of overlap')\n",
    "\n",
    "for b in range(N):\n",
    "    mi = [i for i in range(len(lcmep[b][0])) if np.round(lcmep[b][0][i],1) == eta]\n",
    "    print(len(mi), len(list(set(overlap) & set(mi))))\n",
    "    \n",
    "    ##seems that all the same components stay eta "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
