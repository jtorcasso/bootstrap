'''Bootstrap: Tests

Issues
------


Author:  Jake Torcasso
License: BSD (3-clause)

'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
from scipy.stats import norm
sys.path.insert(0,'/home/jake/Documents/repos/software/bootstrap')
from sampling import bootstrap
import time

np.random.seed(1234)
np.set_printoptions(precision = 4, suppress=True)

data = np.random.randn(100)*5 + 0.6

def mean(data, add=0):

    return data.mean() + add

start = time.time()
boot = bootstrap(mean, data, 1000, threads=1)
print '1 Thread in:', time.time() - start
print 'True:\n'
print 2*norm.sf(data.mean()/(data.std()/np.sqrt(len(data))))
print 'Bootstrapped:\n'
print boot.pvalue(method='normal',twosided=True,null=0)
print boot.pvalue(method='shift',twosided=True,null=0)
print boot.pvalue(method='normal',twosided=False,tail='right',null=0)
print boot.pvalue(method='shift',twosided=False,tail='right',null=0)
print boot.pvalue(method='normal',twosided=False,tail='left',null=0)
print boot.pvalue(method='shift',twosided=False,tail='left',null=0)

start = time.time()
boot = bootstrap(mean, data, 1000, threads=1, fargs={'add':0.2})
print '2 Threads in:', time.time() - start
print 'True:\n'
print 2*norm.sf(data.mean()/(data.std()/np.sqrt(len(data))))
print 'Bootstrapped:\n'
print boot.pvalue(method='normal',twosided=True,null=0)
print boot.pvalue(method='shift',twosided=True,null=0)
print boot.pvalue(method='normal',twosided=False,tail='right',null=0)
print boot.pvalue(method='shift',twosided=False,tail='right',null=0)
print boot.pvalue(method='normal',twosided=False,tail='left',null=0)
print boot.pvalue(method='shift',twosided=False,tail='left',null=0)

start = time.time()
boot = bootstrap(mean, data, 1000, threads=2)
print '2 Threads in:', time.time() - start
print 'True:\n'
print 2*norm.sf(data.mean()/(data.std()/np.sqrt(len(data))))
print 'Bootstrapped:\n'
print boot.pvalue(method='normal',twosided=True,null=0)
print boot.pvalue(method='shift',twosided=True,null=0)
print boot.pvalue(method='normal',twosided=False,tail='right',null=0)
print boot.pvalue(method='shift',twosided=False,tail='right',null=0)
print boot.pvalue(method='normal',twosided=False,tail='left',null=0)
print boot.pvalue(method='shift',twosided=False,tail='left',null=0)

N = 1500
b = 20 + 20*np.random.randn(N)
c = np.random.randn(N)
d = 32 + 16*np.random.randn(N)
constant = np.ones(N)

a = 2*constant - 4*b + d + np.random.randn(N)*2

data = np.vstack((a,constant,b,c,d))
data = data.transpose()
tails = np.array(['right', 'left', 'right', 'right'])

def simpleOLS(X):
    y = X[:,0]
    x = X[:,1:]
    
    results = sm.OLS(y,x).fit()

    B = results.params

    return B


boot = bootstrap(simpleOLS, data, 1000, weight=5,threads=1)
print 'True:\n'
print sm.OLS(data[:,0],data[:,1:]).fit().pvalues
print 'Bootstrapped:\n'
print boot.pvalue(method='normal',twosided=True,null=0)
print boot.pvalue(method='shift',twosided=True,null=0)
print boot.pvalue(method='normal',twosided=False,tail=tails,null=0)
print boot.pvalue(method='shift',twosided=False,tail=tails,null=0)
print boot.pvalue(method='normal',twosided=False,tail=tails,null=0)
print boot.pvalue(method='shift',twosided=False,tail=tails,null=0)
print boot.pvalue(method='normal',twosided=False,tail='right',null=0)
print boot.pvalue(method='shift',twosided=False,tail='right',null=0)
print boot.pvalue(method='normal',twosided=False,tail='left',null=0)
print boot.pvalue(method='shift',twosided=False,tail='left',null=0)

#boot = bootstrap(simpleOLS, data, 1000, threads=2)
#print 'True:\n'
#print sm.OLS(data[:,0],data[:,1:]).fit().pvalues
#print 'Bootstrapped:\n'
#print boot.pvalue(method='normal',twosided=True,null=0)
#print boot.pvalue(method='shift',twosided=True,null=0)
#print boot.pvalue(method='normal',twosided=False,tail='right',null=0)
#print boot.pvalue(method='shift',twosided=False,tail='right',null=0)
#print boot.pvalue(method='normal',twosided=False,tail='left',null=0)
#print boot.pvalue(method='shift',twosided=False,tail='left',null=0)

data = pd.DataFrame(data)

def stackedOLS(X):
    y = X[0]
    x = X[[1,2,3,4]]
    
    results = sm.OLS(y,x).fit()

    B = results.params

    return np.vstack((B,B))

boot = bootstrap(stackedOLS, data, 1000, threads=1)
print 'True:\n'
pvals = sm.OLS(data[0],data[[1,2,3,4]]).fit().pvalues
print np.vstack((pvals,pvals)).T

print 'Bootstrapped:\n'
print boot.pvalue(method='normal',twosided=True,null=0)
print boot.pvalue(method='shift',twosided=True,null=0)
print boot.pvalue(method='normal',twosided=False,tail='right',null=0)
print boot.pvalue(method='shift',twosided=False,tail='right',null=0)
print boot.pvalue(method='normal',twosided=False,tail='left',null=0)
print boot.pvalue(method='shift',twosided=False,tail='left',null=0)
tails = np.array(['right', 'left', 'right', 'right'])
tails = np.vstack((tails,tails)).T
print boot.pvalue(method='normal',twosided=False,tail=tails,null=0)
print boot.pvalue(method='shift',twosided=False,tail=tails,null=0)
#boot = bootstrap(stackedOLS, data, 1000, threads=2)
#print 'True:\n'
#pvals = sm.OLS(data[:,0],data[:,1:]).fit().pvalues
#print np.vstack((pvals,pvals)).T
#
#print 'Bootstrapped:\n'
#print boot.pvalue(method='normal',twosided=True,null=0)
#print boot.pvalue(method='shift',twosided=True,null=0)
#print boot.pvalue(method='normal',twosided=False,tail='right',null=0)
#print boot.pvalue(method='shift',twosided=False,tail='right',null=0)
#print boot.pvalue(method='normal',twosided=False,tail='left',null=0)
#print boot.pvalue(method='shift',twosided=False,tail='left',null=0)

data['treat'] = np.random.randint(low=0,high=2,size=1500)
data['male'] = np.random.randint(low=0,high=2,size=1500)

boot = bootstrap(stackedOLS, data, 1000, by=['treat','male'], threads=1)
print 'True:\n'
pvals = sm.OLS(data[0],data[[1,2,3,4]]).fit().pvalues
print np.vstack((pvals,pvals)).T

print 'Bootstrapped:\n'
print boot.pvalue(method='normal',twosided=True,null=0)
print boot.pvalue(method='shift',twosided=True,null=0)
print boot.pvalue(method='normal',twosided=False,tail='right',null=0)
print boot.pvalue(method='shift',twosided=False,tail='right',null=0)
print boot.pvalue(method='normal',twosided=False,tail='left',null=0)
print boot.pvalue(method='shift',twosided=False,tail='left',null=0)
tails = np.array(['right', 'left', 'right', 'right'])
tails = np.vstack((tails,tails)).T
print boot.pvalue(method='normal',twosided=False,tail=tails,null=0)
print boot.pvalue(method='shift',twosided=False,tail=tails,null=0)

boot = bootstrap(stackedOLS, data, 1000, by=['treat'], threads=1)
print 'True:\n'
pvals = sm.OLS(data[0],data[[1,2,3,4]]).fit().pvalues
print np.vstack((pvals,pvals)).T

print 'Bootstrapped:\n'
print boot.pvalue(method='normal',twosided=True,null=0)
print boot.pvalue(method='shift',twosided=True,null=0)
print boot.pvalue(method='normal',twosided=False,tail='right',null=0)
print boot.pvalue(method='shift',twosided=False,tail='right',null=0)
print boot.pvalue(method='normal',twosided=False,tail='left',null=0)
print boot.pvalue(method='shift',twosided=False,tail='left',null=0)
tails = np.array(['right', 'left', 'right', 'right'])
tails = np.vstack((tails,tails)).T
print boot.pvalue(method='normal',twosided=False,tail=tails,null=0)
print boot.pvalue(method='shift',twosided=False,tail=tails,null=0)