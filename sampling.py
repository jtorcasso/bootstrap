'''Bootstrap: Resampling Methods

Issues
------

1. Right now parallelization only works in some scenarios,
   it hangs up on the statsmodels OLS test

Author:  Jake Torcasso
License: BSD (3-clause)

'''
from __future__ import division

import numpy as np
import pandas as pd
from inference import BootstrapResult
from multiprocessing import Pool
import functools

def draw_index(data, size, by=None, seed=1234):
    '''sample indices from data

    Parameteters
    ------------
    data : array or dataframe
        data to be resampled
    size : int
        number of samples
    by : list
        list of columns in data to sample within, i.e.
        stratified random sampling within blocks defined
        by selected columns
    seed : int
        seed for pseudo-random number generator
    '''
    np.random.seed(1234)

    data = pd.DataFrame(data)

    N = len(data)

    if by is not None:

        assert isinstance(by, list)

        if not set(by).issubset(data.columns):
            raise ValueError('{} not in data columns'.format(by))

        cells = data[by].drop_duplicates().as_matrix()
        
        indices = []
        for i in range(size):
            reindex = np.array([])
            for c in cells:
                cell_index = (data[by]==c).all(axis=1)
                N_ = cell_index.sum()
                reindex = np.hstack((reindex, np.random.choice(\
                                     data[cell_index].index,size=N_)))
            indices.append(reindex)

        return indices

    else:
        return [np.random.choice(data.index, \
                                 size=N) for i in range(size)]


def bootstrap(f, data, size, fargs={}, by=None, **kwargs):
    '''construct a boostrap distribution of parameters

    Parameters
    ----------
    f : function
        function which produces a parameter from data.
    data : dict
        data to resample
    size : int
        number of bootstrap draws
    fargs : dict
        other keyword arguments for f
    by : list
        list of columns in data to statify sampling on

    kwargs
    ------
    seed : int
        seed for pseudo-random number generator
    threads : int
        number of threads to spread bootstrap on,
        default 1

    Returns
    -------
    sample : BootstrapResult instance
        object for exploring bootstrap distribution, including
        hypothesis testing

    Ex: 
    >>> import statsmodels.api as sm
    >>> import pandas as pd
    >>> import numpy as np
    >>> import bootstrap as bstrap
    >>> data = pd.DataFrame(np.ones((100,2)))
    >>> data.columns = ['income', 'ability']
    >>> data['ability'] = np.random.randn(100)
    >>> data['income'] = 4*data['ability']
    >>> def f(data, missing='drop'):
    ...     
    ...     endog = data['income']
    ...     exog = data['ability']
    ...     
    ...     fit = sm.OLS(endog,exog,missing=missing).fit()
    ...     
    ...     return fit.params
    >>> boot = bstrap.bootstrap(f, data, 1000, fargs={'missing':'drop'})
    Bootstrap succeeded on 1000 of 1000 draws
    >>> boot.pvalue()
    array([ 0.])
    '''
    accepted = ['seed', 'threads']
    for key in kwargs.keys():
        if key not in accepted:
            raise KeyError('unexpected keyword argument {}'.format(key))
    seed = 1234 if 'seed' not in kwargs else kwargs['seed']
    threads = 1 if 'threads' not in kwargs else kwargs['threads']

    indices = draw_index(data, size, by=by, seed=seed)

    if isinstance(data, (pd.DataFrame, pd.Series)):
        func = function_wrapper1
    elif isinstance(data, np.ndarray):
        func = function_wrapper2

    point = f(data, **fargs)

    types = (pd.Series, pd.DataFrame, np.ndarray, 
             int, float, np.int, np.float)

    if not isinstance(point, types):
        raise ValueError('function {} returns unsupported type {}'\
                         .format(f, type(point)))
    
    outf = [point]
    if threads == 1:
        for index in indices:
            outf.append(func(f,data,fargs,index))

    else:
        pool = Pool(threads)
        outf.extend(pool.map(functools.partial(func,f,data,fargs),indices))
        pool.close()
        pool.join()

    outf = [np.array(out).flat for out in outf if out is not None]
    print 'Bootstrap succeeded on {} of {} draws'.format(len(outf)-1,size)


    return BootstrapResult(np.vstack(outf), np.array(point).shape)

def function_wrapper1(f, data, fargs, index):
    '''wrapper for functions in bootstrapper

    for dataframes'''
    
    try:
        return f(data.reindex(index), **fargs)
    except:
        return None
        
def function_wrapper2(f, data, fargs, index):
    '''wrapper for functions in bootstrapper

    for numpy arrays'''
    
    try:
        return f(data[index], **fargs)
    except:
        return None