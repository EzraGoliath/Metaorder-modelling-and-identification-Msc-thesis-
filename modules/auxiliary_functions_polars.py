#!/usr/bin/env python
# coding: utf-8

# In[7]:


import random
import math
import polars as pl
import numpy as np
from pandas import StringDtype


# In[3]:


def trader_participation(N, alpha = 2, f_min = 1, f_max = 1000, method = 'homogenous', seed = 1):

    if method == 'power':
        np.random.seed(seed)
        u = np.random.uniform(0, 1, N)
        samples = f_min * (1 - u) ** (-1 / (alpha - 1))
        samples = np.clip(samples, f_min, f_max)
        samples = np.round(samples).astype(int)
        return samples

    elif method == 'uniform':
        np.random.seed(seed)
        samples = np.random.uniform(f_min, f_max, N)
        samples = np.round(samples).astype(int)
        return samples

    elif method == 'homogenous':
        samples = np.linspace(f_min, f_max, N + 1)
        samples = samples[samples > f_min]
        return samples


# In[5]:


def cumulative_probs(participation):
    p_i = participation / sum(participation)
    c_0 = [0]
    c_i = np.cumsum(p_i)
    c   = np.concatenate((c_0, c_i))
    return c


# In[7]:


def orders(N, trades, cumulative_probs):
    
    number_trades = trades.shape[0]
    assignments   = [[] for _ in range(N)]
    orders        = np.zeros(N)
    
    for i in range(number_trades):
        np.random.seed(i)
        u            = np.random.uniform(0 + 1e-10, 1 - 1e-10)
        trader_index = np.searchsorted(cumulative_probs, u, side = 'right')
        
        #if trader_index == N:
        #    trader_index = N - 1
            
        orders[trader_index - 1] += 1 # paper doesnt specify what to do if u = c_i, use - 1 bcs cumulative probs and indexes are offset by 1
                                      # should i use side = 'left' so that i dont get a value thats out of bounds?
        assignments[trader_index - 1].append(i)

    return assignments


# In[1]:


def metaorders(traders_trades):

    if traders_trades.shape[0] <= 1:
        #trades = pl.DataFrame([], schema = traders_trades.columns)
        return [traders_trades]
        
    diffs   = np.diff(traders_trades['Trade Sign'])
    indices = np.where(diffs != 0)[0] + 1

    num_metaorders = len(indices) + 1

    traders_metaorders = []

    if len(indices) == 0:
        #trades = pl.DataFrame([], schema = traders_trades.columns)
        #traders_metaorders.append(trades)
        return [traders_trades]
        
    for i in range(num_metaorders):

        if i == 0:
            trades = traders_trades[0:indices[i], : ]
            n = trades.shape[0]
            if n == 1:
                continue
        elif i == len(indices):
            trades = traders_trades[indices[i-1]:traders_trades.shape[0], : ]
            n = trades.shape[0]
            if n == 1:
                continue
        else:
            trades = traders_trades[(indices[i-1]):indices[i], : ]
            n = trades.shape[0]
            if n == 1:
                continue
            # traders_metaorders[i].append(trades)
        
        traders_metaorders.append(trades)
    
    return traders_metaorders


# In[19]:


# Different ways to measure the impact

# the problem with measuring market impact for a metaorder is that other trades happen between the trades in the metaorder
# so what is the right way to measure the market impact?

# ave_execution_shortfall, vectorized

def ave_execution_shortfall(metaorder):

    p_s         = metaorder['Mid-price before'][0]
    Q           = metaorder['Volume'].sum()
    trade_sign  = metaorder['Trade Sign'][0]

    ave_execution_price = (metaorder['Volume'] * metaorder['Price']).sum() / Q
    impact              = trade_sign * (np.log(ave_execution_price) - np.log(p_s))
    return impact


# In[21]:


# faster per_trade_shortfall

def per_trade_shortfall(metaorder, timing_method = 'immediate'):

    if timing_method == 'immediate':

        p_s = np.log(metaorder['Mid-price before'])
        p_e = np.log(metaorder['Mid-price after(immediate)'])
        trade_sign    = metaorder['Trade Sign'][0]
        shortfalls    = (p_e - p_s) * metaorder['Volume']
        ave_shortfall = shortfalls.sum() / metaorder['Volume'].sum()
        impact        = trade_sign * ave_shortfall
        return impact

    if timing_method == 'delayed':

        p_s = np.log(metaorder['Mid-price before'])
        p_e = np.log(metaorder['Mid-price after(delayed)'])
        trade_sign    = metaorder['Trade Sign'][0]
        shortfalls    = (p_e - p_s) * metaorder['Volume']
        ave_shortfall = shortfalls / metaorder['Volume'].sum()
        impact        = trade_sign * ave_shortfall
        return impact


# In[23]:


def simple_impact(metaorder, timing_method = 'immediate'):

    if timing_method == 'immediate':

        p_s        = np.log(metaorder['Mid-price before'][0])
        p_e        = np.log(metaorder['Mid-price after(immediate)'][-1])
        trade_sign = metaorder['Trade Sign'][0]
        impact     = trade_sign * (p_e - p_s)
        return impact

    if timing_method == 'delayed':
        
        p_s        = np.log(metaorder['Mid-price before'][0])
        p_e        = np.log(metaorder['Mid-price after(delayed)'][-1])
        trade_sign = metaorder['Trade Sign'][0]
        impact     = trade_sign * (p_e - p_s)
        return impact


# In[25]:


def impact(metaorder, timing_method = 'immediate', impact_method = 'simple'):
    
    if metaorder.is_empty():
        brackets = pl.DataFrame([], schema = metaorder.columns)
        return brackets
        
    if impact_method == 'average execution':
        impact = ave_execution_shortfall(metaorder)
        return impact
        
    if impact_method == 'per trade shortfall':
        impact = per_trade_shortfall(metaorder, timing_method = timing_method)
        return impact 

    if impact_method == 'simple':
        impact = simple_impact(metaorder, timing_method = timing_method)
        return impact


# In[27]:


def impact_df(metaorders_list, timing_method = 'immediate', impact_method = 'simple'): 
    
    num_metaorders = len(metaorders_list)

    if impact_method == 'all':
        #features = pl.DataFrame([], schema = {'RIC', 'Date', 'Start time', 'End time', 'daily volume', 'intraday volatility',
        #                                   'number child orders', 'volume traded', 'trade sign', 'impact(shortfall)',
        #                                   'impact(ave per trade)','impact(simple)'})

        rows = []
        
        for i in range(num_metaorders):
            metaorder = metaorders_list[i]
            metaorder = metaorder.with_columns([pl.col('Mid-price before').fill_null(float('nan')),
                                    pl.col('Mid-price after(immediate)').fill_null(float('nan'))])
            if metaorder.is_empty():
                continue
    
            intention               = metaorder[0, 'Trade Sign']
            ave_shortfall_execution = impact(metaorder, timing_method = timing_method, impact_method = 'average execution')
            ave_shortfall_per_trade = impact(metaorder, timing_method = timing_method, impact_method = 'per trade shortfall')
            simple_impact           = impact(metaorder, timing_method = timing_method, impact_method = 'simple')
            n                       = metaorder.shape[0]
            volume                  = metaorder['Volume'].sum()

            #features[i, 'RIC']                   = metaorder['RIC'][0]
            #features[i, 'Date']                  = metaorder['Date'][0]
            #features[i, 'Start time']            = metaorder['DateTime'][0]
            #features[i, 'End time']              = metaorder['DateTime'][-1]
            #features[i, 'daily volume']          = metaorder['Daily Volume'][0]
            #features[i, 'intraday volatility']   = metaorder['Daily Volatility'][0]
            #features[i, 'number child orders']   = n
            #features[i, 'volume traded']         = volume
            #features[i, 'trade sign']            = intention
            #features[i, 'impact(shortfall)']     = ave_shortfall_execution
            #features[i, 'impact(ave per trade)'] = ave_shortfall_per_trade
            #features[i, 'impact(simple)']        = simple_impact
            rows.append({
                'RIC'                   : metaorder['RIC'][0],
                'Date'                  : metaorder['Date'][0],
                'Start time'            : metaorder['DateTime'][0],
                'End time'              : metaorder['DateTime'][-1],
                'daily volume'          : metaorder['Daily Volume'][0],
                'intraday volatility'   : metaorder['Daily Volatility'][0],
                'number child orders'   : n,
                'volume traded'         : volume,
                'trade sign'            : intention,
                'impact (shortfall)'    : ave_shortfall_execution,
                'impact (ave per trade)': ave_shortfall_per_trade,
                'impact (simple)'       : simple_impact})

        schema = {
            'RIC'                : pl.String,
            'Date'               : pl.Date,
            'Start time'         : pl.Datetime('us'),
            'End time'           : pl.Datetime('us'),
            'daily volume'       : pl.Float64,
            'intraday volatility': pl.Float64,
            'number child orders': pl.Int64,
            'volume traded'      : pl.Float64,
            'trade sign'         : pl.Int64,
            'impact'             : pl.Float64}

        features = pl.DataFrame(rows, schema = schema).sort('Start time', descending = False)
        return features

    else:
        #features = pl.DataFrame([], schema = {'RIC', 'Date', 'Start time', 'End time', 'daily volume', 'intraday volatility',
        #                                   'number child orders', 'volume traded', 'trade sign', 'impact'})
        rows = []
        for i in range(num_metaorders):
            metaorder = metaorders_list[i]
            metaorder = metaorder.with_columns([pl.col('Mid-price before').fill_null(float('nan')),
                                    pl.col('Mid-price after(immediate)').fill_null(float('nan'))])    
            if metaorder.is_empty():
                continue
    
            intention  = metaorder[0, 'Trade Sign']
            ave_impact = impact(metaorder, timing_method = timing_method, impact_method = impact_method)
            n          = metaorder.shape[0]
            volume     = metaorder['Volume'].sum()

            #features[i, 'RIC']                 = metaorder['RIC'][0]
            #features[i, 'Date']                = metaorder['Date'][0]
            #features[i, 'Start time']          = metaorder['DateTime'][0]
            #features[i, 'End time']            = metaorder['DateTime'][-1]
            #features[i, 'daily volume']        = metaorder['Daily Volume'][0]
            #features[i, 'intraday volatility'] = metaorder['Daily Volatility'][0]
            #features[i, 'number child orders'] = n
            #features[i, 'volume traded']       = volume
            #features[i, 'trade sign']          = intention
            #features[i, 'impact']              = ave_impact
            rows.append({
                'RIC'                : metaorder['RIC'][0],
                'Date'               : metaorder['Date'][0],
                'Start time'         : metaorder['DateTime'][0],
                'End time'           : metaorder['DateTime'][-1],
                'daily volume'       : metaorder['Daily Volume'][0],
                'intraday volatility': metaorder['Daily Volatility'][0],
                'number child orders': n,
                'volume traded'      : volume,
                'trade sign'         : intention,
                'impact'             : ave_impact})
            
        schema = {
            'RIC'                : pl.String,
            'Date'               : pl.Date,
            'Start time'         : pl.Datetime('us'),
            'End time'           : pl.Datetime('us'),
            'daily volume'       : pl.Float64,
            'intraday volatility': pl.Float64,
            'number child orders': pl.Int64,
            'volume traded'      : pl.Float64,
            'trade sign'         : pl.Int64,
            'impact'             : pl.Float64}

        features = pl.DataFrame(rows, schema = schema).sort('Start time', descending = False)
        return features


# In[1]:


#!jupyter nbconvert --to script auxiliary_functions_polars.ipynb --output-dir='/Users/ezragoliath/Desktop/Masters thesis/code/modules'

