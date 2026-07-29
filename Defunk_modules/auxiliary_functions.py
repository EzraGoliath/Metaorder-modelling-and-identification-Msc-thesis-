#!/usr/bin/env python
# coding: utf-8

# In[25]:


import random
import math
import pandas as pd
import numpy as np
from pandas import StringDtype

# from scipy.stats import zipf


# In[27]:


# mapping function

# we need a power law, use zipf package to get samples from a power law

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


# In[29]:


# creates the cumulative probabilities

def cumulative_probs(participation):
    p_i = participation / sum(participation)
    c_0 = [0]
    c_i = np.cumsum(p_i)
    c   = np.concatenate((c_0, c_i))
    return c


# In[31]:


# assigns the trades to the specific traders

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


# In[33]:


# a metaorder is a sequence of consequitive trades of same sign by same trader in the same session

def metaorders(traders_trades):

    if traders_trades.shape[0] <= 1:
        trades = pd.DataFrame(columns = traders_trades.columns)
        return [trades]
        
    diffs   = np.diff(traders_trades['Trade Sign'])
    indices = np.where(diffs != 0)[0] + 1

    num_metaorders = len(indices) + 1

    traders_metaorders = []

    if len(indices) == 0:
        trades = pd.DataFrame(columns = traders_trades.columns)
        traders_metaorders.append(trades)
        return traders_metaorders
        
    for i in range(num_metaorders):

        if i == 0:
            trades = traders_trades.iloc[0:indices[i], ]
            n = trades.shape[0]
            if n == 1:
                continue
        elif i == len(indices):
            trades = traders_trades.iloc[indices[i-1]:traders_trades.shape[0], ]
            n = trades.shape[0]
            if n == 1:
                continue
        else:
            trades = traders_trades.iloc[(indices[i-1]):indices[i], ]
            n = trades.shape[0]
            if n == 1:
                continue
            # traders_metaorders[i].append(trades)
        
        traders_metaorders.append(trades)
    
    return traders_metaorders


# In[35]:


# will need to find the midprice before and after. This function does that
# note that here the end midprice is the price just before the next trade, not the price immediately after.

# what if i take the mid price just after?
def info_around_metaorder(metaorder, data, method = 'immediate'):

    if metaorder.empty:
        brackets = pd.DataFrame(columns = metaorder.columns)
        return brackets
        
    # Get the start and end row positions in the original data
    # metaorder = trader_1_metaorders[1]
    start_idx = metaorder.iloc[0]['Row']
    end_idx   = metaorder.iloc[-1]['Row']

    # Find the first non-trade before the metaorder
    #before_trade = data.loc[:start_idx - 1]
    before_trade = data[data['Row'] < start_idx]
    before       = before_trade[before_trade['Type'] != ' Trade'].tail(1) # the values in the Type col are ' Quote' and ' Trade'

    if method == 'immediate':
        after_trade = data[data['Row'] > end_idx]
        after = after_trade[after_trade['Type'] != ' Trade'].head(1)
    
    # Find the first non-trade after the metaorder
    #after_trade = data.loc[end_idx + 1:]
    if method == 'lagged':
        after_trade = data[data['Row'] > end_idx]
        interim     = after_trade[after_trade['Type'] != ' Trade']
        diffs       = np.diff(interim['Row'])
        indices     = np.where(diffs != 1)
                
        if len(indices[0]) == 0:
            after = before.copy()
        else:
            index = indices[0][0]
            after = interim.iloc[[index]]  

    brackets = pd.DataFrame(index = ['before', 'after'])
    brackets = pd.DataFrame(columns = before.columns)
    brackets.loc['before'] = before.iloc[0]
    brackets.loc['after'] = after.iloc[0]
    
    return brackets


# In[37]:


# Different ways to measure the impact

# the problem with measuring market impact for a metaorder is that other trades happen between the trades in the metaorder
# so what is the right way to measure the market impact?

# ave_execution_shortfall, vectorized

def ave_execution_shortfall(metaorder):

    p_s         = metaorder['Mid-price before'].iloc[0]
    Q           = metaorder['Volume'].sum()
    trade_sign  = metaorder.iloc[0]['Trade Sign']

    ave_execution_price = (metaorder['Volume'] * metaorder['Price']).sum() / Q
    impact              = trade_sign * (np.log(ave_execution_price) - np.log(p_s))
    return impact


# In[39]:


# faster per_trade_shortfall

def per_trade_shortfall(metaorder, timing_method = 'immediate'):

    if timing_method == 'immediate':

        p_s = np.log(metaorder['Mid-price before'])
        p_e = np.log(metaorder['Mid-price after(immediate)'])
        trade_sign    = metaorder['Trade Sign'].iloc[0]
        shortfalls    = (p_e - p_s) * metaorder['Volume']
        ave_shortfall = shortfalls.sum() / metaorder['Volume'].sum()
        impact        = trade_sign * ave_shortfall
        return impact

    if timing_method == 'delayed':

        p_s = np.log(metaorder['Mid-price before'])
        p_e = np.log(metaorder['Mid-price after(delayed)'])
        trade_sign    = metaorder['Trade Sign'].iloc[0]
        shortfalls    = (p_e - p_s) * metaorder['Volume']
        ave_shortfall = shortfalls / metaorder['Volume'].sum()
        impact        = trade_sign * ave_shortfall
        return impact


# In[41]:


def simple_impact(metaorder, timing_method = 'immediate'):

    if timing_method == 'immediate':

        p_s        = np.log(metaorder['Mid-price before'].iloc[0])
        p_e        = np.log(metaorder['Mid-price after(immediate)'].iloc[-1])
        trade_sign = metaorder['Trade Sign'].iloc[0]
        impact     = trade_sign * (p_e - p_s)
        return impact

    if timing_method == 'delayed':
        
        p_s        = np.log(metaorder['Mid-price before'].iloc[0])
        p_e        = np.log(metaorder['Mid-price after(delayed)'].iloc[-1])
        trade_sign = metaorder['Trade Sign'].iloc[0]
        impact     = trade_sign * (p_e - p_s)
        return impact


# In[43]:


def impact(metaorder, timing_method = 'immediate', impact_method = 'simple'): #, data
    
    if metaorder.empty:
        brackets = pd.DataFrame(columns = metaorder.columns)
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


# In[45]:


def impact_df(metaorders_list, timing_method = 'immediate', impact_method = 'simple'): 
    
    num_trades        = len(metaorders_list)

    if impact_method == 'all':
        features = pd.DataFrame(columns = ['RIC', 'Date', 'Start time', 'End time', 'daily volume', 'intraday volatility',
                                           'number child orders', 'volume traded', 'trade sign', 'impact(shortfall)',
                                           'impact(ave per trade)','impact(simple)'])
    
        for i in range(num_trades):
            metaorder = metaorders_list[i]
    
            if metaorder.empty:
                continue
    
            intention               = metaorder.iloc[0]['Trade Sign']
            ave_shortfall_execution = impact(metaorder, timing_method = timing_method, impact_method = 'average execution')
            ave_shortfall_per_trade = impact(metaorder, timing_method = timing_method, impact_method = 'per trade shortfall')
            simple_impact           = impact(metaorder, timing_method = timing_method, impact_method = 'simple')
            n                       = metaorder.shape[0]
            volume                  = sum(metaorder.loc[:, 'Volume'])

            features.at[i, 'RIC']                   = metaorder['RIC'].iloc[0]
            features.at[i, 'Date']                  = metaorder['Date'].iloc[0]
            features.at[i, 'Start time']            = metaorder['DateTime'].iloc[0]
            features.at[i, 'End time']              = metaorder['DateTime'].iloc[-1]
            features.at[i, 'daily volume']          = metaorder['Daily Volume'].iloc[0]
            features.at[i, 'intraday volatility']   = metaorder['Daily Volatility'].iloc[0]
            features.at[i, 'number child orders']   = n
            features.at[i, 'volume traded']         = volume
            features.at[i, 'trade sign']            = intention
            features.at[i, 'impact(shortfall)']     = ave_shortfall_execution
            features.at[i, 'impact(ave per trade)'] = ave_shortfall_per_trade
            features.at[i, 'impact(simple)']        = simple_impact
            
        return features

    else:
        features = pd.DataFrame(columns = ['RIC', 'Date', 'Start time', 'End time', 'daily volume', 'intraday volatility',
                                           'number child orders', 'volume traded', 'trade sign', 'impact'])
        for i in range(num_trades):
            metaorder = metaorders_list[i]
    
            if metaorder.empty:
                continue
    
            intention  = metaorder.iloc[0]['Trade Sign']
            ave_impact = impact(metaorder, timing_method = timing_method, impact_method = impact_method)
            n          = metaorder.shape[0]
            volume     = sum(metaorder.loc[:, 'Volume'])

            features.at[i, 'RIC']                 = metaorder['RIC'].iloc[0]
            features.at[i, 'Date']                = metaorder['Date'].iloc[0]
            features.at[i, 'Start time']          = metaorder['DateTime'].iloc[0]
            features.at[i, 'End time']            = metaorder['DateTime'].iloc[-1]
            features.at[i, 'daily volume']        = metaorder['Daily Volume'].iloc[0]
            features.at[i, 'intraday volatility'] = metaorder['Daily Volatility'].iloc[0]
            features.at[i, 'number child orders'] = n
            features.at[i, 'volume traded']       = volume
            features.at[i, 'trade sign']          = intention
            features.at[i, 'impact']              = ave_impact
    
        return features


# In[47]:


#!jupyter nbconvert --to script auxiliary_functions.ipynb --output-dir='/Users/ezragoliath/Desktop/Masters thesis/code/modules'

