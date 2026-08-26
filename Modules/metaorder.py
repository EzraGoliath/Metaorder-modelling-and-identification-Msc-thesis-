#pip install 'powerlaw'

import bmll2 as b2
from bmll2 import reference, Security, NormalisedSecurity, SparkHelper, get_market_data, get_market_data_range, VenueMarketError, get_market_tables, save_spark_dataframe, load_spark_dataframe

b2.get_file('modules/LMF.py')
import LMF as LMF

import random
import math
import polars as pl
import numpy as np
from pandas import StringDtype
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import LogFormatterSciNotation

from statsmodels.sandbox.stats.runs import runstest_1samp 
import powerlaw
import itertools
import pylab
import scipy.stats
import scipy.fft
from scipy.optimize import curve_fit
from sklearn.model_selection import ParameterGrid

from statsmodels.tsa.stattools import acf
from scipy.stats import t
from scipy.special import gamma as gamma_func

def trader_participation(N, delta = 2, f_min = 1, f_max = 1000, trader_participation_distribution = 'homogenous', seed = 1):

    if trader_participation_distribution == 'power':
        np.random.seed(seed)
        u       = np.random.uniform(0, 1, N)
        samples = f_min * (1 - u) ** (-1 / (delta - 1))
        return samples

    elif trader_participation_distribution == 'homogenous':
        samples = np.linspace(f_min, f_max, N + 1)
        samples = samples[samples > f_min]
        return samples

def cumulative_probs(participation):
    
    p_i = participation / sum(participation)
    c_0 = [0]
    c_i = np.cumsum(p_i)
    c   = np.concatenate((c_0, c_i))
    return c

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

def metaorder_data(stock_data, group = 'day', N = 4, trader_participation_distribution = 'homogenous', delta = 2):

    if group == 'day':
        stock_data = stock_data.with_columns(pl.col('Date').alias('group key'))

    elif group == 'month':
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1mo').alias('group key'))
    
    elif group == 'year':
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1y').alias('group key'))

    groups = stock_data.select('group key').unique().sort('group key').to_series().to_list()

    if isinstance(N, list):
        if len(N) != len(groups):
            raise ValueError('Length of N list must match number of groups')
        N_map = dict(zip(groups, N))
    else:
        N_map = {g: N for g in groups}

    if isinstance(delta, list):
        if len(delta) != len(groups):
            raise ValueError('Length of delta list must match number of groups')
        delta_map = dict(zip(groups, delta))
    else:
        delta_map = {g: delta for g in groups}    
    
    features_list_pl = []
    
    for trades in stock_data.partition_by('group key'):

        if trades.is_empty():
            continue

        trades = trades.sort(['DateTime', 'ExchangeSequenceNo'])

        group_value = trades['group key'][0]
        N_val       = N_map[group_value]
        delta_val   = delta_map[group_value]
        
        f      = trader_participation(N = N_val, trader_participation_distribution = trader_participation_distribution, delta = delta_val, f_min = 1,
                                            f_max = trades.shape[0], seed = 1)
        c      = cumulative_probs(f)
        output = orders(N = N_val, trades = trades, cumulative_probs = c)
        
        assignments = []
        for trader_id, rows in enumerate(output):
            for r in rows:
                assignments.append((r, trader_id))

        assignments_df = pl.DataFrame(assignments, schema = ['row idx', 'trader id'], orient = 'row')

        trades = trades.with_row_index('row idx')
        trades = trades.join(assignments_df, on = 'row idx', how = 'inner')
        trades = trades.sort(['DateTime', 'ExchangeSequenceNo'])

        trades = trades.with_columns((pl.col('Trade Sign') != pl.col('Trade Sign').shift(1)).cast(pl.Int32)
            .cum_sum().over(['Date', 'trader id']).alias('metaorder id'))
        

        features = trades.group_by(['Date', 'trader id', 'metaorder id']
        ).agg([

            pl.col('Ticker').first(),
            pl.col('MIC').first(),
            pl.col('DateTime').first().alias('Start time'),
            pl.col('DateTime').last().alias('End time'),
            pl.col('Daily Volume').first(),

            pl.col('Intraday Volatility').first(),
            pl.col('20 AD volume').first(),
            pl.col('20 AD volatility').first(),

            pl.col('Trade Sign').first().alias('trade sign'),
            
            pl.col('Volume').sum().alias('volume traded'),
            pl.len().alias('number child orders'),
            
            pl.col('Mid-price before').first(),
            pl.col('Mid-price after').last(),
            pl.col('Mid-price after (delayed)').last(),
            
            #pl.col('Mid-price before (1ms)').first(),
            #pl.col('Mid-price after (1ms)').last(),
            #pl.col('Mid-price after (1ms)(delayed)').last()
        ])

        features = features.filter(pl.col('number child orders') > 1)

        features = features.with_columns([
            ((pl.col('Mid-price after').log() - pl.col('Mid-price before').log())
             * pl.col('trade sign')).alias('metaorder impact'),
        
            ((pl.col('Mid-price after (delayed)').log() - pl.col('Mid-price before').log())
             * pl.col('trade sign')).alias('metaorder impact (delayed)'),
        
            #((pl.col('Mid-price after (1ms)').log() - pl.col('Mid-price before (1ms)').log())
            # * pl.col('trade sign')).alias('impact (1ms)'),
        
            #((pl.col('Mid-price after (1ms)(delayed)').log() - pl.col('Mid-price before (1ms)').log())
            # * pl.col('trade sign')).alias('impact (1ms)(delayed)'), 
            
            (((pl.col('End time') - pl.col('Start time')).dt.total_microseconds() / 6e7).alias('duration (min)'))
        ])
        
        features_list_pl.append(features)

    if features_list_pl:
        return pl.concat(features_list_pl, how = 'vertical', rechunk = True)
    else:
        return pl.DataFrame()

def execution_data(stock_data, group = 'day', N = 4, trader_participation_distribution = 'homogenous', delta = 2, min_num_child_orders = 5):

    if group == 'day':
        stock_data = stock_data.with_columns(pl.col('Date').alias('group key'))

    elif group == 'month':
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1mo').alias('group key'))
    
    elif group == 'year':
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1y').alias('group key'))

    groups = stock_data.select('group key').unique().sort('group key').to_series().to_list()

    if isinstance(N, list):
        if len(N) != len(groups):
            raise ValueError('Length of N list must match number of groups')
        N_map = dict(zip(groups, N))
    else:
        N_map = {g: N for g in groups}

    if isinstance(delta, list):
        if len(delta) != len(groups):
            raise ValueError('Length of delta list must match number of groups')
        delta_map = dict(zip(groups, delta))
    else:
        delta_map = {g: delta for g in groups}    
    
    impact_profile = []
    
    for trades in stock_data.partition_by('group key'):

        if trades.is_empty():
            continue

        trades = trades.sort(['DateTime', 'ExchangeSequenceNo'])

        group_value = trades['group key'][0]
        N_val       = N_map[group_value]
        delta_val   = delta_map[group_value]
        
        f      = trader_participation(N = N_val, trader_participation_distribution = trader_participation_distribution, delta = delta_val, f_min = 1,
                                            f_max = trades.shape[0], seed = 1)
        c      = cumulative_probs(f)
        output = orders(N = N_val, trades = trades, cumulative_probs = c)
        
        assignments = []
        for trader_id, rows in enumerate(output):
            for r in rows:
                assignments.append((r, trader_id))

        assignments_df = pl.DataFrame(assignments, schema = ['row idx', 'trader id'], orient = 'row')

        trades = trades.with_row_index('row idx')
        trades = trades.join(assignments_df, on = 'row idx', how = 'inner')
        trades = trades.sort(['DateTime', 'ExchangeSequenceNo'])

        trades = trades.with_columns((pl.col('Trade Sign') != pl.col('Trade Sign').shift(1)).cast(pl.Int32)
            .cum_sum().over(['Date', 'trader id']).alias('metaorder id'))
            
        grouping_cols = ['Date', 'trader id', 'metaorder id']
        features = trades.with_columns([
            
            pl.col('Ticker'),
            pl.col('MIC'),
            pl.col('DateTime').first().alias('Start time'),
            pl.col('DateTime').last().alias('End time'),
            
            pl.col('Daily Volume').first().over(grouping_cols),
            pl.col('Intraday Volatility').first().over(grouping_cols),
            pl.col('20 AD volume').first().over(grouping_cols),
            pl.col('20 AD volatility').first().over(grouping_cols),

            pl.col('Trade Sign').alias('trade sign'),

            pl.col('Volume').alias('volume'),
            pl.col('Volume').sum().over(grouping_cols).alias('volume traded'),
            (pl.col('Volume').cum_sum().over(grouping_cols) / pl.col('Volume').sum().over(grouping_cols)).alias('phi'),
            pl.len().over(grouping_cols).alias('number child orders'),
            
            pl.col('Mid-price before'),
            pl.col('Mid-price after'),

            (pl.col('Trade Sign').first().over(grouping_cols) * (pl.col('Mid-price after').last().log().over(grouping_cols) - pl.col('Mid-price before').first().log().over(grouping_cols))).alias('metaorder impact')
        ])
        
        features = features.filter(pl.col('number child orders') >= min_num_child_orders)
        
        features = features.with_columns([
            (pl.col('trade sign') * (pl.col('Mid-price after').log() - pl.col('Mid-price before').first().log().over(grouping_cols)) / (pl.col('20 AD volatility') * (pl.col('volume traded') / pl.col('20 AD volume')).sqrt())).alias('dynamical metaorder impact'),
            (((pl.col('End time') - pl.col('Start time')).dt.total_microseconds() / 6e7).alias('duration (min)'))
        ])        
        features = features.select('MIC', 'Ticker', 'Date', 'DateTime', '20 AD volume', '20 AD volatility', 'trade sign',
                                   'trader id', 'metaorder id', 'number child orders', 'volume', 'volume traded',
                                   'phi', 'metaorder impact', 'dynamical metaorder impact', 'duration (min)')
        
        impact_profile.append(features)

    if  impact_profile:
        return pl.concat(impact_profile, how = 'vertical', rechunk = True)
    else:
        return pl.DataFrame()

def decay_data(stock_data, group = 'day', N = 4, trader_participation_distribution = 'homogenous', delta = 2, min_num_child_orders = 5):

    ticker = stock_data['Ticker'][0]
    if group == 'day':
        stock_data = stock_data.with_columns(pl.col('Date').alias('group key'))

    elif group == 'month':
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1mo').alias('group key'))
    
    elif group == 'year':
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1y').alias('group key'))

    groups = stock_data.select('group key').unique().sort('group key').to_series().to_list()

    if isinstance(N, list):
        if len(N) != len(groups):
            raise ValueError('Length of N list must match number of groups')
        N_map = dict(zip(groups, N))
    else:
        N_map = {g: N for g in groups}

    if isinstance(delta, list):
        if len(delta) != len(groups):
            raise ValueError('Length of delta list must match number of groups')
        delta_map = dict(zip(groups, delta))
    else:
        delta_map = {g: delta for g in groups}    
    
    features_list_pl = []
    
    for trades in stock_data.partition_by('group key'):

        if trades.is_empty():
            continue

        trades = trades.sort(['DateTime', 'ExchangeSequenceNo'])

        group_value = trades['group key'][0]
        N_val       = N_map[group_value]
        delta_val   = delta_map[group_value]
        
        f      = trader_participation(N = N_val, trader_participation_distribution = trader_participation_distribution, delta = delta_val, f_min = 1,
                                            f_max = trades.shape[0], seed = 1)
        c      = cumulative_probs(f)
        output = orders(N = N_val, trades = trades, cumulative_probs = c)
        
        assignments = []
        for trader_id, rows in enumerate(output):
            for r in rows:
                assignments.append((r, trader_id))

        assignments_df = pl.DataFrame(assignments, schema = ['row idx', 'trader id'], orient = 'row')

        trades = trades.sort(['Date', 'DateTime', 'ExchangeSequenceNo'])
        trades = trades.with_row_index('row idx')
        trades = trades.join(assignments_df, on = 'row idx', how = 'inner')
        trades = trades.sort(['DateTime', 'ExchangeSequenceNo'])
        
        trades = trades.with_columns((pl.col('Trade Sign') != pl.col('Trade Sign').shift(1)).cast(pl.Int32)
            .cum_sum().over(['Date', 'trader id']).alias('metaorder id'))

        features = trades.group_by(['Date', 'trader id', 'metaorder id']
        ).agg([

            pl.col('Ticker').first(),
            pl.col('MIC').first(),
            pl.col('DateTime').first().alias('Start time'),
            pl.col('DateTime').last().alias('End time'),

            pl.col('Daily Volume').first(),
            pl.col('Intraday Volatility').first(),
            pl.col('20 AD volume').first(),
            pl.col('20 AD volatility').first(),

            pl.col('Trade Sign').first().alias('trade sign'),
            
            pl.col('Volume').sum().alias('volume traded'),
            pl.len().alias('number child orders'),
            
            pl.col('Mid-price before').first(),
            pl.col('Mid-price after').last(),
        ])

        features = features.filter(pl.col('number child orders') >= min_num_child_orders)

        features = features.with_columns([
            ((pl.col('Mid-price after').log() - pl.col('Mid-price before').log())
             * pl.col('trade sign')).alias('impact'),

            (((pl.col('End time') - pl.col('Start time')).dt.total_microseconds() / 6e7).alias('duration (min)'))
        ])
        
        features_list_pl.append(features)

    if features_list_pl:
        metaorder_data = pl.concat(features_list_pl, how = 'vertical', rechunk = True)

        stock_mid_prices = get_market_data_range('XJSE', start_date = '2023-01-03', end_date = '2025-12-31',
                                       table_name = 'l1', ticker = ticker, df_engine = 'polars',
                                      columns = ['Ticker', 'TradeDate', 'LocalTimestamp', 'BidPrice1', 'BidQuantity1', 'AskPrice1', 'AskQuantity1',
                                                 'ExchangeSequenceNo', 'MarketState'])
        stock_mid_prices = stock_mid_prices.filter(pl.col('MarketState') == 'CONTINUOUS_TRADING')
        stock_mid_prices = stock_mid_prices.sort(['LocalTimestamp', 'ExchangeSequenceNo'])
        stock_mid_prices = stock_mid_prices.with_columns(((pl.col('BidPrice1') + pl.col('AskPrice1')) / 2).alias('Mid-price'))
        stock_mid_prices = stock_mid_prices.with_columns(pl.col('LocalTimestamp').dt.replace_time_zone(None))
        stock_mid_prices = stock_mid_prices.sort('LocalTimestamp')
        
        z_max  = 2
        z_grid = np.linspace(1, z_max, 100)
        
        meta = metaorder_data.filter(pl.col('number child orders') >= min_num_child_orders)
        meta = meta.sort('Start time')
        
        expanded = meta.with_row_index('meta id').join(pl.DataFrame({'z': z_grid}), how = 'cross')
        expanded = expanded.with_columns((pl.col('Start time') + pl.duration(microseconds = (pl.col('z') * (pl.col('End time') - pl.col('Start time')).dt.total_microseconds()).cast(pl.Int64))).alias('query time'))
        expanded = expanded.sort('query time')
        
        merged = expanded.join_asof(stock_mid_prices, left_on = 'query time', right_on = 'LocalTimestamp', strategy = 'backward')
        merged = merged.filter(pl.col('LocalTimestamp') >= pl.col('End time'))
        merged = merged.with_columns([pl.col('Mid-price before').alias('P0'), pl.col('Mid-price').alias('Pz')])
        merged = merged.with_columns((pl.col('trade sign') * (pl.col('Pz').log() - pl.col('P0').log())).alias('metaorder impact'))
        merged = merged.with_columns((pl.col('metaorder impact') / (pl.col('20 AD volatility') * (pl.col('volume traded') / pl.col('20 AD volume')).sqrt())).alias('dynamical metaorder impact'))
        
        decay_df = merged.select(['Date', 'z', 'metaorder impact', 'dynamical metaorder impact'])

        return decay_df
        
    else:
        return pl.DataFrame()
        
def metaorder_stylised_facts(stock_data, N, delta = 2, trader_participation_distribution = 'power', min_num_child_orders = 5):

    if trader_participation_distribution == 'power':

        years_col         = []
        ticker_col        = []
        xmin_col          = []
        alpha_col         = []
        gamma_nlls_c0_col = []
        gamma_nlls_col    = []
        N_nlls_col        = []
        gamma_psd_c0_col  = []
        gamma_psd_col     = []
        N_psd_col         = []
        N_st              = []
        sql_est_col       = []
        sql_var_col       = []
        sql_interval_half_width_col  = []
        phi_est_col                  = []
        phi_var_col                  = []
        phi_interval_half_width_col  = []
        beta_est_col                 = []
        beta_var_col                 = []
        beta_interval_half_width_col = []
    
        execution_data_all   = execution_data(stock_data, group = 'year', N = N, trader_participation_distribution = trader_participation_distribution, delta = delta, min_num_child_orders = 1)
        execution_data_all   = execution_data_all.with_columns(pl.col('Date').dt.truncate('1y').dt.year().alias('Year'))
        
        decay_data_all     = decay_data(stock_data, group = 'year', N = N, trader_participation_distribution = trader_participation_distribution, delta = delta, min_num_child_orders = min_num_child_orders)
        decay_data_all     = decay_data_all.with_columns(pl.col('Date').dt.truncate('1y').dt.year().alias('Year'))
        
        metaorder_data_all = execution_data_all.filter((pl.col('phi') == 1) & (pl.col('number child orders') > 1))
        
        execution_data_all   = execution_data_all.filter(pl.col('number child orders') >= min_num_child_orders)
    
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1y').alias('Year'))
        
        for df_year in stock_data.partition_by('Year'):
        
            year   = df_year['Year'].dt.year()[0]
            ticker = df_year['Ticker'][0]
            
            if df_year.shape[0] < 1000:
                continue
    
            execution_data_year   = execution_data_all.filter(pl.col('Year') == year)
            decay_data_year       = decay_data_all.filter(pl.col('Year') == year)
            metaorder_data_year   = metaorder_data_all.filter(pl.col('Year') == year)
        
            if execution_data_year.is_empty() or decay_data_year.is_empty() or metaorder_data_year.is_empty():
                return pl.DataFrame()
    
            ### SQL ###
            n_intervals = 40
            
            df = metaorder_data_year.with_columns((pl.col('volume traded') / pl.col('20 AD volume')).alias('x_meta'),
                                             (pl.col('metaorder impact') / pl.col('20 AD volatility')).alias('y_meta'))
            
            x_min = df.select(pl.col('x_meta').min()).item()
            x_max = df.select(pl.col('x_meta').max()).item()
            
            bins = np.logspace(np.log10(x_min), np.log10(x_max), n_intervals)
            df   = df.with_columns([pl.Series('bin', np.digitize(df['x_meta'].to_numpy(), bins) - 1)])
            
            all_bins = pl.DataFrame({'bin': np.arange(len(bins) - 1)})
            binned   = df.group_by('bin').agg(pl.col('y_meta').mean().alias('y_mean')).sort('bin')
            binned   = all_bins.join(binned, on = 'bin', how = 'left').sort('bin')
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            binned      = binned.with_columns(x_center = bin_centers)
            
            alpha  = 0.5
            binned = binned.with_columns((alpha * pl.col('x_center').sqrt()).alias('sqrt_law'))
    
            def sql_model(x, exp, Y):
                return Y * x ** exp
                
            y = np.array(binned['y_mean'])
            x = np.array(binned['x_center'])
            
            mask       = (x > 0) & np.isfinite(y)
            x_fit      = x[mask]
            impact_fit = y[mask]
            
            if len(x_fit) < 10 or len(impact_fit) < 10:
                return pl.DataFrame()
                        
            try:
                params, covariance = curve_fit(sql_model, x_fit, impact_fit, p0 = [0.5, 0.5], maxfev = 100000)
        
            except RuntimeError:
                return pl.DataFrame()
    
            exp_est = params[0]
            
            n   = len(impact_fit)
            p   = len(params)          # number of parameters (2 here)
            dof = n - p                # degrees of freedom
            
            alpha = 0.05               # 95% CI
            tval  = t.ppf(1 - alpha / 2, dof)
            se    = np.sqrt(np.diag(covariance))
            exp_interval_half_width = tval * se[0]
            exp_var = covariance[0, 0]
            
            ###
            
            ### Execution profile ###
            
            bins = np.linspace(0, 1, 51)
            bin_mids = (bins[:-1] + bins[1:]) / 2
            
            df        = execution_data_year.with_columns(pl.col('phi').cut(bins).alias('bin'))
            binned    = df.group_by('bin').agg(pl.col('dynamical metaorder impact').mean().alias('dynamical metaorder impact (binned)'))
            full_bins = pl.DataFrame({'phi': bin_mids}).with_columns(pl.col('phi').cut(bins).alias('bin'))
            df        = full_bins.join(binned, on = 'bin', how = 'left').sort('phi')
            
            scaled_impact_binned = df['dynamical metaorder impact (binned)']
            if len(scaled_impact_binned) < 10:
                return pl.DataFrame()
                
            scaled_impact_binned = pl.concat([pl.Series([0.0]), scaled_impact_binned])
            phi                  = np.insert(bin_mids, 0, 0)
            
            def execution_profile_model(phi, exponent, Y):
                return  Y * (phi ** exponent)
            
            phi    = np.array(phi)
            impact = np.array(scaled_impact_binned)
            
            mask       = (phi > 0) & np.isfinite(impact)
            phi_fit    = phi[mask]
            impact_fit = impact[mask]
        
            if len(phi_fit) < 10 or len(impact_fit) < 10:
                return pl.DataFrame()
            
            try:
                params, covariance = curve_fit(execution_profile_model, phi_fit, impact_fit, p0 = [0.5, 1], maxfev = 100000)
        
            except RuntimeError:
                return pl.DataFrame()
                
            phi_est = params[0]
            
            n   = len(impact_fit)
            p   = len(params)          # number of parameters (2 here)
            dof = n - p                # degrees of freedom
            
            alpha = 0.05               # 95% CI
            tval  = t.ppf(1 - alpha / 2, dof)
            se    = np.sqrt(np.diag(covariance))
            phi_interval_half_width = tval * se[0]
            phi_var = covariance[0, 0]
            
            ###
            
            ### Decay profile ###
            
            bins = np.linspace(1, 2, 51)
            bin_mids = (bins[:-1] + bins[1:]) / 2
            
            df        = decay_data_year.with_columns(pl.col('z').cut(bins).alias('bin'))
            binned    = df.group_by('bin').agg(pl.col('dynamical metaorder impact').mean().alias('dynamical metaorder impact (binned)')).sort('bin')
            full_bins = pl.DataFrame({'z': bin_mids}).with_columns(pl.col('z').cut(bins).alias('bin'))
            df        = full_bins.join(binned, on = 'bin', how = 'left').sort('z')
           
            z         = np.array(bin_mids)
            impact    = np.array(df['dynamical metaorder impact (binned)'])
            
            def decay_model(z, beta, Y):
                return  Y * (z ** (1 - beta) - (z - 1) ** (1 - beta))
            
            mask       = (z > 1) & np.isfinite(impact)
            z_fit      = z[mask]
            impact_fit = impact[mask]
        
            if len(z_fit) < 10 or len(impact_fit) < 10:
                return pl.DataFrame()
        
            try:
                params, covariance = curve_fit(decay_model, z_fit, impact_fit, p0 = [0.2, 1], maxfev = 100000)
        
            except RuntimeError:
                return pl.DataFrame()
            
            beta_est = params[0]
        
            n   = len(z_fit)
            p   = len(params)          # number of parameters (2 here)
            dof = n - p                # degrees of freedom
            
            alpha = 0.05               # 95% CI
            tval  = t.ppf(1 - alpha / 2, dof)
            se    = np.sqrt(np.diag(covariance))
            beta_interval_half_width = tval * se[0]
            beta_var = covariance[0, 0]
            
            L, n_st     = LMF.extract_ST_run_lengths(df_year, group = 'year', N = N, trader_participation_distribution = trader_participation_distribution, delta = delta)
            signs = df_year['Trade Sign']
            
            if len(L) == 0:
                continue
        
            L     = pl.DataFrame({'L' : L})
            signs = pl.DataFrame({'Trade Sign' : signs})
            
            taus, C = LMF.acf_fft(signs['Trade Sign'], max_lag = 100000)
            out     = LMF.estimate_gamma(taus, C, min_points = 100, nbins = 100, trade_signs = signs['Trade Sign'])

            intercept_nlls = np.exp(out['intercept (nlls)'])
            gamma_nlls     = out['gamma (nlls)']
            N_nlls         = 1 / (((1 + gamma_nlls) * intercept_nlls) ** (1 / (1 - gamma_nlls)))
            
            gamma_psd      = out['gamma (psd)']
            conversion     = 2 * (2 * np.pi) ** (gamma_psd - 1) * gamma_func(1 - gamma_psd) * np.sin(np.pi * gamma_psd / 2)
            A_psd          = np.exp(out['intercept (psd)'])
            intercept_psd  = A_psd / conversion
            N_psd          = 1 / (((1 + gamma_psd) * intercept_psd) ** (1 / (1 - gamma_psd)))
            
            years_col.append(year)
            ticker_col.append(ticker)
            N_st.append(n_st)
            
            gamma_nlls_c0_col.append(intercept_nlls)
            gamma_nlls_col.append(gamma_nlls)
            N_nlls_col.append(N_nlls)
            
            gamma_psd_c0_col.append(intercept_psd)
            gamma_psd_col.append(gamma_psd)
            N_psd_col.append(N_psd)
            
            alpha, xmin = LMF.estimate_alpha(L['L'])

            xmin_col.append(xmin)
            alpha_col.append(alpha)
            sql_est_col.append(exp_est)
            sql_var_col.append(exp_var)
            sql_interval_half_width_col.append(exp_interval_half_width)
            phi_est_col.append(phi_est)
            phi_var_col.append(phi_var)
            phi_interval_half_width_col.append(phi_interval_half_width)
            beta_est_col.append(beta_est)
            beta_var_col.append(beta_var)
            beta_interval_half_width_col.append(beta_interval_half_width)
    
        data_lists = [years_col, ticker_col, xmin_col, alpha_col, gamma_nlls_c0_col, gamma_nlls_col, N_nlls_col,
                      gamma_psd_c0_col, gamma_psd_col, N_psd_col, N_st, phi_est_col, phi_interval_half_width_col,
                      beta_est_col, beta_interval_half_width_col]
        
        if any(not x for x in data_lists):
            return pl.DataFrame()
            
        alpha_gamma_df = pl.DataFrame({
                'year'                    : years_col,
                'ticker'                  : ticker_col,
                'N'                       : N,
                'N (st)'                  : N_st,
                'delta'                   : delta,
                'xmin'                    : xmin_col,
                'alpha'                   : alpha_col,
                'intercept (nlls)'        : gamma_nlls_c0_col,
                'gamma (nlls)'            : gamma_nlls_col,
                'N (nlls)'                : N_nlls_col,
                'intercept (psd)'         : gamma_psd_c0_col,
                'gamma (psd)'             : gamma_psd_col,
                'N (psd)'                 : N_psd_col,
                'impact curve exponent'   : sql_est_col,
                'impact curve exponent var' : sql_var_col,
                'impact curve half width' : sql_interval_half_width_col,
                'phi exponent'            : phi_est_col,
                'phi exponent var'        : phi_var_col,
                'phi exponent half width' : phi_interval_half_width_col,
                'beta'                    : beta_est_col,
                'beta var'                : beta_var_col,
                'beta half width'         : beta_interval_half_width_col},
                 schema = {
                'year'                    : pl.Int64,
                'ticker'                  : pl.Utf8,
                'N'                       : pl.Int64,
                'N (st)'                  : pl.Int64,
                'delta'                   : pl.Float64,
                'xmin'                    : pl.Float64,
                'alpha'                   : pl.Float64,
                'intercept (nlls)'        : pl.Float64,
                'gamma (nlls)'            : pl.Float64,
                'N (nlls)'                : pl.Float64,
                'intercept (psd)'         : pl.Float64,
                'gamma (psd)'             : pl.Float64,
                'N (psd)'                 : pl.Float64,
                'impact curve exponent'   : pl.Float64,
                'impact curve exponent var' : pl.Float64,
                'impact curve half width' : pl.Float64,
                'phi exponent'            : pl.Float64,
                'phi exponent var'        : pl.Float64,
                'phi exponent half width' : pl.Float64,
                'beta'                    : pl.Float64,
                'beta var'                : pl.Float64,
                'beta half width'         : pl.Float64})
        
        return alpha_gamma_df

    if trader_participation_distribution == 'homogenous':
        years_col         = []
        ticker_col        = []
        xmin_col          = []
        alpha_col         = []
        gamma_nlls_c0_col = []
        gamma_nlls_col    = []
        N_nlls_col        = []
        gamma_psd_c0_col  = []
        gamma_psd_col     = []
        N_psd_col         = []
        N_st              = []
        sql_est_col       = []
        sql_var_col       = []
        sql_interval_half_width_col  = []
        phi_est_col                  = []
        phi_var_col                  = []
        phi_interval_half_width_col  = []
        beta_est_col                 = []
        beta_var_col                 = []
        beta_interval_half_width_col = []
    
        execution_data_all   = execution_data(stock_data, group = 'year', N = N, trader_participation_distribution = trader_participation_distribution, delta = delta, min_num_child_orders = 1)
        execution_data_all   = execution_data_all.with_columns(pl.col('Date').dt.truncate('1y').dt.year().alias('Year'))
        
        decay_data_all     = decay_data(stock_data, group = 'year', N = N, trader_participation_distribution = trader_participation_distribution, delta = delta, min_num_child_orders = min_num_child_orders)
        decay_data_all     = decay_data_all.with_columns(pl.col('Date').dt.truncate('1y').dt.year().alias('Year'))
        
        metaorder_data_all = execution_data_all.filter((pl.col('phi') == 1) & (pl.col('number child orders') > 1))
        
        execution_data_all   = execution_data_all.filter(pl.col('number child orders') >= min_num_child_orders)
    
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1y').alias('Year'))
        
        for df_year in stock_data.partition_by('Year'):
        
            year   = df_year['Year'].dt.year()[0]
            ticker = df_year['Ticker'][0]
            
            if df_year.shape[0] < 1000:
                continue
    
            execution_data_year   = execution_data_all.filter(pl.col('Year') == year)
            decay_data_year       = decay_data_all.filter(pl.col('Year') == year)
            metaorder_data_year   = metaorder_data_all.filter(pl.col('Year') == year)
        
            if execution_data_year.is_empty() or decay_data_year.is_empty() or metaorder_data_year.is_empty():
                return pl.DataFrame()
    
            ### SQL ###
            n_intervals = 40
            
            df = metaorder_data_year.with_columns((pl.col('volume traded') / pl.col('20 AD volume')).alias('x_meta'),
                                             (pl.col('metaorder impact') / pl.col('20 AD volatility')).alias('y_meta'))
            
            x_min = df.select(pl.col('x_meta').min()).item()
            x_max = df.select(pl.col('x_meta').max()).item()
            
            bins = np.logspace(np.log10(x_min), np.log10(x_max), n_intervals)
            df   = df.with_columns([pl.Series('bin', np.digitize(df['x_meta'].to_numpy(), bins) - 1)])
            
            all_bins = pl.DataFrame({'bin': np.arange(len(bins) - 1)})
            binned   = df.group_by('bin').agg(pl.col('y_meta').mean().alias('y_mean')).sort('bin')
            binned   = all_bins.join(binned, on = 'bin', how = 'left').sort('bin')
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            binned      = binned.with_columns(x_center = bin_centers)
            
            alpha  = 0.5
            binned = binned.with_columns((alpha * pl.col('x_center').sqrt()).alias('sqrt_law'))
    
            def sql_model(x, exp, Y):
                return Y * x ** exp
                
            y = np.array(binned['y_mean'])
            x = np.array(binned['x_center'])
            
            mask       = (x > 0) & np.isfinite(y)
            x_fit      = x[mask]
            impact_fit = y[mask]
            
            if len(x_fit) < 10 or len(impact_fit) < 10:
                return pl.DataFrame()
                        
            try:
                params, covariance = curve_fit(sql_model, x_fit, impact_fit, p0 = [0.5, 0.5], maxfev = 100000)
        
            except RuntimeError:
                return pl.DataFrame()
    
            exp_est = params[0]
            
            n   = len(impact_fit)
            p   = len(params)          # number of parameters (2 here)
            dof = n - p                # degrees of freedom
            
            alpha = 0.05               # 95% CI
            tval  = t.ppf(1 - alpha / 2, dof)
            se    = np.sqrt(np.diag(covariance))
            exp_interval_half_width = tval * se[0]
            exp_var = covariance[0, 0]
            
            ###
            
            ### Execution profile ###
            
            bins = np.linspace(0, 1, 51)
            bin_mids = (bins[:-1] + bins[1:]) / 2
            
            df        = execution_data_year.with_columns(pl.col('phi').cut(bins).alias('bin'))
            binned    = df.group_by('bin').agg(pl.col('dynamical metaorder impact').mean().alias('dynamical metaorder impact (binned)'))
            full_bins = pl.DataFrame({'phi': bin_mids}).with_columns(pl.col('phi').cut(bins).alias('bin'))
            df        = full_bins.join(binned, on = 'bin', how = 'left').sort('phi')
            
            scaled_impact_binned = df['dynamical metaorder impact (binned)']
            if len(scaled_impact_binned) < 10:
                return pl.DataFrame()
                
            scaled_impact_binned = pl.concat([pl.Series([0.0]), scaled_impact_binned])
            phi                  = np.insert(bin_mids, 0, 0)
            
            def execution_profile_model(phi, exponent, Y):
                return  Y * (phi ** exponent)
            
            phi    = np.array(phi)
            impact = np.array(scaled_impact_binned)
            
            mask       = (phi > 0) & np.isfinite(impact)
            phi_fit    = phi[mask]
            impact_fit = impact[mask]
        
            if len(phi_fit) < 10 or len(impact_fit) < 10:
                return pl.DataFrame()
            
            try:
                params, covariance = curve_fit(execution_profile_model, phi_fit, impact_fit, p0 = [0.5, 1], maxfev = 100000)
        
            except RuntimeError:
                return pl.DataFrame()
                
            phi_est = params[0]
            
            n   = len(impact_fit)
            p   = len(params)          # number of parameters (2 here)
            dof = n - p                # degrees of freedom
            
            alpha = 0.05               # 95% CI
            tval  = t.ppf(1 - alpha / 2, dof)
            se    = np.sqrt(np.diag(covariance))
            phi_interval_half_width = tval * se[0]
            phi_var = covariance[0, 0]
            
            ###
            
            ### Decay profile ###
            
            bins = np.linspace(1, 2, 51)
            bin_mids = (bins[:-1] + bins[1:]) / 2
            
            df        = decay_data_year.with_columns(pl.col('z').cut(bins).alias('bin'))
            binned    = df.group_by('bin').agg(pl.col('dynamical metaorder impact').mean().alias('dynamical metaorder impact (binned)')).sort('bin')
            full_bins = pl.DataFrame({'z': bin_mids}).with_columns(pl.col('z').cut(bins).alias('bin'))
            df        = full_bins.join(binned, on = 'bin', how = 'left').sort('z')
           
            z         = np.array(bin_mids)
            impact    = np.array(df['dynamical metaorder impact (binned)'])
            
            def decay_model(z, beta, Y):
                return  Y * (z ** (1 - beta) - (z - 1) ** (1 - beta))
            
            mask       = (z > 1) & np.isfinite(impact)
            z_fit      = z[mask]
            impact_fit = impact[mask]
        
            if len(z_fit) < 10 or len(impact_fit) < 10:
                return pl.DataFrame()
        
            try:
                params, covariance = curve_fit(decay_model, z_fit, impact_fit, p0 = [0.2, 1], maxfev = 100000)
        
            except RuntimeError:
                return pl.DataFrame()
            
            beta_est = params[0]
        
            n   = len(z_fit)
            p   = len(params)          # number of parameters (2 here)
            dof = n - p                # degrees of freedom
            
            alpha = 0.05               # 95% CI
            tval  = t.ppf(1 - alpha / 2, dof)
            se    = np.sqrt(np.diag(covariance))
            beta_interval_half_width = tval * se[0]
            beta_var = covariance[0, 0]
            
            L, n_st     = LMF.extract_ST_run_lengths(df_year, group = 'year', N = N, trader_participation_distribution = trader_participation_distribution, delta = delta)
            signs = df_year['Trade Sign']
            
            if len(L) == 0:
                continue
        
            L     = pl.DataFrame({'L' : L})
            signs = pl.DataFrame({'Trade Sign' : signs})
            
            taus, C = LMF.acf_fft(signs['Trade Sign'], max_lag = 100000)
            out     = LMF.estimate_gamma(taus, C, min_points = 100, nbins = 100, trade_signs = signs['Trade Sign'])

            intercept_nlls = np.exp(out['intercept (nlls)'])
            gamma_nlls     = out['gamma (nlls)']
            N_nlls         = 1 / (((1 + gamma_nlls) * intercept_nlls) ** (1 / (1 - gamma_nlls)))
            
            gamma_psd      = out['gamma (psd)']
            conversion     = 2 * (2 * np.pi) ** (gamma_psd - 1) * gamma_func(1 - gamma_psd) * np.sin(np.pi * gamma_psd / 2)
            A_psd          = np.exp(out['intercept (psd)'])
            intercept_psd  = A_psd / conversion
            N_psd          = 1 / (((1 + gamma_psd) * intercept_psd) ** (1 / (1 - gamma_psd)))
            
            years_col.append(year)
            ticker_col.append(ticker)
            N_st.append(n_st)
            
            gamma_nlls_c0_col.append(intercept_nlls)
            gamma_nlls_col.append(gamma_nlls)
            N_nlls_col.append(N_nlls)
            
            gamma_psd_c0_col.append(intercept_psd)
            gamma_psd_col.append(gamma_psd)
            N_psd_col.append(N_psd)
        
            alpha, xmin = LMF.estimate_alpha(L['L'])

            xmin_col.append(xmin)
            alpha_col.append(alpha)
            sql_est_col.append(exp_est)
            sql_var_col.append(exp_var)
            sql_interval_half_width_col.append(exp_interval_half_width)
            phi_est_col.append(phi_est)
            phi_var_col.append(phi_var)
            phi_interval_half_width_col.append(phi_interval_half_width)
            beta_est_col.append(beta_est)
            beta_var_col.append(beta_var)
            beta_interval_half_width_col.append(beta_interval_half_width)
    
        data_lists = [years_col, ticker_col, xmin_col, alpha_col, gamma_nlls_c0_col, gamma_nlls_col, N_nlls_col,
                      gamma_psd_c0_col, gamma_psd_col, N_psd_col, N_st, phi_est_col, phi_interval_half_width_col,
                      beta_est_col, beta_interval_half_width_col]
        
        if any(not x for x in data_lists):
            return pl.DataFrame()
            
        alpha_gamma_df = pl.DataFrame({
                'year'                    : years_col,
                'ticker'                  : ticker_col,
                'N'                       : N,
                'N (st)'                  : N_st,
                'xmin'                    : xmin_col,
                'alpha'                   : alpha_col,
                'intercept (nlls)'        : gamma_nlls_c0_col,
                'gamma (nlls)'            : gamma_nlls_col,
                'N (nlls)'                : N_nlls_col,
                'intercept (psd)'         : gamma_psd_c0_col,
                'gamma (psd)'             : gamma_psd_col,
                'N (psd)'                 : N_psd_col,
                'impact curve exponent'   : sql_est_col,
                'impact curve exponent var' : sql_var_col,
                'impact curve half width' : sql_interval_half_width_col,
                'phi exponent'            : phi_est_col,
                'phi exponent var'        : phi_var_col,
                'phi exponent half width' : phi_interval_half_width_col,
                'beta'                    : beta_est_col,
                'beta var'                : beta_var_col,
                'beta half width'         : beta_interval_half_width_col},
                 schema = {
                'year'                    : pl.Int64,
                'ticker'                  : pl.Utf8,
                'N'                       : pl.Int64,
                'N (st)'                  : pl.Int64,
                'xmin'                    : pl.Float64,
                'alpha'                   : pl.Float64,
                'intercept (nlls)'        : pl.Float64,
                'gamma (nlls)'            : pl.Float64,
                'N (nlls)'                : pl.Float64,
                'intercept (psd)'         : pl.Float64,
                'gamma (psd)'             : pl.Float64,
                'N (psd)'                 : pl.Float64,
                'impact curve exponent'   : pl.Float64,
                'impact curve exponent var' : pl.Float64,
                'impact curve half width' : pl.Float64,
                'phi exponent'            : pl.Float64,
                'phi exponent var'        : pl.Float64,
                'phi exponent half width' : pl.Float64,
                'beta'                    : pl.Float64,
                'beta var'                : pl.Float64,
                'beta half width'         : pl.Float64})
        
        return alpha_gamma_df
        