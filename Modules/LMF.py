#pip install 'powerlaw'

import bmll2 as b2
from bmll2 import reference, Security, NormalisedSecurity, SparkHelper, get_market_data, get_market_data_range, VenueMarketError, get_market_tables, save_spark_dataframe, load_spark_dataframe

b2.get_file('modules/metaorder.py')

import metaorder as metaorder

import random
import math
import pandas as pd
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

def one_sided_runs_test(sample: list, runs_correction = 0):

    if len(sample) < 2:
        return None

    if sum(sample) < 100:
        return None

    N   = sum(sample)
    N_a = sum(sample[::2])
    N_b = sum(sample[1::2])

    mu   = ((2 * N_a * N_b) / N) + 1
    runs = len(sample)

    R_corrected = runs + runs_correction
    
    sigma = np.sqrt((2 * N_a * N_b * (2 * N_a * N_b - N)) / (N ** 2 * (N - 1)))

    z = (R_corrected - mu) / sigma
    p_value = scipy.stats.norm.cdf(z)

    return (z, p_value, runs)

def extract_ST_run_lengths(stock_data, group = 'year', N = 100, trader_participation_distribution = 'power',
                           delta = 2, p_threshold = 0.01):

    stock_data = stock_data.sort(['DateTime', 'ExchangeSequenceNo'], descending = False)

    if group == 'day':
        stock_data = stock_data.with_columns(pl.col('Date').alias('group key'))

    elif group == 'month':
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1mo').alias('group key'))
    
    elif group == 'year':
        stock_data = stock_data.with_columns(pl.col('Date').dt.truncate('1y').alias('group key'))

    run_lengths = []
    st_count    = []
    
    for trades in stock_data.partition_by('group key'):

        if trades.is_empty():
            continue

        group_key = trades['group key'][0]
            
        trades = trades.sort(['Date', 'DateTime', 'ExchangeSequenceNo'])
        f = metaorder.trader_participation(N = N, trader_participation_distribution = trader_participation_distribution, delta =  delta, f_min = 1, 
                                       f_max = trades.shape[0], seed = 1)
        c = metaorder.cumulative_probs(f)
        output = metaorder.orders(N = N, trades = trades, cumulative_probs = c)
    
        assignments = []
        for trader_id, rows in enumerate(output):
            for r in rows:
                assignments.append((r, trader_id))
                
        assignments_df = pl.DataFrame(assignments, schema = ['row idx', 'trader id'], orient = 'row')
    
        trades = trades.with_row_index('row idx')
        trades = trades.join(assignments_df, on = 'row idx', how = 'inner')
    
        trades = trades.with_columns((pl.col('Trade Sign') != pl.col('Trade Sign').shift(1)).cast(pl.Int32)
            .cum_sum().over(['Date', 'trader id']).alias('metaorder id'))
    
        grouping_cols = ['trader id', 'metaorder id']
    
        features = trades.with_columns([pl.col('DateTime').diff().over('trader id').alias('gap'), 
                                        (pl.col('Trade Sign') == pl.col('Trade Sign').shift(1)).over('trader id').fill_null(False).alias('same sign')])
        features = features.with_columns(((pl.col('gap') > timedelta(days = 1)) & pl.col('same sign')).fill_null(False).cast(pl.Int64).alias('day break'))
        features = features.with_columns((pl.col('Trade Sign') != pl.col('Trade Sign').shift(1)).over('trader id').fill_null(True).cast(pl.Int64).cum_sum().alias('run id'))
        run_lengths_df = features.group_by(['trader id', 'run id']).agg(pl.col('run id').len().alias('L'), pl.col('day break').max().alias('day break'))
        run_lengths_df = run_lengths_df.with_columns(pl.col('day break').sum().over('trader id').alias('day breaks'))
        run_lengths_df = run_lengths_df.select(['trader id', 'L', 'day breaks'])
        runs_per_trader = run_lengths_df.group_by('trader id').agg([pl.col('L'),pl.col('day breaks').first()])
        runs_per_trader = runs_per_trader.sort('trader id')

        n_st = 0
        
        for row in runs_per_trader.iter_rows(named = True):
        
            runs = row['L']
            day_breaks = row['day breaks']
        
            if len(runs) < 2:
                continue

            if sum(runs) < 100:
                continue

            z, p_val, R = one_sided_runs_test(runs, runs_correction = day_breaks)
            
            if p_val <= p_threshold:
                n_st += 1
                run_lengths.extend(runs)

    return run_lengths, n_st

def acf_fft(signs: pl.Series, max_lag):

    signs   = np.asarray(signs)
    N       = len(signs)
    max_lag = min(max_lag, N - 1)

    nfft = 1 << (2 * N - 1).bit_length()

    f   = np.fft.fft(signs, nfft)
    acf = np.fft.ifft(f * np.conjugate(f)).real
    acf = acf[:max_lag + 1]

    norm = np.arange(N, N - max_lag - 1, -1)

    C    = acf[1:] / norm[1:]
    taus = np.arange(1, max_lag + 1)

    return taus, C

def select_powerlaw_range(taus, C, min_points = 10000, confidence_percent = 0.05):

    log_tau = np.log(taus)
    log_C   = np.log(C)

    N = len(log_tau)

    # cumulative sums
    # x is the log taus
    # y is the log C
    Sx  = np.cumsum(log_tau)
    Sy  = np.cumsum(log_C)
    Sxx = np.cumsum(log_tau ** 2)
    Syy = np.cumsum(log_C ** 2)
    Sxy = np.cumsum(log_tau * log_C)

    best_r2 = -np.inf
    best_i  = 0
    best_slope = 0

    j = N - 1  # fixed right endpoint (tail fit)

    for i in range(N - min_points):
        
        n = j - i

        # gonna be working backwards here. We already precomputed the cumulative sum of the entire series we just need to subtract
        # the part thats no longer in it. It starts off on the largest window and makes it smaller each time. With this method we fix
        # the left endpoint of the window so that we only need to do a O(N) search and not a O(N^2) search. 
        sum_x  = Sx[j - 1]  - (Sx[i - 1]  if i > 0 else 0)
        sum_y  = Sy[j - 1]  - (Sy[i - 1]  if i > 0 else 0)
        sum_xx = Sxx[j - 1] - (Sxx[i - 1] if i > 0 else 0)
        sum_yy = Syy[j - 1] - (Syy[i - 1] if i > 0 else 0)
        sum_xy = Sxy[j - 1] - (Sxy[i - 1] if i > 0 else 0)

        denom = n * sum_xx - sum_x ** 2
        if denom == 0:
            continue

        slope = ((n * sum_xy) - (sum_x * sum_y)) / denom
        intercept = (sum_y - (slope * sum_x)) / n

        ss_tot = sum_yy - (sum_y ** 2) / n
        ss_res = sum_yy + (slope ** 2) * sum_xx + (n * intercept ** 2) - (2 * slope * sum_xy) - (2 * intercept * sum_y) + (2 * slope * intercept * sum_x)

        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf

        if r2 > best_r2:
            best_r2 = r2
            best_i  = i
            best_slope = slope

    return best_i, N - 1, best_slope

def powerlaw_fit(tau, C_0, gamma):
    
    return C_0 * (tau ** (- gamma))

def gamma_NLLS(taus, C):

    x = np.log(taus)
    y = np.log(C)

    slope, intercept = np.polyfit(x, y, 1)
    gamma            = -1 * slope

    return intercept, gamma

def gamma_PSD(C):

    max_lag = len(C)
    PSD     = np.abs(np.fft.fft(C))
    freqs   = np.fft.fftfreq(max_lag)

    mask  = freqs > 0
    PSD   = PSD[mask]
    freqs = freqs[mask]

    cut       = int(0.15 * len(freqs))
    log_PSD   = np.log(PSD[: cut])
    log_freqs = np.log(freqs[: cut])
    
    slope, intercept  = np.polyfit(log_freqs, log_PSD, 1)
    gamma_psd = slope + 1
    
    return np.real(intercept), np.real(gamma_psd)

def estimate_gamma(taus, C, min_points, nbins, trade_signs, confidence_percent = 0.05):

    z_score = scipy.stats.norm.ppf(1 - confidence_percent)
    
    mask              = (C > (z_score / np.sqrt(len(C)))) & np.isfinite(C)
    cut = np.argmax(~mask)
    if cut == 0 and mask[0]:
        # mask is all True → keep everything
        taus, C_truncated = taus, C
    else:
        taus, C_truncated = taus[:cut], C[:cut]
    #taus, C_truncated = taus[:np.argmax(~mask)], C[:np.argmax(~mask)] # the tilde flips it so that where C < ... it says true. 
    # Then argmax returns the largest value where its true.
    # The mask still has the C values in it so it is checking the C values and not just the true false stuff
    #mask = (C > 0) & (np.isfinite(C))
    #taus = taus[mask]
    #C    = C[mask]
    
    i, j, slope     = select_powerlaw_range(taus, C_truncated, min_points)
    gamma_nlls_init = -1 * slope

    tau_min = taus[i]
    tau_max = taus[j]

    mask_fit    = (taus >= tau_min) & (taus <= tau_max)
    taus        = taus[mask_fit]
    C_truncated = C_truncated[mask_fit]

    C_0_init           = C[0] if len(C) > 0 else 1.0
    C_nlls, gamma_nlls = gamma_NLLS(taus, C_truncated)
    
    C_psd, gamma_psd   = gamma_PSD(C)

    return {'intercept (nlls)' : C_nlls, 'gamma (nlls)' : gamma_nlls, 'intercept (psd)' : C_psd, 'gamma (psd)': gamma_psd}

def estimate_alpha(run_lengths):
    
    fit = powerlaw.Fit(run_lengths, discrete = True, verbose = False)
    
    # The powerlaw package gives alpha for CCDF P_>(L) ~ L^(-alpha)
    # For PDF P(L) ~ L^(-alpha-1), we have the same alpha
    alpha = fit.alpha - 1  # Adjust for PDF vs CCDF convention
    xmin  = fit.xmin
    
    return alpha, xmin