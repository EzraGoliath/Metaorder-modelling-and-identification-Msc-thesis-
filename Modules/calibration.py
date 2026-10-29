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

from typing import Literal

def select_fitting_range(volume, impact, min_points = 10):

    volume = np.asarray(volume)
    impact = np.asarray(impact)

    mask = (volume > 0) & (impact > 0)

    volume = volume[mask]
    impact = impact[mask]
    
    log_volume = np.log(volume)
    log_impact = np.log(impact)
    
    N = len(log_volume)

    Sx  = np.cumsum(log_volume)
    Sy  = np.cumsum(log_impact)
    Sxx = np.cumsum(log_volume ** 2)
    Syy = np.cumsum(log_impact ** 2)
    Sxy = np.cumsum(log_volume * log_impact)

    best_score = -np.inf

    best_i = 0
    best_j = N - 1
    best_slope = np.nan
    best_r2 = np.nan

    for i in range(N):

        for j in range(i + min_points - 1, N):

            n = j - i + 1

            sum_x  = Sx[j]  - (Sx[i-1]  if i > 0 else 0)
            sum_y  = Sy[j]  - (Sy[i-1]  if i > 0 else 0)
            sum_xx = Sxx[j] - (Sxx[i-1] if i > 0 else 0)
            sum_yy = Syy[j] - (Syy[i-1] if i > 0 else 0)
            sum_xy = Sxy[j] - (Sxy[i-1] if i > 0 else 0)

            denom = n * sum_xx - sum_x ** 2

            if denom <= 0:
                continue

            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n

            ss_tot = sum_yy - sum_y**2 / n

            if ss_tot <= 0:
                continue

            ss_res = (sum_yy + slope ** 2 * sum_xx + n * intercept ** 2 - 2 * slope * sum_xy - 2 * intercept * sum_y + 2 * slope * intercept * sum_x)

            r2 = 1 - ss_res / ss_tot

            score = r2 

            if score > best_score:
                best_score = score
                best_i     = i
                best_j     = j
                best_slope = slope
                best_r2    = r2

    return best_i, best_j, best_slope, best_r2

# estimate = the estimated value
# expected_value = the expectd_value for the point estimate
# var = the variance of the point estimate
# lambda_penalty = the penalty term that the variance should be multiplied by

def error_calculation(estimate, expected_value, var, lambda_penalty = 1):
    error = np.abs(estimate - expected_value) / np.abs(expected_value) 
    ans   = error + lambda_penalty * (var / expected_value ** 2)
    return ans

# returns a dataframe that contains the best configuration for each time scale
# data = dataframe containing the configurations details with estimates for the exponents with variances and interval half widths
# config = do you want the configuration that fits the microscopic/macroscopic values the best. Microscopic refers to the impact curve exponent
# the phi exponent and beta
# estimation_method = wether you want the results for when gamma is estimated using the NLLS or the PSD method
# lambda_penatly = the penalty term for the variance
# mix = how the ratio that individual errors for the impact curve exponent, exponent for phi, and beta should be mixed to get the total macroscopic error

def find_configs(data, trader_participation_distribution, grouped: bool = False, 
                 config: Literal['metaorder', 'LMF', 'N', 'combined'] = 'metaorder',
                 estimation_method: Literal['NLLS', 'PSD'] = 'PSD', lambda_penalty = 1, mix = [1, 1, 1]):

    impact_error = error_calculation(data[:, 'impact curve exponent'], 0.5, data[:, 'impact curve exponent var'], lambda_penalty)
    phi_error    = error_calculation(data[:, 'phi exponent'], 0.5, data[:, 'phi exponent var'], lambda_penalty)
    gamma        = data[:, f'gamma ({estimation_method.lower()})']
    beta_error   = error_calculation(data[:, 'beta'], 0.5 * (1 - gamma), data[:, 'beta var'], lambda_penalty)
    alpha_error  = np.abs(data[:, 'alpha'] - 1 - gamma) / (gamma + 1)
    N_error      = np.abs(data[:, 'N (st)'] - data[:, f'N ({estimation_method.lower()})']) / data[:, 'N (st)']

    if trader_participation_distribution == 'homogenous':
        
        error_df = pl.DataFrame({'year': data.select('year'), 'ticker': data.select('ticker'), 'N': data.select('N'),
                                 #f'N_{{ST}}': data.select('N (st)'),
                                 #f'N_{{est}}': data.select(f'N ({estimation_method.lower()})'),
                                 #f'$\\delta$': data.select('delta'), 
                                 f'$\\alpha$': data.select('alpha'), 
                                 f'$\\gamma$': data.select(f'gamma ({estimation_method.lower()})'),
                                 
                                 f'\\makecell{{$\\frac{{Q}}{{V_D}}$ \\\\ exponent}}': data.select('impact curve exponent'),
                                 f'\\makecell{{$\\phi$ \\\\ exponent}}': data.select('phi exponent'), 
                                 f'$\\beta$': data.select('beta'), 
                                 f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}': (mix[0] * impact_error + mix[1] * phi_error + mix[2] * beta_error) / sum(mix),
                                 f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}': alpha_error,
                                 f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}': N_error})
                                 #f'\\makecell{{$\\mathrm{{e}}_{{combo}}$ \\\\ ({estimation_method})}}': (mix[0] * impact_error + mix[1] * phi_error + mix[2] * beta_error) / sum(mix) + alpha_error})
   
        error_df = error_df.with_columns(((pl.col(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}') - pl.col(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}').mean().over('ticker')) / pl.col(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}').std().over('ticker')).alias('metaorder error (stnd)'))
        error_df = error_df.with_columns(((pl.col(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}') - pl.col(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}').mean().over('ticker')) / pl.col(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}').std().over('ticker')).alias('LMF error (stnd)')) 
        error_df = error_df.with_columns(((pl.col(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}') - pl.col(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}').mean().over('ticker')) / pl.col(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}').std().over('ticker')).alias('N error (stnd)'))        
        error_df = error_df.with_columns((pl.col('metaorder error (stnd)') + pl.col('LMF error (stnd)')).alias(f'\\makecell{{$\\mathrm{{e}}_{{combo}}$ \\\\ ({estimation_method}) (stnd)}}'))

        if (grouped == True):
                 
            error_df = error_df.group_by(['ticker', 'N', f'$\\delta$']).agg(pl.col('year'), pl.col(f'$\\alpha$'), pl.col(f'$\\gamma$'),
                                                                            pl.col(f'\\makecell{{$\\frac{{Q}}{{V_D}}$ \\\\ exponent}}'), pl.col(f'\\makecell{{$\\phi$ \\\\ exponent}}'),
                                                                            pl.col(f'$\\beta$'),                                                                            
                                                                            pl.col(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{M}}$ \\\\ ({estimation_method})}}'),
                                                                            pl.col('metaorder error (stnd)').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{M}}$ \\\\ ({estimation_method}) (stnd)}}'),
                                                                            pl.col(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{LMF}}$ \\\\ ({estimation_method})}}'),
                                                                            pl.col('LMF error (stnd)').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{LMF}}$ \\\\ ({estimation_method}) (stnd)}}'),
                                                                            pl.col(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{N}}$ \\\\ ({estimation_method})}}'),
                                                                            pl.col('N error (stnd)').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{N}}$ \\\\ ({estimation_method}) (stnd)}}'),
                                                                            ((pl.col('metaorder error (stnd)') + pl.col('LMF error (stnd)')).mean()).alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{combo}}$ \\\\ ({estimation_method}) (stnd)}}'))


            if (config == 'metaorder'):
                sort_how = 'M'
                
            elif (config == 'LMF'):
                sort_how = 'LMF'
                
            elif (config == 'N'):
                sort_how = 'N'
                
            elif (config == 'combined'):
                sort_how = 'combo'
                
            error_df = error_df.sort(['ticker', f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{{sort_how}}}$ \\\\ ({estimation_method}) (stnd)}}'])
            error_df = error_df.group_by('ticker').agg(pl.col('year').first(),
                                                       pl.col('N').first(),
                                                       pl.col(f'$\\delta$').first(),
                                                       pl.col(f'$\\alpha$').first(),
                                                       pl.col(f'$\\gamma$').first(),
                                                       pl.col(f'\\makecell{{$\\frac{{Q}}{{V_D}}$ \\\\ exponent}}').first(),
                                                       pl.col(f'\\makecell{{$\\phi$ \\\\ exponent}}').first(),
                                                       pl.col(f'$\\beta$').first(),
                                                       pl.col(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{M}}$ \\\\ ({estimation_method})}}').first(),
                                                       pl.col(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{LMF}}$ \\\\ ({estimation_method})}}').first(),
                                                       pl.col(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{combo}}$ \\\\ ({estimation_method}) (stnd)}}').first())

            
            return error_df

        if (grouped == False):

            error_df = error_df.drop(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}', 'metaorder error (stnd)', 'LMF error (stnd)', 'N error (stnd)')
            
            if (config == 'metaorder'):
                error_df = error_df.sort(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}').group_by(['ticker', 'year']).first()
                error_df = error_df.sort(['ticker', 'year'])
                return error_df
        
            if (config == 'LMF'):
                error_df = error_df.sort(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}').group_by(['ticker', 'year']).first()
                error_df = error_df.sort(['ticker', 'year'])
                return error_df
        
            # if (config == 'N'):
            #     error_df = error_df.sort(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}').group_by(['ticker', 'year']).first()
            #     error_df = error_df.sort(['ticker', 'year'])
            #     return error_df
        
            if (config == 'combined'):                
                error_df = error_df.sort(f'\\makecell{{$\\mathrm{{e}}_{{combo}}$ \\\\ ({estimation_method}) (stnd)}}').group_by(['ticker', 'year']).first()
                error_df = error_df.sort(['ticker', 'year'])
                return error_df      
        
    if trader_participation_distribution == 'power':
        
        error_df = pl.DataFrame({'year': data.select('year'), 'ticker': data.select('ticker'), 'N': data.select('N'),
                                 #f'N_{{ST}}': data.select('N (st)'),
                                 #f'N_{{est}}': data.select(f'N ({estimation_method.lower()})'),
                                 f'$\\delta$': data.select('delta'), 
                                 f'$\\alpha$': data.select('alpha'), 
                                 f'$\\gamma$': data.select(f'gamma ({estimation_method.lower()})'),
                                 
                                 f'\\makecell{{$\\frac{{Q}}{{V_D}}$ \\\\ exponent}}': data.select('impact curve exponent'),
                                 f'\\makecell{{$\\phi$ \\\\ exponent}}': data.select('phi exponent'), 
                                 f'$\\beta$': data.select('beta'), 
                                 f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}': (mix[0] * impact_error + mix[1] * phi_error + mix[2] * beta_error) / sum(mix),
                                 f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}': alpha_error,
                                 f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}': N_error})
                                 #f'\\makecell{{$\\mathrm{{e}}_{{combo}}$ \\\\ ({estimation_method})}}': (mix[0] * impact_error + mix[1] * phi_error + mix[2] * beta_error) / sum(mix) + alpha_error})
   
        error_df = error_df.with_columns(((pl.col(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}') - pl.col(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}').mean().over('ticker')) / pl.col(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}').std().over('ticker')).alias('metaorder error (stnd)'))
        error_df = error_df.with_columns(((pl.col(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}') - pl.col(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}').mean().over('ticker')) / pl.col(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}').std().over('ticker')).alias('LMF error (stnd)')) 
        error_df = error_df.with_columns(((pl.col(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}') - pl.col(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}').mean().over('ticker')) / pl.col(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}').std().over('ticker')).alias('N error (stnd)'))        
        error_df = error_df.with_columns((pl.col('metaorder error (stnd)') + pl.col('LMF error (stnd)')).alias(f'\\makecell{{$\\mathrm{{e}}_{{combo}}$ \\\\ ({estimation_method}) (stnd)}}'))

        if (grouped == True):
            
            error_df = error_df.group_by(['ticker', 'N', f'$\\delta$']).agg(pl.col('year'), pl.col(f'$\\alpha$'), pl.col(f'$\\gamma$'),
                                                                            pl.col(f'\\makecell{{$\\frac{{Q}}{{V_D}}$ \\\\ exponent}}'), pl.col(f'\\makecell{{$\\phi$ \\\\ exponent}}'),
                                                                            pl.col(f'$\\beta$'),                                                                            
                                                                            pl.col(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{M}}$ \\\\ ({estimation_method})}}'),
                                                                            pl.col('metaorder error (stnd)').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{M}}$ \\\\ ({estimation_method}) (stnd)}}'),
                                                                            pl.col(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{LMF}}$ \\\\ ({estimation_method})}}'),
                                                                            pl.col('LMF error (stnd)').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{LMF}}$ \\\\ ({estimation_method}) (stnd)}}'),
                                                                            pl.col(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{N}}$ \\\\ ({estimation_method})}}'),
                                                                            pl.col('N error (stnd)').mean().alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{N}}$ \\\\ ({estimation_method}) (stnd)}}'),
                                                                            ((pl.col('metaorder error (stnd)') + pl.col('LMF error (stnd)')).mean()).alias(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{combo}}$ \\\\ ({estimation_method}) (stnd)}}'))


            if (config == 'metaorder'):
                sort_how = 'M'
                
            elif (config == 'LMF'):
                sort_how = 'LMF'
                
            elif (config == 'N'):
                sort_how = 'N'
                
            elif (config == 'combined'):
                sort_how = 'combo'
                
            error_df = error_df.sort(['ticker', f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{{sort_how}}}$ \\\\ ({estimation_method}) (stnd)}}'])
            error_df = error_df.group_by('ticker').agg(pl.col('year').first(),
                                                       pl.col('N').first(),
                                                       pl.col(f'$\\delta$').first(),
                                                       pl.col(f'$\\alpha$').first(),
                                                       pl.col(f'$\\gamma$').first(),
                                                       pl.col(f'\\makecell{{$\\frac{{Q}}{{V_D}}$ \\\\ exponent}}').first(),
                                                       pl.col(f'\\makecell{{$\\phi$ \\\\ exponent}}').first(),
                                                       pl.col(f'$\\beta$').first(),
                                                       pl.col(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{M}}$ \\\\ ({estimation_method})}}').first(),
                                                       pl.col(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{LMF}}$ \\\\ ({estimation_method})}}').first(),
                                                       pl.col(f'\\makecell{{$\\bar{{\\mathrm{{e}}}}_{{combo}}$ \\\\ ({estimation_method}) (stnd)}}').first())

            
            return error_df

        if (grouped == False):

            error_df = error_df.drop(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}', 'metaorder error (stnd)', 'LMF error (stnd)', 'N error (stnd)')
            
            if (config == 'metaorder'):
                error_df = error_df.sort(f'\\makecell{{$\\mathrm{{e}}_{{M}}$ \\\\ ({estimation_method})}}').group_by(['ticker', 'year']).first()
                error_df = error_df.sort(['ticker', 'year'])
                return error_df
        
            if (config == 'LMF'):
                error_df = error_df.sort(f'\\makecell{{$\\mathrm{{e}}_{{LMF}}$ \\\\ ({estimation_method})}}').group_by(['ticker', 'year']).first()
                error_df = error_df.sort(['ticker', 'year'])
                return error_df
        
            # if (config == 'N'):
            #     error_df = error_df.sort(f'\\makecell{{$\\mathrm{{e}}_{{N}}$ \\\\ ({estimation_method})}}').group_by(['ticker', 'year']).first()
            #     error_df = error_df.sort(['ticker', 'year'])
            #     return error_df
        
            if (config == 'combined'):                
                error_df = error_df.sort(f'\\makecell{{$\\mathrm{{e}}_{{combo}}$ \\\\ ({estimation_method}) (stnd)}}').group_by(['ticker', 'year']).first()
                error_df = error_df.sort(['ticker', 'year'])
                return error_df     

