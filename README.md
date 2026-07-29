# Metaorder modelling and identification



## Thesis information
**Institution:** University of Cape Town
**Programme:** MSc in Statistics and Data Science
**Author:** Ezra Goliath
**Supervisor:** Tim Gebbie
**Status:** Work in progress

## Overview

This repository contains the code used to reconstruct synthetic metaorders from anonymous transaction and quote data and to evaluate their consistency with the Lillo–Mike–Farmer (LMF) order-splitting theory. The analysis uses data for 239 Johannesburg Stock Exchange securities and evaluates candidate reconstruction configurations against established metaorder stylised facts and the LMF exponent relation. The repository includes the data-processing, metaorder-reconstruction, model-evaluation, and figure-generation code used in the accompanying master's thesis pre-print. 

## Abstract

Market-order flow in financial markets exhibits long-range correlations. This is a widely known stylised fact of financial markets. A popular hypothesis for this stylised fact comes from the Lillo–Mike–Farmer (LMF) order-splitting theory. However, quantitative tests of this theory have historically relied on proprietary datasets with trader identifiers, limiting reproducibility and cross-market validation. We investigate whether it can be recovered from anonymous public data using synthetic metaorder reconstruction. Using transaction and quote data for the largest 239 stocks by market capitlisation on the JSE as of 13 March 2026, we conduct a grid search over reconstruction parameters and evaluate each configuration against established metaorder stylised facts and the LMF relation. Configurations selected to minimise errors across the metaorder stylised facts reproduce those properties but yield a poor LMF relation. In contrast, configurations selected to minimise error in the LMF relation reproduce the LMF relation but also reproduce the metaorder stylised facts well. These asymmetric results show that recovering the stylised facts alone is insufficient to identify LMF-consistent order splitting, while configurations consistent with LMF provide a stronger joint description of the reconstructed order flow. The findings support consistency with, rather than direct validation of, the LMF theory using anonymous market data.


# Repository structure:

data_processing_notebooks: contains notebooks for processing raw data and outputting .parquet files used for the metaorder generation, plots and tables.

plotting_notebooks: contains the code for processing .parquet files and outputting plots.

