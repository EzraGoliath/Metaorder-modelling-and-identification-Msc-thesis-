# Metaorder modelling and identification



## Thesis information
- **Institution:** University of Cape Town 
- **Programme:** MSc in Statistics and Data Science 
- **Author:** Ezra Goliath 
- **Supervisor:** Tim Gebbie 
- **Status:** Work in progress

> This repository accompanies the master's thesis *Metaorder Modelling and Identification*. The code and documentation may change while the thesis is under development.

## Overview

This repository contains the code used to reconstruct synthetic metaorders from anonymous transaction and quote data and to evaluate their consistency with the Lillo–Mike–Farmer (LMF) order-splitting theory. The analysis uses data for 239 Johannesburg Stock Exchange securities and evaluates candidate reconstruction configurations against established metaorder stylised facts and the LMF exponent relation. The repository includes the data-processing, metaorder-reconstruction, model-evaluation, and figure-generation code used in the accompanying master's thesis pre-print. 

## Abstract

Market-order flow in financial markets exhibits long-range correlations. This is a widely known stylised fact of financial markets. A popular hypothesis for this stylised fact comes from the Lillo–Mike–Farmer (LMF) order-splitting theory. However, quantitative tests of this theory have historically relied on proprietary datasets with trader identifiers, limiting reproducibility and cross-market validation. We investigate whether it can be recovered from anonymous public data using synthetic metaorder reconstruction. Using transaction and quote data for the largest 239 stocks by market capitlisation on the JSE as of 13 March 2026, we conduct a grid search over reconstruction parameters and evaluate each configuration against established metaorder stylised facts and the LMF relation. Configurations selected to minimise errors across the metaorder stylised facts reproduce those properties but yield a poor LMF relation. In contrast, configurations selected to minimise error in the LMF relation reproduce the LMF relation but also reproduce the metaorder stylised facts well. These asymmetric results show that recovering the stylised facts alone is insufficient to identify LMF-consistent order splitting, while configurations consistent with LMF provide a stronger joint description of the reconstructed order flow. The findings support consistency with, rather than direct validation of, the LMF theory using anonymous market data.

## Repository contents

- process transaction and quote data
- reconstruct synthetic metaorders
- evaluate reconstructed metaorders against empirical stylised facts
- test consistency with the LMF exponent relation
- perform parameter-grid searches
- generate the figures and tables used in the thesis

## Repository structure:

## Repository Structure

This repository contains the code, calibration outputs, and supporting data used in the thesis analysis and validation of the Lillo-Mike-Farmer (LMF) order-splitting framework.

### Directory Layout

```text

├── Calibration_results/
│   └── calibration_results.csv
│
├── Data_processing_notebooks/
│   ├── data_retriever_BMLL.ipynb   
│   └── grid_search_BMLL.ipynb
│
├── Modules/
│   ├── metaorders.py
│   └── LMF.py
│
├── Plotting_notebooks/
│   ├── alpha_gamma_plots_BMLL.ipynb
│   ├── calibration_results_BMLL.ipynb
│   ├── concave_profile_plots_BMLL.ipynb
│   ├── decay_plots_BMLL.ipynb
│   ├── impact_curves_plots_BMLL.pynb
│   └── time_independence_plots_BMLL.ipynb
│
├── Stock_information/
│   └── JSE_companies_list.csv
│
└── README.md
```

### Folder Descriptions

#### `Calibration_results/`

Contains a CSV file containing the grid search results used throughout the analysis.

#### `Data_processing_notebooks/`

Contains Jupyter notebooks used to:

- Retrieve limit order book data from the BMLL platform.
- Process and clean the raw data.
- Generate synthetic metaorders.
- Perform the calibration grid search for the metaorder generation model.

#### `Plotting_notebooks/`

Contains notebooks used to analyse calibration results and reproduce the figures presented in the thesis, including:

- Market impact curves
- Independence plots
- Concave profile plots
- Impact decay plots
- Alpha-Gamma relationship plots

#### `Modules/`

Contains reusable Python modules implementing the core methodology developed in the thesis, including:

- Synthetic metaorder generation
- LMF model calculations
- Alpha and Gamma estimation
- Utility functions used across notebooks

#### `Stock_information/`

Contains a CSV file that contains information on the stocks included in the analysis, such as ticker symbols, company names, and other metadata required for filtering and interpretation of results.

### Workflow

1. Retrieve raw order data from BMLL using the `data_retriever_BMLL.pynb` file.
    ↓
2. Perform a grid search and save the results to a CSV file using the `grid_search_BMLL.pynb` file
    ↓
3. Retrieve and interpret the grid search data using the `calibration_results_BMLL.pynb` file.
    ↓
4. Plot the relevant plots for the optimal configurations when minimising $e_{\mathrm{M}}$ and $e_{\mathrm{LMF}}$. All of this is done using the `calibration_results_BMLL.pynb` file.
    ↓
5. Individual plots can be acquired using the relevant plotting notebooks in the `Plotting_notebooks/` folder.


