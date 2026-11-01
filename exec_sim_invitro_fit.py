# Script predicting IL2-induced pSTAT5 among different immune cell types and cancer cell types using different IL-2 variants

from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from src.models.IL2s_parallel import model_function
import pandas as pd
from statistics import median
from sys import exit
from scipy.optimize import curve_fit
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import set_simulation,simulate,func_time

# Import datasets
df_IC_data = pd.read_csv('data/expression/whole_dataset.csv')
df_linear_models = pd.read_csv('data/lin_models/linear_models.csv')
df_sim_data = pd.read_csv('data/signaling/IL2_data.csv')
df_bind = pd.read_csv('results/fit_ABC_SMC/IL2_data_param_ABC_SMC.csv')

# Create dataset of simulations
num_sim = 20
df_sim = set_simulation(df_bind, df_sim_data, df_IC_data, df_linear_models, num_sim)

# Simulate
num_cores = 24
df_res = simulate(df_sim, num_cores, model_function)
df_res.to_csv("results/simulations_IL2s_fit.csv")


