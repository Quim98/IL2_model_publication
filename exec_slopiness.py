# Script performing slopiness analysis to rank the model’s parameters by their effect on its prediction

from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from src.models.IL2_parallel import model_function
import pandas as pd
from statistics import median
from sys import exit
from scipy.optimize import curve_fit
from multiprocessing import Pool
import matplotlib.pyplot as plt
from src.simulate_func import set_simulation_fit,simulate,func_time,update_df_bind,update_df_bind_up_down,local_slopiness,calculate_cost_slopiness_fit
from multiprocessing import Pool
import time # BORRAR

# Import datasets
df_bind = pd.read_csv('data/binding/IL2_data_param.csv')
df_sim_data = pd.read_csv('data/signaling/IL2_data.csv')
df_EC50_AMP = pd.read_csv('data/signaling/IL2_data_hill.csv')
df_linear_models = pd.read_csv('data/lin_models/linear_models.csv')
df_IC_data = pd.read_csv('data/expression/whole_dataset.csv')
num_IL_sim = 20
num_cores = 24
    
# param_mut = {"WT": [elem for elem in df_bind.columns[1:] if '_f' not in elem][1:-1],"NKTR-358":['k_IL_RA_b','k_IL_RB_b'],"MDNA11":['k_IL_RB_b'],"ALKS4230":['k_IL_RB_b'],"THOR-707":['k_IL_RA_b','k_IL_RB_b']} # Erase mutants that were not fitted
param_depen = {'k_IL_RB_b':['k_IL_RA_RB_b'], 'k_RA_RB_b':['k_IL_RA_RB_b'], 'k_IL_RB_RG_b':['k_IL_RA_RB_RG_b'], 'k_IL_RA_RB_RG_b':['k_IL_RB_RG_b']}
param_mut = ['vol_EC','surf_cell','vol_cell','k_IL_RA_b','k_IL_RB_b','k_IL_RG_b','k_IL_RA_RB_b','k_IL_RB_RG_b','k_IL_RA_RB_RG_b','k_STAT5_b','k_PHOS','k_DEPHOS']

# Analyze the model's slopiness
cells_target = cells_notarget = 0
mode = "Fitting"
delta_0 = 5e-4
H = local_slopiness(delta_0, mode, df_bind, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim, cells_target, cells_notarget, num_cores, model_function, param_mut, param_depen)
np.savetxt('results/Hessian_nofit.csv', H, delimiter=',')