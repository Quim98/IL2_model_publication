from pysb import *
from pysb.integrate import Solver
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import curve_fit
from multiprocessing import Pool
from itertools import combinations_with_replacement
import sympy as sp
from scipy.optimize import fsolve
import pyabc
import pyabc.sge

import time # BORRAR

def func_time(IL):
    if IL > 1e-8:
        return 0.1,1e4
    else:
        IL = abs(np.log10(IL)) - 4
        return 10**(IL)/(1e5),10**(IL)

######################################################################################################################################################    
    
def simulate(df_sim_data, num_cores, model_func):
    param_parallel = []
    for i in range(0,len(df_sim_data.index)):
        param_parallel.append([df_sim_data.loc[[i]]])
    with Pool(processes=num_cores) as pool:
        out_vec = pool.map(model_func, param_parallel)  
    return pd.concat(out_vec, axis=0)

######################################################################################################################################################

def set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, cell_list, variant_pair, STAT_list, IL_vec_1, IL_vec_2):
    num_sim = len(IL_vec_1)
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant_1","Variant_2","STAT_type"]+list(df_bind.columns[1:]+"_1")+list(df_bind.columns[1:]+"_2")+["IL0_1","IL0_2","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    plot_i = 0
    for cell in cell_list:
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==cell]
        for STAT_type in STAT_list:
            for i in range(0,num_sim):
                IL0 = np.mean([IL_vec_1[i],IL_vec_2[i]])
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log10 Prot. count"].values[0]):
                    RA0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RA_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RA_HUMAN"]["Slope"].values[0]
                else:
                    RA0 = df_cell.loc[df_cell["Gene"]=="IL2RA_HUMAN"]["Log10 Prot. count"].values[0]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log10 Prot. count"].values[0]):
                    RB0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RB_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RB_HUMAN"]["Slope"].values[0]
                else:
                    RB0 = df_cell.loc[df_cell["Gene"]=="IL2RB_HUMAN"]["Log10 Prot. count"].values[0]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log10 Prot. count"].values[0]):
                    RG0 = df_linear_models.loc[df_linear_models["Gene"]=="IL2RG_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="IL2RG_HUMAN"]["Slope"].values[0]
                else:
                    RG0 = df_cell.loc[df_cell["Gene"]=="IL2RG_HUMAN"]["Log10 Prot. count"].values[0]
                if np.isnan(df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log10 Prot. count"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log10 Prot. count"].values[0]):
                    STAT50 = np.log10(10**(df_linear_models.loc[df_linear_models["Gene"]=="STA5A_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="STA5A_HUMAN"]["Slope"].values[0]) + 10**(df_linear_models.loc[df_linear_models["Gene"]=="STA5B_HUMAN"]["Intercept"].values[0] + df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log2 TPM"].values[0] * df_linear_models.loc[df_linear_models["Gene"]=="STA5B_HUMAN"]["Slope"].values[0]))
                else:
                    STAT50 = np.log10(10**(df_cell.loc[df_cell["Gene"]=="STA5A_HUMAN"]["Log10 Prot. count"].values[0]) + 10**(df_cell.loc[df_cell["Gene"]=="STA5B_HUMAN"]["Log10 Prot. count"].values[0]))
                dT, Tf = func_time(IL0)
                df_sim.loc[len(df_sim.index)] = [plot_i,cell,variant_pair[0],variant_pair[1],STAT_type] + df_bind.loc[df_bind["Variant"]==variant_pair[0]].values.flatten().tolist()[1:] + df_bind.loc[df_bind["Variant"]==variant_pair[1]].values.flatten().tolist()[1:] + [IL_vec_1[i],IL_vec_2[i],RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
        plot_i += 1
    return df_sim

######################################################################################################################################################

def hill_func(inp,B,K):
    out=B*inp/(K+inp)
    return out