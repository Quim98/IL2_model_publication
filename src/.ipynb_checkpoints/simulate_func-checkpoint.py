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
from scipy.interpolate import interp1d

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

def set_simulation(df_bind, df_sim_data, df_IC_data, df_linear_models, num_sim):
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        IL_vec = 10**(np.linspace(np.log10(df_plot["IL"].min()), np.log10(df_plot["IL"].max()*10), num_sim))
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            for i in range(0,num_sim):
                IL0 = IL_vec[i]
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
                df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
    return df_sim

######################################################################################################################################################

def set_simulation_cell_list(df_bind, df_IC_data, df_linear_models, cell_list, variant_list, STAT_list, num_sim, low_IL, high_IL):
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    plot_i = 0
    for cell in cell_list:
        IL_vec = 10**(np.linspace(np.log10(low_IL), np.log10(high_IL), num_sim))
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==cell]
        for variant in variant_list:
            for STAT_type in STAT_list:
                for i in range(0,num_sim):
                    IL0 = IL_vec[i]
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
                    df_sim.loc[len(df_sim.index)] = [plot_i,cell,variant,STAT_type] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
        plot_i += 1
    return df_sim

######################################################################################################################################################

def hill_func(inp,B,K):
    out=B*inp/(K+inp)
    return out

######################################################################################################################################################

def sensitivity_analysis(df_bind, df_EC50_data, df_IC_data, df_linear_models, num_IL_sim, num_var_sim, vec_param, cell_sim):
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Var. parameter","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    param_vec = list(10**(np.linspace(np.log10(0.001), np.log10(1), int(num_var_sim/2)+1))) + list(10**(np.linspace(np.log10(1), np.log10(1000), int(num_var_sim/2)+1)))[1:]
    plot_num = 0
    for parameter in vec_param:
        for cell in cell_sim:
            df_cell = df_IC_data.loc[df_IC_data["Cell"]==cell]
            # Higher IL2 concentrations since model overestiates EC50
            IL_vec = 10**(np.linspace(df_EC50_data.loc[df_EC50_data["Cell_type"] == cell]["EC50_t"].values[0]-3, df_EC50_data.loc[df_EC50_data["Cell_type"] == cell]["EC50_t"].values[0]+5, num_IL_sim))
            for j in range(0,num_var_sim):
                df_WT = df_bind.loc[df_bind["Variant"] == "WT"]
                df_WT[parameter] = df_WT[parameter]*param_vec[j]
                for i in range(0,num_IL_sim):
                    IL0 = IL_vec[i]
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
                    df_sim.loc[len(df_sim.index)] = [plot_num, cell, parameter, "pSTAT5"] + df_WT.values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
                plot_num += 1
    return df_sim

######################################################################################################################################################

## FUNCTIONS FOR THE FITTING OF THE IL2 MODEL

def calculate_RSS(df_sim, df_EC50_AMP, w_EC50, w_AMP):
    for plot_num in list(dict.fromkeys(df_sim["Plot"])):
        df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        i = 0
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max() 
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            else:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            i += 1
    df_EC50_AMP["EC50_e"] = (df_EC50_AMP["EC50_t"] - df_EC50_AMP["EC50_m"])**2/w_EC50
    df_EC50_AMP["Amp_e"] = (df_EC50_AMP["Amp_t"] - df_EC50_AMP["Amp_m"])**2/w_AMP
    df_EC50_AMP["Error"] = df_EC50_AMP["EC50_e"] + df_EC50_AMP["Amp_e"]
    
    print("Mean EC50 error: "+str((df_EC50_AMP["EC50_e"].mean()*w_EC50)**(1/2))+" Mean Amp error: "+str((df_EC50_AMP["Amp_e"].mean()*w_AMP)**(1/2))+" RSS: "+str(df_EC50_AMP["Error"].sum()))
    
    return df_EC50_AMP["Error"].sum()

######################################################################################################################################################

def update_df_bind(df_bind, param_list, param_mut, df_fit_data, param_depen):
    # Create copy of binding dataframe
    df_bind_fit = df_bind.copy()

    # Update df_bind with new parameter values
    i=0
    for variant in list(dict.fromkeys(df_fit_data["Variant"])):
        if variant == "WT":
            for param in df_fit_data.loc[df_fit_data["Variant"] == variant]["Parameter"]:
                # Copy new parameter to dataframe
                df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param] = param_list[i]
                # Parameter dependancies
                if param in param_depen.keys():
                    for param_d in param_depen[param]:
                        param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                        param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                        df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param_d] = param_list[i]/param1_0*param2_0
                # Parameter which are the same for WT and other variants
                for variant_WT_mut in list(param_mut.keys())[1:]:
                    if param not in param_mut[variant_WT_mut]:
                        param_dep_bool = True
                        for param2 in param_mut[variant_WT_mut]:
                            if param2 in param_depen.keys():
                                if param == param_depen[param2]:
                                    param_dep_bool = False
                        if param_dep_bool:
                            df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).Variant, :].index[0], param] = param_list[i]
                            # Parameter dependancies
                            if param in param_depen.keys():
                                for param_d in param_depen[param]:
                                    param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                                    param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                                    df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).Variant, :].index[0], param_d] = param_list[i]/param1_0*param2_0
                i += 1
        else:
            for param in df_fit_data.loc[df_fit_data["Variant"] == variant]["Parameter"]:
                df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param] = param_list[i]
                # Parameter dependancies
                if param in param_depen.keys():
                    for param_d in param_depen[param]:
                        param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                        param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                        df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param_d] = param_list[i]/param1_0*param2_0
                i += 1
    return df_bind_fit

######################################################################################################################################################

def set_simulation_fit(df_bind, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim):
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            df_var_EC50 = df_plot_EC50.loc[df_plot_EC50["Variant"] == variant]
            IL_vec =  10**(np.linspace(df_var_EC50["EC50_t"].values[0]-3, df_var_EC50["EC50_t"].values[0]+4, num_IL_sim))
            for i in range(0,num_IL_sim):
                IL0 = IL_vec[i]
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
                df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-1] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[2:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
    return df_sim

######################################################################################################################################################

def fit_data(df_bind, df_sim_data, df_EC50_AMP, df_linear_models, df_IC_data, df_fit_data, num_IL_sim, num_cores, model_function, param_mut, param_depen, num_iter, exit_cond):
     
    def error_func(param_list):
        start = time.time() # BORRAR
        
        # Update parameters dataframe
        df_bind_fit = update_df_bind(df_bind, param_list, param_mut, df_fit_data, param_depen)
        
        # Create dataset of simulations
        df_sim = set_simulation_fit(df_bind_fit, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim)
        
        # Simulate
        df_res = simulate(df_sim, num_cores, model_function)
        
        # Calculate error
        error = calculate_RSS(df_res, df_EC50_AMP, w_EC50, w_AMP)
        
        print(param_list)
        print("Elapsed time in seconds per fit iteration: " + str(time.time() - start)) # BORRAR
        return error
    
    
    # Create first simulation dataframe
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            df_var_EC50 = df_plot_EC50.loc[df_plot_EC50["Variant"] == variant]
            IL_vec =  10**(np.linspace(df_var_EC50["EC50_t"].values[0]-3, df_var_EC50["EC50_t"].values[0]+4, num_IL_sim))
            for i in range(0,num_IL_sim):
                IL0 = IL_vec[i]
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
                df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
    
    
    # Perform simulations with raw parameters
    df_sim = simulate(df_sim, num_cores, model_function)
    
    # Calculate EC50 and Amplitude error to add weight
    for plot_num in list(dict.fromkeys(df_sim["Plot"])):
        df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        i = 0
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max()
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            else:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            i += 1
    df_EC50_AMP["EC50_e"] = (df_EC50_AMP["EC50_t"] - df_EC50_AMP["EC50_m"])**2
    df_EC50_AMP["Amp_e"] = (df_EC50_AMP["Amp_t"] - df_EC50_AMP["Amp_m"])**2
    w_EC50 = df_EC50_AMP["EC50_e"].mean()
    w_AMP = df_EC50_AMP["Amp_e"].mean()
    print("EC50 mean error: " + str(w_EC50**(1/2)))
    print("Amplitude mean error: " + str(w_AMP**(1/2)))
    
    # Initial values and bounds for the parameters to fit
    list_initial = []
    list_bounds = []
    for variant in list(dict.fromkeys(df_fit_data["Variant"])):
        df_fit_variant = df_fit_data.loc[df_fit_data["Variant"] == variant]
        for param in df_fit_variant["Parameter"]:
            list_initial.append(df_fit_variant.loc[df_fit_variant["Parameter"] == param]["Initial_val"].values[0])
            list_bounds.append((df_fit_variant.loc[df_fit_variant["Parameter"] == param]["Lower_bound"].values[0],df_fit_variant.loc[df_fit_variant["Parameter"] == param]["Upper_bound"].values[0]))
            
    res = optimize.minimize(
        error_func,
        list_initial,
        bounds=list_bounds,
        method="Powell",
        options={ 'maxiter':num_iter, 'ftol':exit_cond}
    )
    print("Finalized fitting of parameters")
    
    return df_sim, df_EC50_AMP, res

######################################################################################################################################################

def calculate_cost_target1(df_res, cells_target, cells_notarget, w_EC50, w_Amp):
    EC50_target = []
    EC50_notarget = []
    Amp_target = []
    Amp_notarget = []
    cells = cells_target + cells_notarget 
    max_target = df_res.loc[df_res["Cell_type"]==cells_target[0],"Result"].max() 
    for cell in cells:
        df_cell = df_res.loc[df_res["Cell_type"]==cell]
        fit, cov = curve_fit(hill_func, df_cell["IL0"], df_cell["Result"].values*100/(max_target), bounds = ([0,df_cell["IL0"].min()/10], [df_cell["Result"].max()*100/(max_target)+10, df_cell["IL0"].max()*10]))
        if cell in cells_target:
            EC50_target.append(abs(np.log10(fit[1])))
            Amp_target.append(abs(fit[0]))
        else:
            EC50_notarget.append(abs(np.log10(fit[1])))
            Amp_notarget.append(abs(fit[0]))
    
    print("Mean EC50 target cells: "+str(np.mean(EC50_target)))
    print("Mean EC50 non-target cells: "+str(np.mean(EC50_notarget)))
    print("Mean amplitude target cells: "+str(np.mean(Amp_target)))
    print("Mean amplitude non-target cells: "+str(np.mean(Amp_notarget)))
    print("Cost EC50: "+str(((np.mean(EC50_target)-np.mean(EC50_notarget))/w_EC50)))
    print("Cost amplitude: "+str(((np.mean(Amp_target)-np.mean(Amp_notarget))/w_Amp)))
    print("Total cost: "+str(-(np.mean(EC50_target)-np.mean(EC50_notarget))/w_EC50 - (np.mean(Amp_target)-np.mean(Amp_notarget))/w_Amp))
    
    return -(np.mean(EC50_target)-np.mean(EC50_notarget))/w_EC50 - (np.mean(Amp_target)-np.mean(Amp_notarget))/w_Amp

######################################################################################################################################################

def max_EC50(df_bind, df_linear_models, df_IC_data, num_IL_sim, num_cores, model_function, param_depen, num_iter, exit_cond, param_change, cells_target, cells_notarget):
    def error_func(param_list):
        start = time.time() # BORRAR
        # Update parameters dataframe
        for i in range(0,len(param_change)):
            df_bind.loc[df_bind["Variant"]=="Mut",param_change[i]] = param_list[i]
        
        # Create new simulation dataset
        k=0
        cells = cells_target + cells_notarget
        df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
        for cell in cells:
            IL_vec =  10**(np.linspace(-14, -6, num_IL_sim))
            df_cell = df_IC_data.loc[df_IC_data["Cell"]==cell]
            for i in range(0,num_IL_sim):
                IL0 = IL_vec[i]
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
                df_sim.loc[len(df_sim.index)] = [k,cell, 'Mut', 'pSTAT5'] + df_bind.loc[df_bind["Variant"]=="Mut"].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
            k += 1
        
        # Simulate
        df_res = simulate(df_sim, num_cores, model_function)
        
        # Calculate error
        error = calculate_cost_target1(df_res, cells_target, cells_notarget, w_EC50, w_Amp)
        
        print(param_list)
        print("Elapsed time in seconds per fit iteration: " + str(time.time() - start)) # BORRAR
        return error
    
    # Get only mutant being designed
    df_bind = df_bind.loc[df_bind["Variant"]=="WT"]
    df_bind.loc[df_bind["Variant"]=="WT","Variant"] = "Mut"
    
    # First simulation to generate the weights
    
    # Create new simulation dataset
    k=0
    cells = cells_target + cells_notarget
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    for cell in cells:
        IL_vec =  10**(np.linspace(-14, -6, num_IL_sim))
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==cell]
        for i in range(0,num_IL_sim):
            IL0 = IL_vec[i]
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
            df_sim.loc[len(df_sim.index)] = [k,cell, 'Mut', 'pSTAT5'] + df_bind.loc[df_bind["Variant"]=="Mut"].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
        k += 1

    # Simulate
    df_res = simulate(df_sim, num_cores, model_function)
    
    # Generate weights
    EC50_target = []
    EC50_notarget = []
    Amp_target = []
    Amp_notarget = []
    cells = cells_target + cells_notarget 
    max_target = df_res.loc[df_res["Cell_type"]==cells_target[0],"Result"].max() 
    for cell in cells:
        df_cell = df_res.loc[df_res["Cell_type"]==cell]
        fit, cov = curve_fit(hill_func, df_cell["IL0"], df_cell["Result"].values*100/(max_target), bounds = ([0,df_cell["IL0"].min()/10], [df_cell["Result"].max()*100/(max_target)+10, df_cell["IL0"].max()*10]))
        if cell in cells_target:
            EC50_target.append(abs(np.log10(fit[1])))
            Amp_target.append(abs(fit[0]))
        else:
            EC50_notarget.append(abs(np.log10(fit[1])))
            Amp_notarget.append(abs(fit[0]))
    w_EC50 = abs(np.mean(EC50_target)-np.mean(EC50_notarget))
    w_Amp = abs(np.mean(Amp_target)-np.mean(Amp_notarget))
    
    print("Minimization algorithm begins!")
    # Initial values and bounds for the parameters to fit
    list_initial = []
    list_bounds = []
    for param in param_change:
        list_initial.append(df_bind.loc[0,param])
        list_bounds.append((df_bind.loc[0,param]/100, df_bind.loc[0,param]*100))
            
    res = optimize.minimize(
        error_func,
        list_initial,
        bounds=list_bounds,
        method="Nelder-Mead",
        options={ 'maxiter':num_iter, 'fatol':exit_cond}
    )
    print("Finalized fitting of parameters")
    return res

######################################################################################################################################################

def set_simulation_mutant_optimization(df_bind, df_IC_data, df_linear_models, num_IL_sim, cells_target, cells_notarget):
    k=0
    cells = cells_target + cells_notarget
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    for cell in cells:
        IL_vec =  10**(np.linspace(-14, -6, num_IL_sim))
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==cell]
        for i in range(0,num_IL_sim):
            IL0 = IL_vec[i]
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
            df_sim.loc[len(df_sim.index)] = [k,cell, 'Mut', 'pSTAT5'] + df_bind.loc[df_bind["Variant"]=="Mut"].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
        k += 1
    return df_sim

######################################################################################################################################################

def calculate_cost_slopiness_fit(df_sim, df_EC50_AMP, w_EC50, w_AMP):
    for plot_num in list(dict.fromkeys(df_sim["Plot"])):
        df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        i = 0
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max() 
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            else:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
            i += 1
    df_EC50_AMP["EC50_e"] = (df_EC50_AMP["EC50_t"] - df_EC50_AMP["EC50_m"])**2/w_EC50
    df_EC50_AMP["Amp_e"] = (df_EC50_AMP["Amp_t"] - df_EC50_AMP["Amp_m"])**2/w_AMP
    df_EC50_AMP["Error"] = df_EC50_AMP["EC50_e"] + df_EC50_AMP["Amp_e"]
    
    return df_EC50_AMP

######################################################################################################################################################

def calculate_cost_slopiness_target(df_res, cells_target, cells_notarget, w_EC50, w_Amp):
    EC50_target = {}
    EC50_notarget = {}
    Amp_target = {}
    Amp_notarget = {}
    cells = cells_target + cells_notarget 
    max_target = df_res.loc[df_res["Cell_type"]==cells_target[0],"Result"].max() 
    for cell in cells:
        df_cell = df_res.loc[df_res["Cell_type"]==cell]
        fit, cov = curve_fit(hill_func, df_cell["IL0"], df_cell["Result"].values*100/(max_target), bounds = ([0,df_cell["IL0"].min()/10], [df_cell["Result"].max()*100/(max_target)+10, df_cell["IL0"].max()*10]))
        if cell in cells_target:
            EC50_target[cell] = abs(np.log10(fit[1]))
            Amp_target[cell] = abs(fit[0])
        else:
            EC50_notarget[cell] = abs(np.log10(fit[1]))
            Amp_notarget[cell] = abs(fit[0])
    cost_vec = []
    for cell_target in cells_target:
        for cell_notarget in cells_notarget:
            cost_vec.append(10**((Amp_notarget[cell_notarget] - Amp_target[cell_target])/w_Amp+(EC50_notarget[cell_notarget] - EC50_target[cell_target])/w_EC50-1))
    
    return cost_vec

######################################################################################################################################################

def update_df_bind_up_down(df_bind, param, param_depen, delta):
    # Create copies of binding dataframe
    df_bind_up = df_bind.copy()
    df_bind_down = df_bind.copy()

    # Update df_bind with new parameter values
    df_bind_up[param] = df_bind[param] + delta*df_bind[param]/2
    df_bind_down[param] = df_bind[param] - delta*df_bind[param]/2

    # Parameter dependancies
    if param in param_depen.keys():
        for param_d in param_depen[param]:
            df_bind_up[param_d] = df_bind[param_d] + delta*df_bind[param_d]/2
            df_bind_down[param_d] = df_bind[param_d] - delta*df_bind[param_d]/2
            
    return df_bind_up, df_bind_down

######################################################################################################################################################

def local_slopiness(delta_0, mode, df_bind, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim, cells_target, cells_notarget, num_cores, model_function, param_mut, param_depen):
    # Perform an initial simualtion to get weights
    # Create dataset of simulations
    if mode == "Fitting":
        
        df_sim = set_simulation_fit(df_bind, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim)

    elif mode == "Mutant_Optimization":
        
        # Change df_bind to include the novel mutant
        df_bind = df_bind.loc[df_bind["Variant"]=="WT"]
        df_bind.loc[df_bind["Variant"]=="WT","Variant"] = "Mut"
        
        df_sim = set_simulation_mutant_optimization(df_bind, df_IC_data, df_linear_models, num_IL_sim, cells_target, cells_notarget)

    # Perform simulations with raw parameters
    df_sim = simulate(df_sim, num_cores, model_function)

    # Calculate EC50 and Amplitude error to add weight
    if mode == "Fitting":
        for plot_num in list(dict.fromkeys(df_sim["Plot"])):
            df_plot = df_sim.loc[df_sim["Plot"] == plot_num]
            df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
            i = 0
            for variant in list(dict.fromkeys(df_plot["Variant"])):
                if i == 0:
                    df_var = df_plot.loc[df_plot["Variant"] == variant]
                    max_WT = df_var["Result"].max()
                    fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
                    df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                    df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
                else:
                    df_var = df_plot.loc[df_plot["Variant"] == variant]
                    fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
                    df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "Amp_m"] = fit[0]
                    df_EC50_AMP.loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).loc[pd.DataFrame(df_plot_EC50["Variant"] == variant).Variant, :].index[0], "EC50_m"] = np.log10(fit[1])
                i += 1
        
        # Use same cost function as parameter inference (weighted sum of squares)
        df_EC50_AMP["EC50_e"] = (df_EC50_AMP["EC50_t"] - df_EC50_AMP["EC50_m"])**2
        df_EC50_AMP["Amp_e"] = (df_EC50_AMP["Amp_t"] - df_EC50_AMP["Amp_m"])**2
        w_EC50 = df_EC50_AMP["EC50_e"].sum()*2
        w_AMP = df_EC50_AMP["Amp_e"].sum()*2

    elif mode == "Mutant_Optimization":
        EC50_target = []
        EC50_notarget = []
        Amp_target = []
        Amp_notarget = []
        cells = cells_target + cells_notarget 
        max_target = df_sim.loc[df_sim["Cell_type"]==cells_target[0],"Result"].max() 
        for cell in cells:
            df_cell = df_sim.loc[df_sim["Cell_type"]==cell]
            fit, cov = curve_fit(hill_func, df_cell["IL0"], df_cell["Result"].values*100/(max_target), bounds = ([0,df_cell["IL0"].min()/10], [df_cell["Result"].max()*100/(max_target)+10, df_cell["IL0"].max()*10]))
            if cell in cells_target:
                EC50_target.append(abs(np.log10(fit[1])))
                Amp_target.append(abs(fit[0]))
            else:
                EC50_notarget.append(abs(np.log10(fit[1])))
                Amp_notarget.append(abs(fit[0]))
                
        w_AMP = abs(2*(np.mean(Amp_notarget) - np.mean(Amp_target)))
        w_EC50 = abs(2*(np.mean(EC50_notarget) - np.mean(EC50_target)))
    
    print("EC50 mean distance: " + str(w_EC50/2))
    print("Amplitude mean distance: " + str(w_AMP/2))

    # Local slopiness algorithm begins
    if mode == "Fitting":
        N_param = len(param_mut)
        N_obs = len(df_EC50_AMP.index)
    elif mode == "Mutant_Optimization":
        N_param = len(param_mut)
        N_obs = len(cells_target)*len(cells_notarget)   
    delta = delta_0
    Jacobian = np.zeros([N_obs,N_param])
    k = 0
    while True:
        k = k + 1
        j = 0
        for param in param_mut:
            # Derive using central differences
            df_bind_up, df_bind_down = update_df_bind_up_down(df_bind, param, param_depen, delta)

            # Create dataset of simulations
            if mode == "Fitting":
                df_sim_up = set_simulation_fit(df_bind_up, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim)
                df_sim_down = set_simulation_fit(df_bind_down, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim)

            elif mode == "Mutant_Optimization":
                df_sim_up = set_simulation_mutant_optimization(df_bind_up, df_IC_data, df_linear_models, num_IL_sim, cells_target, cells_notarget)
                df_sim_down = set_simulation_mutant_optimization(df_bind_down, df_IC_data, df_linear_models, num_IL_sim, cells_target, cells_notarget)

            # Simulate
            df_res_up = simulate(df_sim_up, num_cores, model_function)
            df_res_down = simulate(df_sim_down, num_cores, model_function)

            if mode == "Fitting":
                # Get residuals
                df_EC50_AMP_up = calculate_cost_slopiness_fit(df_res_up, df_EC50_AMP.copy(), w_EC50, w_AMP)
                df_EC50_AMP_down = calculate_cost_slopiness_fit(df_res_down, df_EC50_AMP.copy(), w_EC50, w_AMP)

                # Derive the cost function using the central differences
                iJ = 0
                for i in df_EC50_AMP_up.index:
                    Jacobian[iJ,j] = (df_EC50_AMP_up.loc[i,"Error"] - df_EC50_AMP_down.loc[i,"Error"])/(delta) # Divide cost per number of samples to normalize
                    iJ += 1

            elif mode == "Mutant_Optimization":
                # Get residuals 
                df_EC50_AMP_up = calculate_cost_slopiness_target(df_res_up, cells_target, cells_notarget, w_EC50, w_AMP)
                df_EC50_AMP_down = calculate_cost_slopiness_target(df_res_down, cells_target, cells_notarget, w_EC50, w_AMP)

                # Derive the cost function using the central differences
                for i in range(0,N_obs):
                    Jacobian[i,j] = (df_EC50_AMP_up[i] - df_EC50_AMP_down[i])/(delta)     
            j += 1
            
        # Calculate the Hessian using the Gauss-Newton aproximation
        Hessian = np.zeros([N_param,N_param])
        for l in range(0,N_param):
            for m in range(0,N_param):
                for i in range(0,N_obs):
                    Hessian[l,m] = Hessian[l,m] + Jacobian[i,l]*Jacobian[i,m]


        # Check that there are no complex, NaNs or inf eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(Hessian)
        if not np.iscomplexobj(eigenvalues)  and not np.all(np.isinf(eigenvalues)) and not np.all(np.isnan(eigenvalues)):
            U, svd_eigenvalues, VT = np.linalg.svd(Hessian)

            # Check for numerical noise
            cond = np.where(abs(svd_eigenvalues/max(svd_eigenvalues)) <= np.sqrt(np.finfo(float).eps))
            svd_eigenvalues[cond] = np.zeros(len(cond))

            if np.all(svd_eigenvalues >= 0) and not np.iscomplexobj(svd_eigenvalues):
                break
            else:
                delta = delta_0*0.1 + round(np.random.uniform(0,1),1)*(delta_0*10-delta_0*0.1)

        if k >= 10:
            print('The Hessian matrix has negative eigenvalues!')
            break

    return Hessian

######################################################################################################################################################

def convert_to_ODE(model):
    # Get sympy symbols for all the model's species
    species_symbols = {}
    for i in range(0,len(model.species)):
        species_symbols["__s"+str(i)] = sp.Symbol("__s"+str(i))
        
    # Get the complexes that form the initial conditions of the model
    IC_list = []
    IC_molecule = {}
    IC_variables = {}
    for IC in model.initial_conditions:
        i = 0
        for species in model.species:
            if str(IC[0]) == str(species):
                IC_list.append("__s"+str(i))
                IC_variables["__s"+str(i)] = str(IC[0]).split("(")[0]
            i += 1
        IC_molecule[str(IC[0]).split("(")[0]] = []
    
    # Get sympy symbols for all the model's initial conditions
    IC_symbols = {}
    for IC in IC_molecule.keys():
        IC_symbols[IC] = sp.Symbol(IC+"0")
        
    # Get all variables that contain the molecule of the IC
    for molecule in IC_molecule.keys():
        i = 0
        for species in model.species:
            if molecule in str(species):
                IC_molecule[molecule].append("__s"+str(i))
            i += 1
    
    # Get the ODE system
    ODE_system_raw = [model.odes[i] for i in range(0,len(model.odes))]
    
    # Remove the ODEs that correspond to the species in the IC
    ODE_system = {}
    for i in range(0,len(ODE_system_raw)):
        if "__s"+str(i) not in IC_list:
            ODE_system["__s"+str(i)] = ODE_system_raw[i]
    
    # Apply the conservation law to remove the ICs' species from the ODEs
    for species_IC in IC_list:
        for species_ODE in ODE_system.keys():
            # Get the symbol from the initial condition
            expression = IC_symbols[IC_variables[species_IC]]
            # For all the species that contains the molecule found in the initial condition
            for species_var in IC_molecule[IC_variables[species_IC]]:
                # If species is not the one we need to remove we substract it to the IC value (conservation law)
                if species_var != species_IC:
                    expression -= species_symbols[species_var]
            # Substitute the species that contains the molecule found in the initial condition for the conservation law expression
            ODE_system[species_ODE] = ODE_system[species_ODE].subs(species_IC, expression)
            
    # Get sympy symbols for all the model's parameters
    param_symbols = {}
    # For all the parameters except the initial conditions
    for i in range(0,len(model.parameters)-len(IC_symbols.keys())):
        # Get only the parameter name and change the K from lower to upper case
        param_symbols[model.parameters[i]] = sp.Symbol('k'+str(model.parameters[i]).split("'")[1].split("K")[1])
        
    # Convert the parameters
    for param_old in param_symbols.keys():
        for species_ODE in ODE_system.keys():
            # Substitute the species that contains the molecule found in the initial condition for the conservation law expression
            ODE_system[species_ODE] = ODE_system[species_ODE].subs(param_old, param_symbols[param_old])
    return ODE_system, IC_symbols, param_symbols, IC_molecule

######################################################################################################################################################

def FP_search(PA_ODE):
    PA_ODE=PA_ODE[0]

    # Define the function here as the ony input can be the initial guess vector
    def ODE_solve_FP_search(x_guess):
        # Given an ODE system where only the ODE system variables are non-numerical and a value for them, calculate dx/dt
        ODE_system_copy = PA_ODE["Model"].values[0].copy()
        var_list = [sp.Symbol(symbol) for symbol in ODE_system_copy.keys()]
        result = []
        j=0
        for var in ODE_system_copy.keys():
            for i in range(0,len(x_guess)):
                ODE_system_copy[var] = ODE_system_copy[var].subs(var_list[i], x_guess[i])
            result.append(ODE_system_copy[var].evalf())
        return result

    # Find the FP
    PA_ODE["Result"] = [fsolve(ODE_solve_FP_search, PA_ODE["Guess"].values[0], xtol=1e-15)] # POT SER UN PARAMETRE!!!!!!!!!!
    
    return PA_ODE

######################################################################################################################################################

def FP_stability_analysis(ODE_system_raw, df_sim, param_type, num_cores, n_guess, IC_symbols, param_symbols, IC_molecule):
    # From given set of IC, estimate the value of all SS and determine if they are stable or unstable
    def ODE_solve(ODE_system_copy, x_guess):
        # Given an ODE system where only the ODE system variables are non-numerical and a value for them, calculate dx/dt
        ODE_system_copy = ODE_system_copy.copy()
        var_list = [sp.Symbol(symbol) for symbol in ODE_system_copy.keys()]
        result = []
        j=0
        for var in ODE_system_copy.keys():
            for i in range(0,len(x_guess)):
                ODE_system_copy[var] = ODE_system_copy[var].subs(var_list[i], x_guess[i])
            result.append(ODE_system_copy[var].evalf())
        return result
    
    # Get extra data
    Na=6.023e23
    Width_PM = 1e-7 # In dm
    x_n = len(ODE_system_raw.keys()) # Number of variables in the model
    
    # Reverse the variable IC_molecule
    IC_molecule_rev = {}
    for var in ODE_system_raw.keys():
        vec_var = []
        for IC in IC_molecule.keys():
            if var in IC_molecule[IC]:
                vec_var.append(IC+"0")
        IC_molecule_rev[var] = vec_var
    
    # Create a dataset of FP search, for each parameter + IC there is one row for each inital guess
    df_sim = df_sim.drop(["Tf","dT","Result"], axis=1)
    df_sim_FP = pd.DataFrame(columns=list(df_sim.columns)+['IC/param combination', 'Guess', 'Model', 'Result'])
    
    j = 0
    for row in df_sim.index:
        # Each row will have a different set of parameters/IC
        PA = df_sim.loc[[row]]
        ODE_system = ODE_system_raw.copy()

        # Put values for parameters and initial conditions of the ODEs
        IC_param_string = {str(symbol):symbol for symbol in IC_symbols.values()} | {str(symbol):symbol for symbol in param_symbols.values()}
        for param_IC in IC_param_string.keys():     
            for species_ODE in ODE_system.keys():
                if param_IC in param_type["first_binding"]:
                    ODE_system[species_ODE] = ODE_system[species_ODE].subs(IC_param_string[param_IC], PA[param_IC].values[0]/(Na*PA["vol_EC"].values[0]))
                if param_IC in param_type["non_first_binding"]:
                    if "2" in param_IC:
                        ODE_system[species_ODE] = ODE_system[species_ODE].subs(IC_param_string[param_IC], PA[param_IC.replace("2","")].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
                    else:
                        ODE_system[species_ODE] = ODE_system[species_ODE].subs(IC_param_string[param_IC], PA[param_IC].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
                if param_IC in param_type["no_normalization"]:
                    if "2" in param_IC:
                        ODE_system[species_ODE] = ODE_system[species_ODE].subs(IC_param_string[param_IC], PA[param_IC.replace("2","")].values[0])
                    else:
                        ODE_system[species_ODE] = ODE_system[species_ODE].subs(IC_param_string[param_IC], PA[param_IC].values[0])
                if param_IC in param_type["IC_concentration"]:
                    ODE_system[species_ODE] = ODE_system[species_ODE].subs(IC_param_string[param_IC], PA[param_IC].values[0]*Na*PA["vol_EC"].values[0])
                if param_IC in param_type["IC_linear_model"]:
                    ODE_system[species_ODE] = ODE_system[species_ODE].subs(IC_param_string[param_IC], 10**(PA[param_IC].values[0]))

        # Generate the initial guesses
        # We want to make guesses only within values that are possible i.e. < Initial conditions
        # Generate for each IC maximum value a vector of possible inital vectors
        vec_init_raw = {}
        for IC in IC_molecule.keys():
            if IC+"0" in param_type["IC_concentration"]:
                IC_val = PA[IC+"0"].values[0]*Na*PA["vol_EC"].values[0]
            if IC+"0" in param_type["IC_linear_model"]:
                IC_val = 10**(PA[IC+"0"].values[0])
            vec_init_raw[IC+"0"] = np.transpose(np.array(list(dict.fromkeys(list(combinations_with_replacement([j for i in [list(np.linspace(0,IC_val,n_guess)) for i in range(0,x_n)] for j in i],x_n))))))

        # For each variable of the model, we select which is the limiting IC molecule to include it in the vector of possible inital vectors
        i = 0
        # For all the variables of the model
        IC_limit = []
        for key in IC_molecule_rev.keys():
            # For all the inital conditions in a variable we save which is the one limiting
            IC = IC_molecule_rev[key][0]
            if IC in param_type["IC_concentration"]:
                IC_val_saved = PA[IC].values[0]*Na*PA["vol_EC"].values[0]
            if IC in param_type["IC_linear_model"]:
                IC_val_saved = 10**(PA[IC].values[0])
            IC_saved = IC
            for IC in IC_molecule_rev[key][1:]:
                if IC in param_type["IC_concentration"]:
                    IC_val = PA[IC].values[0]*Na*PA["vol_EC"].values[0]
                if IC in param_type["IC_linear_model"]:
                    IC_val = 10**(PA[IC].values[0])
                if IC_val < IC_val_saved:
                    IC_saved = IC
                    IC_val_saved = IC_val
            IC_limit.append(IC_saved)

        # Construct the vector of inital guesses by taking values from limiting variables
        vec_init = []
        i = 0
        for IC in IC_limit:
            vec_init.append(vec_init_raw[IC][i])
            i += 1
        vec_init = np.transpose(np.array(vec_init))


        # Build the dataset by including the ODE_system with all numerical parameters/IC and the initial guesses for FP search
        for vec in vec_init:
            df_sim_FP.loc[len(df_sim_FP.index)] = df_sim.loc[row].tolist() + [j , vec, ODE_system, vec]
        j += 1
        
    # Get the FP of the system (dx/dt=0)
    df_res = simulate(df_sim_FP, num_cores, FP_search)
    
    # After the calculation of the FP, we determine the stability of them by calculating the gradient
    df_stab = pd.DataFrame(columns=list(df_res.columns[:-4])+['IC/param combination', 'Fixed points', 'Eigenvalues', 'Eigenvectors', 'Stability'])

    # Convert ODE system variables from char to sympy symbol
    species_symbols = {}
    for symbol in ODE_system.keys():
        species_symbols[symbol] = sp.Symbol(symbol)

    for IC_param in np.unique(df_res["IC/param combination"].values):
        # Retrieve data from one IC/parameter combination point
        df_point = df_res.loc[df_res["IC/param combination"] == IC_param]
        ODE_system = df_point.loc[df_point.index[0],"Model"]
        FP_vec_old = df_point["Result"].tolist()
        
        # Remove solutions that contain nan or inf
        FP_vec = []
        for vec in FP_vec_old:
            if True not in np.isnan(vec):
                if True not in np.isinf(vec):
                    FP_vec.append(vec)
            
        # We round the numbers so that we can then eliminate the repeated FP
        mag_order_smallest = np.log10(np.min([np.min([num for num in vec if num > 0]) for vec in np.absolute(FP_vec)]))
        if mag_order_smallest >= 0:
            FP_vec = np.around(FP_vec, 3)
        else:
            FP_vec = np.around(FP_vec, int(abs(mag_order_smallest))+2)

        # Eliminate repeated steady states
        FP_vec = np.unique(FP_vec, axis=0)

        # Get only solutions that are true dx/dt=0. We put the limit to mean dx/dt < 1e-4 -> Gives aproximate solution
        FP_vec_fin = []
        for vec in FP_vec:
            if np.mean(np.absolute(ODE_solve(ODE_system, vec))) < 1e-3: # POT SER UN PARAMETRE!!!!!!!!!!
                FP_vec_fin.append(vec)

        # Calculate the gradient
        # Derive all the ODEs with respect to all the variables
        Gradient = {}
        for symbol1 in species_symbols.keys():
            Gradient[symbol1] = {}
            for symbol2 in species_symbols.keys():
                Gradient[symbol1][symbol2] = sp.diff(ODE_system[symbol1], species_symbols[symbol2])

        # Solve the gradient for each of the FP and calculate the eigenvalues to determine the stability
        Stability = []
        Eigenvalues = []
        Eigenvectors = []

        for FP in FP_vec_fin:
            # For each element of the gradient we substitute each of the variables at the FP
            Gradient_FP = np.zeros([len(species_symbols.keys()),len(species_symbols.keys())])
            i = 0
            for symbol1 in species_symbols.keys():
                j = 0
                for symbol2 in species_symbols.keys():
                    Curr_der = Gradient[symbol1][symbol2].copy()
                    k = 0
                    for symbol3 in species_symbols.keys():
                        Curr_der = Curr_der.subs(species_symbols[symbol3], FP[k])
                        k += 1
                    Gradient_FP[i,j] = Curr_der
                    j += 1
                i += 1

            # Calculate the eigenvalues and eigenvectors of the FP to assess its stability
            eigenval, eigenvec = np.linalg.eig(Gradient_FP)
            Eigenvalues.append(eigenval)
            Eigenvectors.append(eigenvec)

            # Determine the stability of the FP
            if np.all(eigenval < 0):
                Stability.append("Stable FP")
            elif np.all(eigenval > 0):
                Stability.append("Unstable FP")
            else:
                Stability.append("Saddle FP")

        df_stab.loc[len(df_stab.index)] = df_point.loc[df_point.index[0]].tolist()[:-4] + [IC_param , FP_vec_fin, Eigenvalues, Eigenvectors, Stability]
    
    return df_stab

######################################################################################################################################################    
def simulate_sequential(df_sim_data, model_func):
    out_vec = []
    for i in range(0,len(df_sim_data.index)):
        param_parallel = [df_sim_data.loc[[i]]]
        out_vec.append(model_func(param_parallel))
    return pd.concat(out_vec, axis=0)

######################################################################################################################################################

def set_simulation_ABC(df_bind, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim, variant_list):
    cell__variant_list = []
    df_sim = pd.DataFrame(columns=["Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            cell__variant = df_plot["Cell_type"].values[0]+"__"+variant
            if cell__variant not in cell__variant_list:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                df_var_EC50 = df_plot_EC50.loc[df_plot_EC50["Variant"] == variant]
                IL_vec =  10**(np.linspace(df_var_EC50["EC50_t"].values[0]-3, df_var_EC50["EC50_t"].values[0]+4, num_IL_sim))
                if variant in variant_list:
                    for i in range(0,num_IL_sim):
                        IL0 = IL_vec[i]
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
                        df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
                else: # If variant is not fitted will be WT, only get maximum reponse to then normalize
                    IL0 = IL_vec[-1]
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
                    df_sim.loc[len(df_sim.index)] = df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind.loc[df_bind["Variant"]==variant].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
                cell__variant_list.append(cell__variant)
    return df_sim

######################################################################################################################################################

# Update the dataset of the binding rates
def update_df_bind_ABC(df_bind, param_dict, param_depen, param_unique_var, variants_list):
    # Create copy of binding dataframe
    df_bind_fit = df_bind.copy()

    # Update df_bind with new parameter values
    i=0
    for variant in variants_list:
        if variant == "WT":
            for param in [key.split('__')[1] for key in param_dict.keys() if key.split('__')[0] == variant]:
                # Copy new parameter to dataframe
                df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param] = 10**param_dict[variant+"__"+param]
                # Parameter dependancies
                if param in param_depen.keys():
                    for param_d in param_depen[param]:
                        param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                        param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                        df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param_d] = (10**param_dict[variant+"__"+param])/param1_0*param2_0
                # Parameter which are the same for WT and other variants must be also changed in the variant parameter row
                for variant_WT_mut in variants_list[1:]: # For all the non-WT variants
                    if param not in param_unique_var[variant_WT_mut]:
                        param_dep_bool = True # If parameter is not unique to the variant, then it must be changed following the WT
                        for param2 in param_unique_var[variant_WT_mut]:
                            if param2 in param_depen.keys(): 
                                if param == param_depen[param2]:
                                    param_dep_bool = False # If parameter is not unique to the variant, but is dependant to a unique one, then it must be changed following the variant parameter
                        # Change the variant non-unqiue parameter and the ones that depend on it, following the WT one
                        if param_dep_bool:
                            df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).Variant, :].index[0], param] = 10**param_dict[variant+"__"+param]
                            # Parameter dependancies
                            if param in param_depen.keys():
                                for param_d in param_depen[param]:
                                    param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                                    param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                                    df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).Variant, :].index[0], param_d] = (10**param_dict[variant+"__"+param])/param1_0*param2_0
        # If parameter is not specified as WT, then its unique to the variant -> change only parameter and the ones that depend on it
        else:
            for param in [key.split('__')[1] for key in param_dict.keys() if key.split('__')[0] == variant]:
                df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param] = 10**param_dict[variant+"__"+param]
                # Parameter dependancies
                if param in param_depen.keys():
                    for param_d in param_depen[param]:
                        param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                        param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                        df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param_d] = (10**param_dict[variant+"__"+param])/param1_0*param2_0
    return df_bind_fit

######################################################################################################################################################

def param_inference_ABC_SMC(model_function, df_fit_data, df_bind, df_EC50_AMP, df_IC_data, df_linear_models, df_sim_data, param_depen, param_unique_var, num_IL_sim, num_cores_ABC_SMC, ABC_SMC_path, conf95, pop_size_init, pop_size_max, E_cv, min_epsilon, max_num_iter, WT_norm):

    # From EC50_AMP_model and EC50_AMP_exp
    def cost_ABC(EC50_AMPm, EC50_AMP0):
        # Get wighted sum of squares (Euclidian distance)    
        e_AMP = 0
        e_EC50 = 0
        for key in EC50_AMP0.keys():
            e_AMP += (EC50_AMP0[key][0] - EC50_AMPm[key][0])**2
            e_EC50 += (EC50_AMP0[key][1] - EC50_AMPm[key][1])**2
        e_AMP = e_AMP/w_AMP
        e_EC50 = e_EC50/w_EC50
        if WT_norm: # == True
            return e_EC50 + e_AMP
        else:
            return e_EC50*2 # If amplitude of variant is normalized to 100 (WT normally), only consider EC50 for cost
    
    # From parameter_dict, simulate the model for all cells/variants we need
    def model(parameter_dict):
        # If sampled parameters are outside of the bounds do not simulate and output the simulation with original parameter values (high cost -> individual is not selected)
        if False in [(np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==var__param,"Lower_bound"]).values[0] < parameter_dict[var__param]) & (np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==var__param,"Upper_bound"]) > parameter_dict[var__param]).values[0] for var__param in parameter_dict.keys()]:
            EC50_AMP = EC50_AMP_prior.copy()
            
        # If the parameter values are inside the bounds simulate the model
        else:
            # Update df_bind
            df_bind_fit = update_df_bind_ABC(df_bind, parameter_dict, param_depen, param_unique_var, variants_fit)

            # Create dataset of simulations
            df_sim = set_simulation_ABC(df_bind_fit, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim, variants_fit)

            # Simulate
            df_sim = simulate_sequential(df_sim, model_function)

            # Add missing simulation data on certain plots where the pair of cell__variant is found more than one time but it can be simualted oly once
            for plot_num_copy in cell__variant_missing_in_plot.keys():
                for plot_num_paste in cell__variant_missing_in_plot[plot_num_copy]:
                    df_copy = df_sim.loc[df_sim["Plot"]==plot_num_copy].loc[df_sim.loc[df_sim["Plot"]==plot_num_copy]["Variant"]=="WT"]
                    df_copy['Plot'] = plot_num_paste
                    df_sim = pd.concat([df_sim, df_copy], ignore_index=True)

            # Calculate EC50 and Amplitude error to add weight
            EC50_AMP = {key: [] for key in df_EC50_AMP["Plot"].astype(str)+"__"+df_EC50_AMP["Variant"] if key.split("__")[1] in variants_fit}
            
            plot_num_old = 0
            # For each celltype/variant simulated
            for key in EC50_AMP.keys():
                plot_num, variant = key.split("__")
                plot_num = int(plot_num)
                df_var = df_sim.loc[df_sim["Plot"] == plot_num].loc[df_sim.loc[df_sim["Plot"] == plot_num]["Variant"] == variant]
                # Data is normalized to the first variant that appears in a plot
                if plot_num_old != plot_num:
                    if WT_norm: # == True
                        max_WT = df_sim.loc[df_sim["Plot"] == plot_num].loc[df_sim.loc[df_sim["Plot"] == plot_num]["Variant"] == "WT"]["Result"].max()
                    else:
                        max_WT = df_var["Result"].max()
                    fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
                    EC50_AMP[key].append(fit[0])
                    EC50_AMP[key].append(np.log10(fit[1]))
                else:
                    fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
                    EC50_AMP[key].append(fit[0])
                    EC50_AMP[key].append(np.log10(fit[1]))
                plot_num_old = plot_num
        return EC50_AMP
    
    # Save parameters to fit in a way that can be used by pyABC
    param_fit = list(df_fit_data["Var__Param"]) # List of parameters to fit(variant__parameter)
    variants_fit = list(dict.fromkeys([param.split("__")[0] for param in param_fit])) # List of variants to fit

    # Get the EC50 and Amplitude of data to fit in a dictionary (pyABC required)
    EC50_AMP_0 = {key: [] for key in df_EC50_AMP["Plot"].astype(str)+"__"+df_EC50_AMP["Variant"] if key.split("__")[1] in variants_fit}
    for key in EC50_AMP_0.keys():
        plot_num, variant = key.split("__")
        plot_num = int(plot_num)
        EC50_AMP_0[key].append(df_EC50_AMP.loc[df_EC50_AMP["Plot"]==plot_num].loc[df_EC50_AMP.loc[df_EC50_AMP["Plot"]==plot_num]["Variant"]==variant,"Amp_t"].values[0])
        EC50_AMP_0[key].append(df_EC50_AMP.loc[df_EC50_AMP["Plot"]==plot_num].loc[df_EC50_AMP.loc[df_EC50_AMP["Plot"]==plot_num]["Variant"]==variant,"EC50_t"].values[0])

    
    # Generate the priors
    prior_dict = {}
    for param in param_fit:
        prior_dict[param] = pyabc.RV("norm", loc=np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==param,"Initial_val"].values[0]), scale=conf95/4)
    prior = pyabc.Distribution(**prior_dict)
    
    
    # Get plot numbers of repeated pairs of cell__variant. Keys on dictionary are the plot numbers where WT data has to be copied from. Values on dictionary are the plot numbers where WT data has to be copied to.
    cell__variant_missing_in_plot = {df_sim_data.loc[df_sim_data["Cell_type"]==cell__variant.split("__")[0],"Plot"].drop_duplicates().tolist()[0]:df_sim_data.loc[df_sim_data["Cell_type"]==cell__variant.split("__")[0],"Plot"].drop_duplicates().tolist()[1:] for cell__variant in df_sim_data.loc[df_sim_data["Variant"]=="WT"]["Cell_type"].drop_duplicates().values+"__WT" if len(df_sim_data.loc[df_sim_data["Cell_type"]==cell__variant.split("__")[0],"Plot"].drop_duplicates().tolist()) > 1}
    
    # Simulate at the peak of the prior distribution to get cost function weights
    start = time.time()
    dict_param_fit_init = {key: np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==key,"Initial_val"].values[0]) for key in param_fit}
    df_bind_init = update_df_bind_ABC(df_bind, dict_param_fit_init, param_depen, param_unique_var, variants_fit)

    # Create dataset of simulations
    df_sim = set_simulation_ABC(df_bind_init, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim, variants_fit)

    # Simulate
    df_sim = simulate_sequential(df_sim, model_function)
    print("Number of simulations: "+str(len(df_sim.index)))
    
    # Add missing simulation data on certain plots where the pair of cell__variant is found more than one time but it can be simualted only once
    for plot_num_copy in cell__variant_missing_in_plot.keys():
        for plot_num_paste in cell__variant_missing_in_plot[plot_num_copy]:
            df_copy = df_sim.loc[df_sim["Plot"]==plot_num_copy].loc[df_sim.loc[df_sim["Plot"]==plot_num_copy]["Variant"]=="WT"]
            df_copy['Plot'] = plot_num_paste
            df_sim = pd.concat([df_sim, df_copy], ignore_index=True)
    print("Number of ICs: "+str(len(df_sim.index)))

    # Calculate EC50 and Amplitude error to add weight
    EC50_AMP = {key: [] for key in df_EC50_AMP["Plot"].astype(str)+"__"+df_EC50_AMP["Variant"] if key.split("__")[1] in variants_fit}

    plot_num_old = 0
    # For each celltype/variant simulated
    for key in EC50_AMP.keys():
        plot_num, variant = key.split("__")
        plot_num = int(plot_num)
        df_var = df_sim.loc[df_sim["Plot"] == plot_num].loc[df_sim.loc[df_sim["Plot"] == plot_num]["Variant"] == variant]
        
        # Data is normalized to the first variant that appears in a plot
        if plot_num_old != plot_num:
            if WT_norm: # == True
                max_WT = df_sim.loc[df_sim["Plot"] == plot_num].loc[df_sim.loc[df_sim["Plot"] == plot_num]["Variant"] == "WT"]["Result"].max()
            else:
                max_WT = df_var["Result"].max()
            fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
            EC50_AMP[key].append(fit[0])
            EC50_AMP[key].append(np.log10(fit[1]))
        else:
            fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
            EC50_AMP[key].append(fit[0])
            EC50_AMP[key].append(np.log10(fit[1]))
        plot_num_old = plot_num

    # Get weights for the sum of squares (Weighted euclidian distance with 0 as minimum and 1 as maximum)    
    w_AMP = 0
    w_EC50 = 0
    for key in EC50_AMP_0.keys():
        w_AMP += (EC50_AMP_0[key][0] - EC50_AMP[key][0])**2
        w_EC50 += (EC50_AMP_0[key][1] - EC50_AMP[key][1])**2
    w_AMP = w_AMP*2
    w_EC50 = w_EC50*2
    
    # Save prior EC50_AMP so it can be used to create the bounds
    EC50_AMP_prior = EC50_AMP.copy()
    
    print("Elapsed time in seconds per full model simulation: " + str(time.time() - start))
    print("EC50 mean error: " + str((w_EC50/len(EC50_AMP_0.keys()))**(1/2)))
    print("Amplitude mean error: " + str((w_AMP/len(EC50_AMP_0.keys()))**(1/2)))
    print("Inital cost: " + str(cost_ABC(EC50_AMP, EC50_AMP_0)))
    
    # Execute ABC-SMC parameter inference
    # abc = pyabc.ABCSMC(model, prior, cost_ABC, sampler=pyabc.sampler.MappingSampler(map_=pyabc.sge.SGE(tmp_directory="data/ABC_SMC/tmp", memory='5G', time_h=1, name='pyabc_inf_sim', queue='short-centos79,long-centos79', num_threads=1).map(model, prior), mapper_pickles=True), population_size=pyabc.populationstrategy.AdaptivePopulationSize(start_nr_particles=pop_size_init, mean_cv=E_cv, max_population_size=pop_size_max))
    
    # To update epsilon, it uses the median distance from last iteration, other epsilon updating strategies can be selected: eps=pyabc.epsilon...
    # pyabc.sampler.MulticoreParticleParallelSampler gets stuck in a strange single core execution
    abc = pyabc.ABCSMC(model, prior, cost_ABC, sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=num_cores_ABC_SMC), population_size=pyabc.populationstrategy.AdaptivePopulationSize(start_nr_particles=pop_size_init, mean_cv=E_cv, max_population_size=pop_size_max), )
    
    # Get path to save ABC-SMC runs
    abc.new("sqlite:///" + ABC_SMC_path, EC50_AMP_0)
    
    # Execute ABC-SMC parameter inference
    abc.run(minimum_epsilon=min_epsilon, max_nr_populations=max_num_iter)
        
######################################################################################################################################################

def mutant_search_ABC_SMC(model_function, df_bind, df_IC_data, df_linear_models, param_fit, cells_target, cells_notarget, param_depen, param_unique_var, num_IL_sim, num_cores_ABC_SMC, ABC_SMC_path, conf95, pop_size_init, pop_size_max, E_cv, min_epsilon, max_num_iter):

    # From EC50_AMP_model and EC50_AMP_exp
    def cost_ABC(EC50_AMPm, EC50_AMP0):
        AMP_target = np.mean([EC50_AMPm[cell][0] for cell in cells_target])
        AMP_notarget = np.mean([EC50_AMPm[cell][0] for cell in cells_notarget])
        EC50_target = abs(np.mean([EC50_AMPm[cell][1] for cell in cells_target]))
        EC50_notarget = abs(np.mean([EC50_AMPm[cell][1] for cell in cells_notarget]))
        return 10**((AMP_notarget - AMP_target)/w_AMP+(EC50_notarget - EC50_target)/w_EC50-1) # Cost is bounded from 1 to 0
    
    # From parameter_dict, simulate the model for all cells/variants we need
    def model(parameter_dict):
        # If sampled parameters are outside of the bounds do not simulate and output the simulation with original parameter values (high cost -> individual is not selected)
        if False in [(np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==var__param,"Lower_bound"]).values[0] < parameter_dict[var__param]) & (np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==var__param,"Upper_bound"]) > parameter_dict[var__param]).values[0] for var__param in parameter_dict.keys()]:
            EC50_AMP = EC50_AMP_prior.copy()
            
        # If the parameter values are inside the bounds simulate the model
        else:
            # Update df_bind
            df_bind_fit = update_df_bind_ABC(df_bind, parameter_dict, param_depen, param_unique_var, variants_fit)

            # Create dataset of simulations
            df_sim = set_simulation_mutant_optimization(df_bind_fit, df_IC_data, df_linear_models, num_IL_sim, cells_target, cells_notarget)

            # Simulate
            df_sim = simulate_sequential(df_sim, model_function)

            # Calculate EC50 and Amplitude error to add weight
            EC50_AMP = {key: [] for key in cells_fit}
            
            # For each cell type simulated
            max_target = df_sim.loc[df_sim["Cell_type"].isin(cells_target),"Result"].max()
            for cell in EC50_AMP.keys():
                df_cell = df_sim.loc[df_sim["Cell_type"] == cell]
                fit, cov = curve_fit(hill_func, df_cell["IL0"], df_cell["Result"].values*100/(max_target), bounds = ([0,df_cell["IL0"].min()/10], [df_cell["Result"].max()*100/(max_target)+10, df_cell["IL0"].max()*10]))
                EC50_AMP[cell].append(fit[0])
                EC50_AMP[cell].append(np.log10(fit[1]))
            
        return EC50_AMP
    
    # Save parameters to fit in a way that can be used by pyABC
    df_bind = df_bind.loc[df_bind["Variant"]=="WT"]
    df_bind.loc[df_bind["Variant"]=="WT","Variant"] = "Mut"
    variants_fit = ["Mut"] # List of variants to fit
    cells_fit = cells_target + cells_notarget
    
    # Save a dataset with bounds on the parameters to be infered
    df_fit_data = pd.DataFrame()
    df_fit_data["Initial_val"] = [df_bind.loc[df_bind["Variant"]=="Mut",param].values[0] for param in param_fit]
    df_fit_data["Lower_bound"] = [df_bind.loc[df_bind["Variant"]=="Mut",param].values[0]/50 for param in param_fit]
    df_fit_data["Upper_bound"] = [df_bind.loc[df_bind["Variant"]=="Mut",param].values[0]*50 for param in param_fit]
    param_fit = ["Mut__"+param for param in param_fit]
    df_fit_data["Var__Param"] = param_fit

    # Get the EC50 and Amplitude of data to fit in a dictionary (pyABC required, not needed fro mutant search)
    EC50_AMP_0 = {key: [] for key in cells_fit}
    
    # Generate the priors
    #prior_dict = {}
    #for var__param in param_fit:
    #    var,param = var__param.split("__")
    #    prior_dict[var__param] = pyabc.RV("norm", loc=np.log10(df_bind.loc[df_bind["Variant"]=="Mut",param].values[0]), scale=conf95/4)
    #prior = pyabc.Distribution(**prior_dict)
    
    # Generate the priors
    prior_dict = {}
    for var__param in param_fit:
        var,param = var__param.split("__")
        prior_dict[var__param] = pyabc.RV("uniform", np.log10(df_bind.loc[df_bind["Variant"]=="Mut",param].values[0])-conf95, conf95*2)
    prior = pyabc.Distribution(**prior_dict)
    
    # Simulate at the peak of the prior distribution to get cost function weights
    start = time.time()
    dict_param_fit_init = {key: np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==key,"Initial_val"].values[0]) for key in param_fit}
    df_bind_init = update_df_bind_ABC(df_bind, dict_param_fit_init, param_depen, param_unique_var, variants_fit)

    # Create dataset of simulations
    df_sim = set_simulation_mutant_optimization(df_bind, df_IC_data, df_linear_models, num_IL_sim, cells_target, cells_notarget)

    # Simulate
    df_sim = simulate_sequential(df_sim, model_function)
    print("Number of simulations: "+str(len(df_sim.index)))
    print("Number of ICs: "+str(len(df_sim.index)))

    # Calculate EC50 and Amplitude error to add weight
    EC50_AMP = {key: [] for key in cells_fit}

    # For each cell type simulated
    max_target = df_sim.loc[df_sim["Cell_type"].isin(cells_target),"Result"].max()
    for cell in EC50_AMP.keys():
        df_cell = df_sim.loc[df_sim["Cell_type"] == cell]
        fit, cov = curve_fit(hill_func, df_cell["IL0"], df_cell["Result"].values*100/(max_target), bounds = ([0,df_cell["IL0"].min()/10], [df_cell["Result"].max()*100/(max_target)+10, df_cell["IL0"].max()*10]))
        EC50_AMP[cell].append(fit[0])
        EC50_AMP[cell].append(np.log10(fit[1]))

    # Get weights for the sum of squares (Weighted euclidian distance with 0 as minimum and 1 as maximum)    
    AMP_target = np.mean([EC50_AMP[cell][0] for cell in cells_target])
    AMP_notarget = np.mean([EC50_AMP[cell][0] for cell in cells_notarget])
    EC50_target = abs(np.mean([EC50_AMP[cell][1] for cell in cells_target]))
    EC50_notarget = abs(np.mean([EC50_AMP[cell][1] for cell in cells_notarget]))
    w_AMP = abs(2*(AMP_notarget - AMP_target))
    w_EC50 = abs(2*(EC50_notarget - EC50_target))
    
    # Save prior EC50_AMP so it can be used to create the bounds
    EC50_AMP_prior = EC50_AMP.copy()
    
    print("Elapsed time in seconds per full model simulation: " + str(time.time() - start))
    print("EC50 distance: " + str(EC50_notarget - EC50_target))
    print("Amplitude distance: " + str(AMP_notarget - AMP_target))
    print("Inital cost: " + str(cost_ABC(EC50_AMP, EC50_AMP_0)))
    
    # Execute ABC-SMC parameter inference
    # abc = pyabc.ABCSMC(model, prior, cost_ABC, sampler=pyabc.sampler.MappingSampler(map_=pyabc.sge.SGE(tmp_directory="data/ABC_SMC/tmp", memory='5G', time_h=1, name='pyabc_inf_sim', queue='short-centos79,long-centos79', num_threads=1).map(model, prior), mapper_pickles=True), population_size=pyabc.populationstrategy.AdaptivePopulationSize(start_nr_particles=pop_size_init, mean_cv=E_cv, max_population_size=pop_size_max))
    
    # To update epsilon, it uses the median distance from last iteration, other epsilon updating strategies can be selected: eps=pyabc.epsilon...
    # pyabc.sampler.MulticoreParticleParallelSampler gets stuck in a strange single core execution
    abc = pyabc.ABCSMC(model, prior, cost_ABC, sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=num_cores_ABC_SMC), population_size=pyabc.populationstrategy.AdaptivePopulationSize(start_nr_particles=pop_size_init, mean_cv=E_cv, max_population_size=pop_size_max), )
    
    # Get path to save ABC-SMC runs
    abc.new("sqlite:///" + ABC_SMC_path, EC50_AMP_0)
    
    # Execute ABC-SMC parameter inference
    abc.run(minimum_epsilon=min_epsilon, max_nr_populations=max_num_iter)
        
######################################################################################################################################################

def sample_prior_posterior(model_function, df_fit_data, df_bind, df_EC50_AMP, df_IC_data, df_linear_models, df_sim_data, param_depen, param_unique_var, num_IL_sim, num_samples, conf95, variants_list, path_data_ABC):
    # Sample from prior and posterior distributions
    df_prior = pd.DataFrame()
    df_post = pd.DataFrame()
    for variant in variants_list:
        # Extract which are the parameters that were fitted and info to compute prior distribution
        history = pyabc.History("sqlite:///"+path_data_ABC+variant+".db")
        df_fit_data_copy = df_fit_data.copy() # Parameters to fit
        df_fit_data_copy["Var__Param"] = df_fit_data_copy["Variant"]+"__"+df_fit_data_copy["Parameter"]
        df_fit_data_copy = df_fit_data_copy[["Var__Param","Initial_val","Lower_bound","Upper_bound"]]
        df_fit_data_copy = df_fit_data_copy.loc[df_fit_data_copy["Var__Param"].str.contains(variant)]

        # Sample from prior and posterior distributions using the inverse transform sampling
        for var__param in df_fit_data_copy["Var__Param"]:
            var, param = var__param.split("__") 
            # Get prior
            x = np.linspace(np.log10(df_bind.loc[df_bind["Variant"]==var,param].values[0])-2,np.log10(df_bind.loc[df_bind["Variant"]==var,param].values[0])+2,10000)
            pdf = pyabc.RV("norm", loc=np.log10(df_fit_data_copy.loc[df_fit_data_copy["Var__Param"]==var__param,"Initial_val"].values[0]), scale=conf95/2).pdf(x)

            dx = np.diff(x)[0]  # Evenly spaced x values
            pdf /= np.sum(pdf) # Normalize so sum is 1
            cdf = np.cumsum(pdf) # Compute the CDF

            inverse_cdf = interp1d(cdf, x, bounds_error=False, fill_value=(x[0], x[-1])) # Get inverse of the CDF
            df_prior[var__param] = 10**inverse_cdf(np.random.uniform(0, 1, num_samples)) # Sample from the inverse CDF

            # Get posterior
            df, w = history.get_distribution(m=0, t=history.max_t)
            x, pdf = pyabc.visualization.kde.kde_1d(df, w, x=var__param,xmin = np.log10(df_bind.loc[df_bind["Variant"]==var,param].values[0])-2, xmax = np.log10(df_bind.loc[df_bind["Variant"]==var,param].values[0])+2, numx=10000)

            dx = np.diff(x)[0]  # Evenly spaced x values
            pdf /= np.sum(pdf) # Normalize so sum is 1
            cdf = np.cumsum(pdf) # Compute the CDF

            inverse_cdf = interp1d(cdf, x, bounds_error=False, fill_value=(x[0], x[-1])) # Get inverse of the CDF
            df_post[var__param] = 10**inverse_cdf(np.random.uniform(0, 1, num_samples)) # Sample from the inverse CDF

    # For each parameter set generate a df_bind, save it a list and prepare the simulations
    # Modify the df_binds and save them in a list
    df_bind_prior_vec = []
    df_bind_post_vec = []
    for j in range(0,len(df_prior)):
        df_bind_prior = df_bind.copy()
        df_bind_post = df_bind.copy()
        # Generate a df_bind for each row of prior and posterior distributions sampled values
        for var__param in df_prior.columns:
            var, param = var__param.split("__")
            df_bind_prior.loc[df_bind_prior["Variant"]==var,param] = df_prior[var__param].values[j] # Get sampled parameter values of prior distributions from row i
            df_bind_post.loc[df_bind_post["Variant"]==var,param] = df_post[var__param].values[j] # Get sampled parameter values of posterior distributions from row i
        df_bind_prior_vec.append(df_bind_prior)
        df_bind_post_vec.append(df_bind_post)

    # Set up simulations    
    df_sim_prior = pd.DataFrame(columns=["Num. sample","Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    df_sim_post = pd.DataFrame(columns=["Num. sample","Plot","Cell_type","Variant","STAT_type"]+df_bind.columns.tolist()[1:]+["IL0","RA0","RB0","RG0","STAT50","Tf","dT","Result"])
    # Generate a dataset of simulations for sampled values of prior and posterior distributions 
    for plot_num in list(dict.fromkeys(df_sim_data["Plot"])):
        df_plot = df_sim_data.loc[df_sim_data["Plot"] == plot_num]
        df_plot_EC50 = df_EC50_AMP.loc[df_EC50_AMP["Plot"] == plot_num]
        df_cell = df_IC_data.loc[df_IC_data["Cell"]==df_plot.loc[df_plot.index[0]][1]]
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            df_var = df_plot.loc[df_plot["Variant"] == variant]
            df_var_EC50 = df_plot_EC50.loc[df_plot_EC50["Variant"] == variant]
            IL_vec =  10**(np.linspace(df_var_EC50["EC50_t"].values[0]-3, df_var_EC50["EC50_t"].values[0]+4, num_IL_sim))
            for i in range(0,num_IL_sim):
                IL0 = IL_vec[i]
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
                for j in range(0,len(df_prior)):
                    df_sim_prior.loc[len(df_sim_prior.index)] = [j] + df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind_prior_vec[j].loc[df_bind_prior_vec[j]["Variant"]==variant].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
                    df_sim_post.loc[len(df_sim_post.index)] = [j] + df_var.loc[df_var.index[0]].tolist()[:-2] + df_bind_post_vec[j].loc[df_bind_post_vec[j]["Variant"]==variant].values.flatten().tolist()[1:] + [IL0,RA0,RB0,RG0,STAT50,Tf,dT,np.nan]
                    
    return df_sim_prior, df_sim_post

######################################################################################################################################################

def get_EC50_AMP_sample(df_sample):
    df_sample = df_sample[0]
    df_sample_EC50_AMP = pd.DataFrame(columns=['Num. sample', 'Plot', 'Cell_type', 'Variant', 'EC50', 'AMP'])
    for plot_num in df_sample["Plot"].drop_duplicates():
        df_plot = df_sample.loc[df_sample["Plot"] == plot_num]
        i = 0
        for variant in list(dict.fromkeys(df_plot["Variant"])):
            if i == 0:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                max_WT = df_var["Result"].max() 
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
                df_sample_EC50_AMP.loc[len(df_sample_EC50_AMP)] = [df_var["Num. sample"].values[0], df_var['Plot'].values[0], df_var['Cell_type'].values[0], df_var['Variant'].values[0], np.log10(fit[1]), fit[0]]
            else:
                df_var = df_plot.loc[df_plot["Variant"] == variant]
                fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
                df_sample_EC50_AMP.loc[len(df_sample_EC50_AMP)] = [df_var["Num. sample"].values[0], df_var['Plot'].values[0], df_var['Cell_type'].values[0], df_var['Variant'].values[0], np.log10(fit[1]), fit[0]]
            i += 1
    return df_sample_EC50_AMP

######################################################################################################################################################

def get_EC50_AMP_prior_post(df_res, num_cores):
    
    param_parallel = []
    for i in df_res["Num. sample"].drop_duplicates():
        param_parallel.append([df_res.loc[df_res["Num. sample"]==i]])
    with Pool(processes=num_cores) as pool:
        out_vec = pool.map(get_EC50_AMP_sample, param_parallel)  
    return pd.concat(out_vec, axis=0)

######################################################################################################################################################
    
def model_selection_ABC_SMC(model_function_dict, df_fit_data, df_bind, df_EC50_AMP, df_IC_data, df_linear_models, df_sim_data, param_depen, num_IL_sim, num_cores_ABC_SMC, ABC_SMC_path, conf95, pop_size_init, pop_size_max, E_cv, min_epsilon, max_num_iter):
    """
    # Update the dataset of the binding rates
    def update_df_bind_ABC(df_bind, param_dict, param_depen, variants_list):
        # Create copy of binding dataframe
        df_bind_fit = df_bind.copy()

        # Update df_bind with new parameter values
        i=0
        for variant in variants_list:
            if variant == "WT":
                for param in [key.split('__')[1] for key in param_dict.keys() if key.split('__')[0] == variant]:
                    # Copy new parameter to dataframe
                    df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param] = 10**param_dict[variant+"__"+param]
                    # Parameter dependancies
                    if param in param_depen.keys():
                        for param_d in param_depen[param]:
                            param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                            param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                            df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param_d] = (10**param_dict[variant+"__"+param])/param1_0*param2_0
                    # Parameter which are the same for WT and other variants must be also changed in the variant parameter row
                    for variant_WT_mut in variants_fit[1:]: # For all the non-WT variants
                        if param not in [key.split('__')[1] for key in param_dict.keys() if key.split('__')[0] == variant_WT_mut]:
                            param_dep_bool = True # If parameter is not unique to the variant, then it must be changed following the WT
                            for param2 in [key.split('__')[1] for key in param_dict.keys() if key.split('__')[0] == variant_WT_mut]:
                                if param2 in param_depen.keys(): 
                                    if param == param_depen[param2]:
                                        param_dep_bool = False # If parameter is not unique to the variant, but is dependant to a unique one, then it must be changed following the variant parameter
                            # Change the variant non-unqiue parameter and the ones that depend on it, following the WT one
                            if param_dep_bool:
                                df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).Variant, :].index[0], param] = 10**param_dict[variant+"__"+param]
                                # Parameter dependancies
                                if param in param_depen.keys():
                                    for param_d in param_depen[param]:
                                        param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                                        param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                                        df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).loc[pd.DataFrame(df_bind_fit["Variant"] == variant_WT_mut).Variant, :].index[0], param_d] = (10**param_dict[variant+"__"+param])/param1_0*param2_0
            # If parameter is not specified as WT, then its unique to the variant -> change only parameter and the ones that depend on it
            else:
                for param in [key.split('__')[1] for key in param_dict.keys() if key.split('__')[0] == variant]:
                    df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param] = 10**param_dict[variant+"__"+param]
                    # Parameter dependancies
                    if param in param_depen.keys():
                        for param_d in param_depen[param]:
                            param1_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param]
                            param2_0 = df_bind.loc[pd.DataFrame(df_bind["Variant"] == variant).loc[pd.DataFrame(df_bind["Variant"] == variant).Variant, :].index[0], param_d]
                            df_bind_fit.loc[pd.DataFrame(df_bind_fit["Variant"] == variant).loc[pd.DataFrame(df_bind_fit["Variant"] == variant).Variant, :].index[0], param_d] = (10**param_dict[variant+"__"+param])/param1_0*param2_0
        return df_bind_fit
    """

    # From EC50_AMP_model and EC50_AMP_exp
    def cost_ABC(EC50_AMPm, EC50_AMP0):
        # Get wighted sum of squares (Euclidian distance)    
        e_AMP = 0
        e_EC50 = 0
        for key in EC50_AMP0.keys():
            e_AMP += (EC50_AMP0[key][0] - EC50_AMPm[key][0])**2
            e_EC50 += (EC50_AMP0[key][1] - EC50_AMPm[key][1])**2
        e_AMP = e_AMP/w_AMP
        e_EC50 = e_EC50/w_EC50
        return e_AMP + e_EC50
    
    # Create the model's functions
    def create_function(name, model_function):
        # Define the function dynamically. From parameter_dict, simulate the desired model for all cells/variants we need
        def dynamic_model(parameter_dict):
            # Update df_bind
            df_bind_fit = update_df_bind_ABC(df_bind, parameter_dict, param_depen, variants_fit)

            # Create dataset of simulations
            df_sim = set_simulation_ABC(df_bind_fit, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim)

            # Simulate
            df_sim = simulate_sequential(df_sim, model_function)
            
            # Add missing simulation data on certain plots where the pair of cell__variant is found more than one time but it can be simualted oly once
            for plot_num_copy in cell__variant_missing_in_plot.keys():
                for plot_num_paste in cell__variant_missing_in_plot[plot_num_copy]:
                    df_copy = df_sim.loc[df_sim["Plot"]==plot_num_copy].loc[df_sim.loc[df_sim["Plot"]==plot_num_copy]["Variant"]=="WT"]
                    df_copy['Plot'] = plot_num_paste
                    df_sim = pd.concat([df_sim, df_copy], ignore_index=True)

            # Calculate EC50 and Amplitude error to add weight
            EC50_AMP = {key: [] for key in df_EC50_AMP["Plot"].astype(str)+"__"+df_EC50_AMP["Variant"]}
            plot_num_old = 0
            # For each celltype/variant simulated
            for key in EC50_AMP.keys():
                plot_num, variant = key.split("__")
                plot_num = int(plot_num)
                df_var = df_sim.loc[df_sim["Plot"] == plot_num].loc[df_sim.loc[df_sim["Plot"] == plot_num]["Variant"] == variant]
                # Data is normalized to the first variant that appears in a plot
                if plot_num_old != plot_num:
                    max_WT = df_var["Result"].max()
                    fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
                    EC50_AMP[key].append(fit[0])
                    EC50_AMP[key].append(np.log10(fit[1]))
                else:
                    fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
                    EC50_AMP[key].append(fit[0])
                    EC50_AMP[key].append(np.log10(fit[1]))
                plot_num_old = plot_num
            return EC50_AMP,df_sim

        # Set the function name
        dynamic_model.__name__ = name

        return dynamic_model

    # Create the different models' functions dynamically
    model_function_list = [create_function(name, value) for name, value in model_function_dict.items()]
    
    # Save parameters to fit in a way that can be used by pyABC
    param_fit = {} # Dictionary of lists of parameters to fit(variant__parameter)
    for model_name in model_function_dict.keys():
        param_fit[model_name] = list(df_fit_data.loc[df_fit_data["Model"].str.contains("Control"),"Var__Param"])
    variants_fit = list(dict.fromkeys([param.split("__")[0] for param in param_fit])) # List of variants to fit
    
    # Get the EC50 and Amplitude of data to fit in a dictionary (pyABC required)
    EC50_AMP_0 = {key: [] for key in df_EC50_AMP["Plot"].astype(str)+"__"+df_EC50_AMP["Variant"]}
    for key in EC50_AMP_0.keys():
        plot_num, variant = key.split("__")
        plot_num = int(plot_num)
        EC50_AMP_0[key].append(df_EC50_AMP.loc[df_EC50_AMP["Plot"]==plot_num].loc[df_EC50_AMP.loc[df_EC50_AMP["Plot"]==plot_num]["Variant"]==variant,"Amp_t"].values[0])
        EC50_AMP_0[key].append(df_EC50_AMP.loc[df_EC50_AMP["Plot"]==plot_num].loc[df_EC50_AMP.loc[df_EC50_AMP["Plot"]==plot_num]["Variant"]==variant,"EC50_t"].values[0])
        
    # Generate the priors
    prior = []
    for model_name in model_function_dict.keys():
        prior_dict = {}
        for param in param_fit[model_name]:
            prior_dict[param] = pyabc.RV("norm", loc=np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==param,"Initial_val"].values[0]), scale=conf95/4)
        prior.append(pyabc.Distribution(**prior_dict))
        
    # Simulate at the peak of the prior distribution to get cost function weights using "Control" model
    start = time.time()
    dict_param_fit_init = {key: np.log10(df_fit_data.loc[df_fit_data["Var__Param"]==key,"Initial_val"].values[0]) for key in param_fit["Control"]}
    df_bind_init = update_df_bind_ABC(df_bind, dict_param_fit_init, param_depen, variants_fit)

    # Create dataset of simulations
    df_sim = set_simulation_ABC(df_bind_init, df_sim_data, df_IC_data, df_linear_models, df_EC50_AMP, num_IL_sim)

    # Simulate
    df_sim = simulate_sequential(df_sim, model_function_dict["Control"])
    
    # Add missing simulation data on certain plots where the pair of cell__variant is found more than one time but it can be simualted oly once
    for plot_num_copy in cell__variant_missing_in_plot.keys():
        for plot_num_paste in cell__variant_missing_in_plot[plot_num_copy]:
            df_copy = df_sim.loc[df_sim["Plot"]==plot_num_copy].loc[df_sim.loc[df_sim["Plot"]==plot_num_copy]["Variant"]=="WT"]
            df_copy['Plot'] = plot_num_paste
            df_sim = pd.concat([df_sim, df_copy], ignore_index=True)

    # Calculate EC50 and Amplitude error to add weight
    EC50_AMP = {key: [] for key in df_EC50_AMP["Plot"].astype(str)+"__"+df_EC50_AMP["Variant"]}
    plot_num_old = 0
    # For each celltype/variant simulated
    for key in EC50_AMP.keys():
        plot_num, variant = key.split("__")
        plot_num = int(plot_num)
        df_var = df_sim.loc[df_sim["Plot"] == plot_num].loc[df_sim.loc[df_sim["Plot"] == plot_num]["Variant"] == variant]
        # Data is normalized to the first variant that appears in a plot
        if plot_num_old != plot_num:
            max_WT = df_var["Result"].max()
            fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [110, df_var["IL0"].max()*10]))
            EC50_AMP[key].append(fit[0])
            EC50_AMP[key].append(np.log10(fit[1]))
        else:
            fit, cov = curve_fit(hill_func, df_var["IL0"], df_var["Result"].values*100/(max_WT), bounds = ([0,df_var["IL0"].min()/10], [df_var["Result"].max()*100/(max_WT)+10, df_var["IL0"].max()*10]))
            EC50_AMP[key].append(fit[0])
            EC50_AMP[key].append(np.log10(fit[1]))
        plot_num_old = plot_num

    # Get weights for the sum of squares (Euclidian distance function)    
    w_AMP = 0
    w_EC50 = 0
    for key in EC50_AMP_0.keys():
        w_AMP += (EC50_AMP_0[key][0] - EC50_AMP[key][0])**2
        w_EC50 += (EC50_AMP_0[key][1] - EC50_AMP[key][1])**2

    print("EC50 mean error: " + str((w_EC50/len(EC50_AMP_0.keys()))**(1/2)))
    print("Amplitude mean error: " + str((w_AMP/len(EC50_AMP_0.keys()))**(1/2)))
    print("Inital cost: " + str(cost_ABC(EC50_AMP, EC50_AMP_0)))
    
    # Execute ABC-SMC model selection
    abc = pyabc.ABCSMC(model_function_list, prior, cost_ABC, sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=num_cores_ABC_SMC), population_size=pyabc.populationstrategy.AdaptivePopulationSize(start_nr_particles=pop_size_init, mean_cv=E_cv, max_population_size=pop_size_max))

    # Get path to save ABC-SMC runs
    abc.new("sqlite:///" + ABC_SMC_path, EC50_AMP_0)

    # Execute ABC-SMC parameter inference
    abc.run(minimum_epsilon=min_epsilon, max_nr_populations=max_num_iter)
    
###################################################################################################################################################### 
    
