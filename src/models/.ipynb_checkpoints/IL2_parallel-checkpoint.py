from pysb import Model, Parameter, Compartment, Monomer, Parameter, Rule, Initial, Observable
from pysb.simulator import ScipyOdeSimulator
import numpy as np
import cython

def model_function(PA):
    PA=PA[0]
    Na=6.023e23
    Width_PM = 1e-7 # In dm
    # Initialize model
    Model()
    
    # Compartment
    Compartment(name='EC', parent=None, dimension=3, size=None)
    Compartment(name='Cell_PM', parent=EC, dimension=2, size=None)
    Compartment(name='Cell_CP', parent=Cell_PM, dimension=3, size=None)
    
    # Monomers
    Monomer('IL', ['bA', 'bB', 'bG','B_t'], {'B_t': ['B', 'U']})
    Monomer('RA', ['bA'])
    Monomer('RB', ['bB', 'bSTAT5']) 
    Monomer('RG', ['bG'])
    Monomer('STAT5', ['b5','P'], {'P': ['u', 'p']})
    
    # Parameters
    Parameter('K_IL_RA_f', PA["k_IL_RA_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    Parameter('K_IL_RA_b', PA["k_IL_RA_b"].values[0])
    Parameter('K_IL_RB_f', PA["k_IL_RB_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    Parameter('K_IL_RB_b', PA["k_IL_RB_b"].values[0])
    Parameter('K_IL_RG_f', PA["k_IL_RG_f"].values[0]/(Na*PA["vol_EC"].values[0]))
    Parameter('K_IL_RG_b', PA["k_IL_RG_b"].values[0])
    
    Parameter('K_IL_RA2_f', PA["k_IL_RA_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RA2_b', PA["k_IL_RA_b"].values[0])
    Parameter('K_IL_RB2_f', PA["k_IL_RB_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RB2_b', PA["k_IL_RB_b"].values[0])
    Parameter('K_IL_RG2_f', PA["k_IL_RG_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RG2_b', PA["k_IL_RG_b"].values[0])
    
    Parameter('K_IL_RA_RB_f', PA["k_IL_RA_RB_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RA_RB_b', PA["k_IL_RA_RB_b"].values[0])
    Parameter('K_IL_RB_RG_f', PA["k_IL_RB_RG_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RB_RG_b', PA["k_IL_RB_RG_b"].values[0])
    Parameter('K_IL_RA_RB_RG_f', PA["k_IL_RA_RB_RG_f"].values[0]/(Na*PA["surf_cell"].values[0]*Width_PM))
    Parameter('K_IL_RA_RB_RG_b', PA["k_IL_RA_RB_RG_b"].values[0])
    
    Parameter('K_STAT5_f', PA["k_STAT5_f"].values[0]/(Na*PA["vol_cell"].values[0]))
    Parameter('K_STAT5_b', PA["k_STAT5_b"].values[0])
    
    Parameter('K_PHOS', PA["k_PHOS"].values[0])   
    Parameter('K_DEPHOS', PA["k_DEPHOS"].values[0])
    
    # Complexes and reactions (binding of receptors)
    IL_free = IL(bA=None, bB=None, bG=None, B_t='U')
    IL_RA = IL(bA=1, bB=None, bG=None, B_t='B') % RA(bA=1)
    Rule('IL_RA_binding', IL_free + RA(bA=None)| IL_RA, *[K_IL_RA_f, K_IL_RA_b])
    IL_RB = IL(bA=None, bB=2, bG=None, B_t='B') % RB(bB=2)
    Rule('IL_RB_binding', IL_free + RB(bB=None)| IL_RB, *[K_IL_RB_f, K_IL_RB_b])
    IL_RG = IL(bA=None, bB=None, bG=3, B_t='B') % RG(bG=3)
    Rule('IL_RG_binding', IL_free + RG(bG=None)| IL_RG, *[K_IL_RG_f, K_IL_RG_b])
    
    IL_RA_RB = IL(bA=1, bB=2, bG=None, B_t='B') % RA(bA=1) % RB(bB=2)
    Rule('IL_RA_RB_binding', IL_RA + RB(bB=None)| IL_RA_RB, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    IL_RA_RG = IL(bA=1, bB=None, bG=3, B_t='B') % RA(bA=1) % RG(bG=3)
    Rule('IL_RA_RG_binding', IL_RA + RG(bG=None)| IL_RA_RG, *[K_IL_RG2_f, K_IL_RG2_b])
    Rule('IL_RA_RB_binding_2', IL_RB + RA(bA=None)| IL_RA_RB, *[K_IL_RA2_f, K_IL_RA2_b])
    IL_RB_RG = IL(bA=None, bB=2, bG=3, B_t='B') % RB(bB=2) % RG(bG=3)
    Rule('IL_RB_RG_binding', IL_RB + RG(bG=None)| IL_RB_RG, *[K_IL_RB_RG_f, K_IL_RB_RG_b])
    Rule('IL_RA_RG_binding_2', IL_RG + RA(bA=None)| IL_RA_RG, *[K_IL_RA2_f, K_IL_RA2_b])
    Rule('IL_RB_RG_binding_2', IL_RG + RB(bB=None)| IL_RB_RG, *[K_IL_RB2_f, K_IL_RB2_b])
    
    IL_RA_RB_RG = IL(bA=1, bB=2, bG=3, B_t='B') % RA(bA=1) % RB(bB=2) % RG(bG=3)
    Rule('IL_RA_RB_RG_binding_1', IL_RA_RB + RG(bG=None)| IL_RA_RB_RG, *[K_IL_RA_RB_RG_f, K_IL_RA_RB_RG_b])
    Rule('IL_RA_RB_RG_binding_2', IL_RA_RG + RB(bB=None)| IL_RA_RB_RG, *[K_IL_RA_RB_f, K_IL_RA_RB_b])
    Rule('IL_RA_RB_RG_binding_3', IL_RB_RG + RA(bA=None)| IL_RA_RB_RG, *[K_IL_RA2_f, K_IL_RA2_b])
    
    # Complexes and reactions (binding of STAT)
    IL_RB_RG_noSTAT5 = IL(bB=2, bG=3, B_t='B') % RB(bB=2, bSTAT5=None) % RG(bG=3)
    IL_RB_RG_STAT5 = IL(bB=2, bG=3, B_t='B') % RB(bB=2, bSTAT5=4) % RG(bG=3) % STAT5(b5=4, P='u')
    Rule('IL_RB_RG_STAT5_binding', IL_RB_RG_noSTAT5 + STAT5(b5=None, P='u') | IL_RB_RG_STAT5, *[K_STAT5_f, K_STAT5_b])
    
    # Reactions (STAT phosphorylation)
    Rule('IL_RB_RG_STAT5_phosphorylation', IL_RB_RG_STAT5 >> IL_RB_RG_noSTAT5 + STAT5(b5=None, P='p'), K_PHOS)
 
    # Reactions (STAT dephosphorylation)
    Rule('STAT5_dephosphorylation', STAT5(b5=None, P='p') >> STAT5(b5=None, P='u'), K_DEPHOS)
    
    
    # Initial conditions
    Parameter('IL_0', PA["IL0"].values[0]*Na*PA["vol_EC"].values[0])
    Initial(IL(bA=None, bB=None, bG=None, B_t='U') ** EC, IL_0)
    Parameter('RA_0', 10**(PA["RA0"].values[0]))
    Parameter('RB_0', 10**(PA["RB0"].values[0]))
    Parameter('RG_0', 10**(PA["RG0"].values[0]))
    Initial(RA(bA=None) ** Cell_PM, RA_0)
    Initial(RB(bB=None, bSTAT5=None) ** Cell_PM, RB_0)
    Initial(RG(bG=None) ** Cell_PM, RG_0)
    
    Parameter('STAT_5_0', 10**(PA["STAT50"].values[0]))
    Initial(STAT5(b5=None, P='u') ** Cell_CP, STAT_5_0)
    
    # Simulation
    Observable('STAT5P', STAT5(b5=None, P='p') ** Cell_CP)
    
    t = np.arange(0,PA["Tf"].values[0],PA["dT"].values[0])
    simulator = ScipyOdeSimulator(model, tspan=t, compiler='python', integrator = 'lsoda').run()
    
    if PA["STAT_type"].values[0] == "pSTAT5":
        PA["Result"] = simulator.all['STAT5P'][-1]
    
    return PA

