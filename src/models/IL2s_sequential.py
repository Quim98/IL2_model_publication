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
    Monomer('RB', ['bB']) 
    Monomer('RG', ['bG'])
    
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
    
    # Initial conditions
    Parameter('IL_0', PA["IL0"].values[0]*Na*PA["vol_EC"].values[0])
    Initial(IL(bA=None, bB=None, bG=None, B_t='U') ** EC, IL_0)
    
    Parameter('RA_0',10**(PA["RA0"].values[0]))
    Parameter('RB_0',10**(PA["RB0"].values[0]))
    Parameter('RG_0',10**(PA["RG0"].values[0]))
    Initial(RA(bA=None) ** Cell_PM, RA_0)
    Initial(RB(bB=None) ** Cell_PM, RB_0)
    Initial(RG(bG=None) ** Cell_PM, RG_0)
    
    # Simulation
    Observable('IL2_complex', IL(bB=2, bG=3, B_t='B') % RB(bB=2) % RG(bG=3))

    t = np.arange(0,PA["Tf"].values[0],PA["dT"].values[0])
    simulator = ScipyOdeSimulator(model, tspan=t, compiler='cython', integrator = 'lsoda').run()
    
    if PA["STAT_type"].values[0] == "pSTAT5":
        KbS = PA["k_STAT5_f"].values[0]/(Na*PA["vol_cell"].values[0])
        KuS = PA["k_STAT5_b"].values[0]
        KpS = PA["k_PHOS"].values[0]
        KdS = PA["k_DEPHOS"].values[0]
        C0 = simulator.all['IL2_complex'][-1]
        S0 = 10**(PA["STAT50"].values[0])
        PA["Result"] = ((KbS*C0+KdS/KpS*(KbS*(C0+S0)+(KuS+KpS)))-((KbS*C0+KdS/KpS*(KbS*(C0+S0)+(KuS+KpS)))**2-4*KbS*KdS/KpS*(1+KdS/KpS)*KbS*C0*S0)**(1/2))/(2*KbS*KdS/KpS*(1+KdS/KpS))
        
    return PA

