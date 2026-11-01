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
    Monomer('IL', ['bA', 'bB', 'bG','B_t','typ'], {'B_t': ['B', 'U'],'typ': ['t1', 't2']})
    Monomer('RA', ['bA'])
    Monomer('RB', ['bB']) 
    Monomer('RG', ['bG'])
    
    # Parameters IL2 type 1
    Parameter('K_IL_RA_f_1', PA["k_IL_RA_f_1"].values[0]/(Na*PA["vol_EC_1"].values[0]))
    Parameter('K_IL_RA_b_1', PA["k_IL_RA_b_1"].values[0])
    Parameter('K_IL_RB_f_1', PA["k_IL_RB_f_1"].values[0]/(Na*PA["vol_EC_1"].values[0]))
    Parameter('K_IL_RB_b_1', PA["k_IL_RB_b_1"].values[0])
    Parameter('K_IL_RG_f_1', PA["k_IL_RG_f_1"].values[0]/(Na*PA["vol_EC_1"].values[0]))
    Parameter('K_IL_RG_b_1', PA["k_IL_RG_b_1"].values[0])
    
    Parameter('K_IL_RA2_f_1', PA["k_IL_RA_f_1"].values[0]/(Na*PA["surf_cell_1"].values[0]*Width_PM))
    Parameter('K_IL_RA2_b_1', PA["k_IL_RA_b_1"].values[0])
    Parameter('K_IL_RB2_f_1', PA["k_IL_RB_f_1"].values[0]/(Na*PA["surf_cell_1"].values[0]*Width_PM))
    Parameter('K_IL_RB2_b_1', PA["k_IL_RB_b_1"].values[0])
    Parameter('K_IL_RG2_f_1', PA["k_IL_RG_f_1"].values[0]/(Na*PA["surf_cell_1"].values[0]*Width_PM))
    Parameter('K_IL_RG2_b_1', PA["k_IL_RG_b_1"].values[0])
    
    Parameter('K_IL_RA_RB_f_1', PA["k_IL_RA_RB_f_1"].values[0]/(Na*PA["surf_cell_1"].values[0]*Width_PM))
    Parameter('K_IL_RA_RB_b_1', PA["k_IL_RA_RB_b_1"].values[0])
    Parameter('K_IL_RB_RG_f_1', PA["k_IL_RB_RG_f_1"].values[0]/(Na*PA["surf_cell_1"].values[0]*Width_PM))
    Parameter('K_IL_RB_RG_b_1', PA["k_IL_RB_RG_b_1"].values[0])
    Parameter('K_IL_RA_RB_RG_f_1', PA["k_IL_RA_RB_RG_f_1"].values[0]/(Na*PA["surf_cell_1"].values[0]*Width_PM))
    Parameter('K_IL_RA_RB_RG_b_1', PA["k_IL_RA_RB_RG_b_1"].values[0])
    
    # Parameters IL2 type 2
    Parameter('K_IL_RA_f_2', PA["k_IL_RA_f_2"].values[0]/(Na*PA["vol_EC_2"].values[0]))
    Parameter('K_IL_RA_b_2', PA["k_IL_RA_b_2"].values[0])
    Parameter('K_IL_RB_f_2', PA["k_IL_RB_f_2"].values[0]/(Na*PA["vol_EC_2"].values[0]))
    Parameter('K_IL_RB_b_2', PA["k_IL_RB_b_2"].values[0])
    Parameter('K_IL_RG_f_2', PA["k_IL_RG_f_2"].values[0]/(Na*PA["vol_EC_2"].values[0]))
    Parameter('K_IL_RG_b_2', PA["k_IL_RG_b_2"].values[0])
    
    Parameter('K_IL_RA2_f_2', PA["k_IL_RA_f_2"].values[0]/(Na*PA["surf_cell_2"].values[0]*Width_PM))
    Parameter('K_IL_RA2_b_2', PA["k_IL_RA_b_2"].values[0])
    Parameter('K_IL_RB2_f_2', PA["k_IL_RB_f_2"].values[0]/(Na*PA["surf_cell_2"].values[0]*Width_PM))
    Parameter('K_IL_RB2_b_2', PA["k_IL_RB_b_2"].values[0])
    Parameter('K_IL_RG2_f_2', PA["k_IL_RG_f_2"].values[0]/(Na*PA["surf_cell_2"].values[0]*Width_PM))
    Parameter('K_IL_RG2_b_2', PA["k_IL_RG_b_2"].values[0])
    
    Parameter('K_IL_RA_RB_f_2', PA["k_IL_RA_RB_f_2"].values[0]/(Na*PA["surf_cell_2"].values[0]*Width_PM))
    Parameter('K_IL_RA_RB_b_2', PA["k_IL_RA_RB_b_2"].values[0])
    Parameter('K_IL_RB_RG_f_2', PA["k_IL_RB_RG_f_2"].values[0]/(Na*PA["surf_cell_2"].values[0]*Width_PM))
    Parameter('K_IL_RB_RG_b_2', PA["k_IL_RB_RG_b_2"].values[0])
    Parameter('K_IL_RA_RB_RG_f_2', PA["k_IL_RA_RB_RG_f_2"].values[0]/(Na*PA["surf_cell_2"].values[0]*Width_PM))
    Parameter('K_IL_RA_RB_RG_b_2', PA["k_IL_RA_RB_RG_b_2"].values[0])
    
    
    # Complexes and reactions (binding of receptors to IL2 type 1)
    IL_free_1 = IL(bA=None, bB=None, bG=None, B_t='U', typ='t1')
    IL_RA_1 = IL(bA=1, bB=None, bG=None, B_t='B', typ='t1') % RA(bA=1)
    Rule('IL_RA_binding_1', IL_free_1 + RA(bA=None)| IL_RA_1, *[K_IL_RA_f_1, K_IL_RA_b_1])
    IL_RB_1 = IL(bA=None, bB=2, bG=None, B_t='B', typ='t1') % RB(bB=2)
    Rule('IL_RB_binding_1', IL_free_1 + RB(bB=None)| IL_RB_1, *[K_IL_RB_f_1, K_IL_RB_b_1])
    IL_RG_1 = IL(bA=None, bB=None, bG=3, B_t='B', typ='t1') % RG(bG=3)
    Rule('IL_RG_binding_1', IL_free_1 + RG(bG=None)| IL_RG_1, *[K_IL_RG_f_1, K_IL_RG_b_1])
    
    IL_RA_RB_1 = IL(bA=1, bB=2, bG=None, B_t='B', typ='t1') % RA(bA=1) % RB(bB=2)
    Rule('IL_RA_RB_binding_1', IL_RA_1 + RB(bB=None)| IL_RA_RB_1, *[K_IL_RA_RB_f_1, K_IL_RA_RB_b_1])
    IL_RA_RG_1 = IL(bA=1, bB=None, bG=3, B_t='B', typ='t1') % RA(bA=1) % RG(bG=3)
    Rule('IL_RA_RG_binding_1', IL_RA_1 + RG(bG=None)| IL_RA_RG_1, *[K_IL_RG2_f_1, K_IL_RG2_b_1])
    Rule('IL_RA_RB_binding_2_1', IL_RB_1 + RA(bA=None)| IL_RA_RB_1, *[K_IL_RA2_f_1, K_IL_RA2_b_1])
    IL_RB_RG_1 = IL(bA=None, bB=2, bG=3, B_t='B', typ='t1') % RB(bB=2) % RG(bG=3)
    Rule('IL_RB_RG_binding_1', IL_RB_1 + RG(bG=None)| IL_RB_RG_1, *[K_IL_RB_RG_f_1, K_IL_RB_RG_b_1])
    Rule('IL_RA_RG_binding_2_1', IL_RG_1 + RA(bA=None)| IL_RA_RG_1, *[K_IL_RA2_f_1, K_IL_RA2_b_1])
    Rule('IL_RB_RG_binding_2_1', IL_RG_1 + RB(bB=None)| IL_RB_RG_1, *[K_IL_RB2_f_1, K_IL_RB2_b_1])
    
    IL_RA_RB_RG_1 = IL(bA=1, bB=2, bG=3, B_t='B', typ='t1') % RA(bA=1) % RB(bB=2) % RG(bG=3)
    Rule('IL_RA_RB_RG_binding_1_1', IL_RA_RB_1 + RG(bG=None)| IL_RA_RB_RG_1, *[K_IL_RA_RB_RG_f_1, K_IL_RA_RB_RG_b_1])
    Rule('IL_RA_RB_RG_binding_2_1', IL_RA_RG_1 + RB(bB=None)| IL_RA_RB_RG_1, *[K_IL_RA_RB_f_1, K_IL_RA_RB_b_1])
    Rule('IL_RA_RB_RG_binding_3_1', IL_RB_RG_1 + RA(bA=None)| IL_RA_RB_RG_1, *[K_IL_RA2_f_1, K_IL_RA2_b_1])
    
    # Complexes and reactions (binding of receptors to IL2 type 2)
    IL_free_2 = IL(bA=None, bB=None, bG=None, B_t='U', typ='t2')
    IL_RA_2 = IL(bA=1, bB=None, bG=None, B_t='B', typ='t2') % RA(bA=1)
    Rule('IL_RA_binding_2', IL_free_2 + RA(bA=None)| IL_RA_2, *[K_IL_RA_f_2, K_IL_RA_b_2])
    IL_RB_2 = IL(bA=None, bB=2, bG=None, B_t='B', typ='t2') % RB(bB=2)
    Rule('IL_RB_binding_2', IL_free_2 + RB(bB=None)| IL_RB_2, *[K_IL_RB_f_2, K_IL_RB_b_2])
    IL_RG_2 = IL(bA=None, bB=None, bG=3, B_t='B', typ='t2') % RG(bG=3)
    Rule('IL_RG_binding_2', IL_free_2 + RG(bG=None)| IL_RG_2, *[K_IL_RG_f_2, K_IL_RG_b_2])
    
    IL_RA_RB_2 = IL(bA=1, bB=2, bG=None, B_t='B', typ='t2') % RA(bA=1) % RB(bB=2)
    Rule('IL_RA_RB_binding_2', IL_RA_2 + RB(bB=None)| IL_RA_RB_2, *[K_IL_RA_RB_f_2, K_IL_RA_RB_b_2])
    IL_RA_RG_2 = IL(bA=1, bB=None, bG=3, B_t='B', typ='t2') % RA(bA=1) % RG(bG=3)
    Rule('IL_RA_RG_binding_2', IL_RA_2 + RG(bG=None)| IL_RA_RG_2, *[K_IL_RG2_f_2, K_IL_RG2_b_2])
    Rule('IL_RA_RB_binding_2_2', IL_RB_2 + RA(bA=None)| IL_RA_RB_2, *[K_IL_RA2_f_2, K_IL_RA2_b_2])
    IL_RB_RG_2 = IL(bA=None, bB=2, bG=3, B_t='B', typ='t2') % RB(bB=2) % RG(bG=3)
    Rule('IL_RB_RG_binding_2', IL_RB_2 + RG(bG=None)| IL_RB_RG_2, *[K_IL_RB_RG_f_2, K_IL_RB_RG_b_2])
    Rule('IL_RA_RG_binding_2_2', IL_RG_2 + RA(bA=None)| IL_RA_RG_2, *[K_IL_RA2_f_2, K_IL_RA2_b_2])
    Rule('IL_RB_RG_binding_2_2', IL_RG_2 + RB(bB=None)| IL_RB_RG_2, *[K_IL_RB2_f_2, K_IL_RB2_b_2])
    
    IL_RA_RB_RG_2 = IL(bA=1, bB=2, bG=3, B_t='B', typ='t2') % RA(bA=1) % RB(bB=2) % RG(bG=3)
    Rule('IL_RA_RB_RG_binding_1_2', IL_RA_RB_2 + RG(bG=None)| IL_RA_RB_RG_2, *[K_IL_RA_RB_RG_f_2, K_IL_RA_RB_RG_b_2])
    Rule('IL_RA_RB_RG_binding_2_2', IL_RA_RG_2 + RB(bB=None)| IL_RA_RB_RG_2, *[K_IL_RA_RB_f_2, K_IL_RA_RB_b_2])
    Rule('IL_RA_RB_RG_binding_3_2', IL_RB_RG_2 + RA(bA=None)| IL_RA_RB_RG_2, *[K_IL_RA2_f_2, K_IL_RA2_b_2])
    
    # Initial conditions
    Parameter('IL_0_1', PA["IL0_1"].values[0]*Na*PA["vol_EC_1"].values[0])
    Initial(IL(bA=None, bB=None, bG=None, B_t='U', typ='t1') ** EC, IL_0_1)
    Parameter('IL_0_2', PA["IL0_2"].values[0]*Na*PA["vol_EC_2"].values[0])
    Initial(IL(bA=None, bB=None, bG=None, B_t='U', typ='t2') ** EC, IL_0_2)
    
    Parameter('RA_0',10**(PA["RA0"].values[0]))
    Parameter('RB_0',10**(PA["RB0"].values[0]))
    Parameter('RG_0',10**(PA["RG0"].values[0]))
    Initial(RA(bA=None) ** Cell_PM, RA_0)
    Initial(RB(bB=None) ** Cell_PM, RB_0)
    Initial(RG(bG=None) ** Cell_PM, RG_0)
    
    # Simulation
    Observable('IL2_complex', IL(bB=2, bG=3, B_t='B') % RB(bB=2) % RG(bG=3))
    
    t = np.arange(0,PA["Tf"].values[0],PA["dT"].values[0])
    simulator = ScipyOdeSimulator(model, tspan=t, compiler='python', integrator = 'lsoda').run()
    
    if PA["STAT_type"].values[0] == "pSTAT5":
        KbS = PA["k_STAT5_f_1"].values[0]/(Na*PA["vol_cell_1"].values[0])
        KuS = PA["k_STAT5_b_1"].values[0]
        KpS = PA["k_PHOS_1"].values[0]
        KdS = PA["k_DEPHOS_1"].values[0]
        C0 = simulator.all['IL2_complex'][-1]
        S0 = 10**(PA["STAT50"].values[0])
        PA["Result"] = ((KbS*C0+KdS/KpS*(KbS*(C0+S0)+(KuS+KpS)))-((KbS*C0+KdS/KpS*(KbS*(C0+S0)+(KuS+KpS)))**2-4*KbS*KdS/KpS*(1+KdS/KpS)*KbS*C0*S0)**(1/2))/(2*KbS*KdS/KpS*(1+KdS/KpS))
        
    return PA

