import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _decimal_places(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

def hyperscore(ions, proof):
    ## 1. Normalize intensity to 10^5
    ions["MSF_INT"] = (ions.INT / ions.INT.max()) * 10E4
    
    ## 2. Pick matched ions ##
    matched_ions = pd.merge(proof, ions, on="MZ")
    
    ## 3. Adjust intenisty
    matched_ions.MSF_INT = matched_ions.MSF_INT / 10E2
    
    ## 4. Hyperscore ##
    
    return(hs)