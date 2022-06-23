import math
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
    
    ## 3. Adjust intensity
    matched_ions.MSF_INT = matched_ions.MSF_INT / 10E2
    
    ## 4. Hyperscore ##
    matched_ions["SERIES"] = matched_ions.apply(lambda x: x.FRAGS[0], axis=1)
    n_b = matched_ions.SERIES.value_counts()['b']
    n_y = matched_ions.SERIES.value_counts()['y']
    
    hs = math.log10(math.factorial(n_b) * math.factorial(n_y))
    return(hs)