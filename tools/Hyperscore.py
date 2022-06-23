import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _decimal_places(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

def hyperscore(ions, matched_ions):
    ## 1. Normalize intensity to 10^5
    norm = (ions.INT / ions.INT.max()) * 10E4
    ions["MSF_INT"] = norm
    
    return(hs)