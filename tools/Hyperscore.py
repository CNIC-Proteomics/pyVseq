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
    norm = (ions.INT / ions.INT.max()) * 1E5
    ions.reset_index(drop=True, inplace=True)
    ions["MSF_INT"] = norm
    ## 2. Pick matched ions ##
    matched_ions = pd.merge(proof, ions, on="MZ")
    if len(matched_ions) == 0:
        hs = 0
        return(hs)
    ## 3. Adjust intensity
    matched_ions.MSF_INT = matched_ions.MSF_INT / 1E3
    ## 4. Hyperscore ##
    matched_ions["SERIES"] = matched_ions.apply(lambda x: x.FRAGS[0], axis=1)
    matched_ions.FRAGS = matched_ions.FRAGS.str.replace('+', '', regex=False)
    matched_ions.FRAGS = matched_ions.FRAGS.str.replace('*', '', regex=False)
    matched_ions.FRAGS = matched_ions.FRAGS.str.replace('#', '', regex=False)
    temp = matched_ions.copy()
    # temp = temp.drop_duplicates(subset='FRAGS', keep="first") # Count each kind of fragment only once
    try:
        n_b = temp.SERIES.value_counts()['b']
        i_b = matched_ions[matched_ions.SERIES=='b'].MSF_INT.sum()
    except KeyError:
        n_b = 1 # So that hyperscore will not be 0 if one series is missing
        i_b = 1
    try:
        n_y = temp.SERIES.value_counts()['y']
        i_y = matched_ions[matched_ions.SERIES=='y'].MSF_INT.sum()
    except KeyError:
        n_y = 1
        i_y = 1
    try:
        hs = math.log10(math.factorial(n_b) * math.factorial(n_y) * i_b * i_y)
    except ValueError:
        hs = 0
    return(hs)

# Problem 1 is that MGF and mzML conversion is different - always use same format for MSFragger and Vseq to compare
# Problem 2 is that MSFragger does binning and Vseq does not. Build Recom-like with binning to check.
# MSFragger does *not* take into account b1 and y1 fragments?