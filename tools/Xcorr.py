import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _decimal_places(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

infile = r"\\Tierra\SC\U_Proteomica\UNIDAD\DatosCrudos\JorgeAlegreCebollada\Glyco_Titin\experiment_Oct21\8870\Titin_glyco.51762.51762.0.dta"
scan = pd.read_table(infile, index_col=None, header=0, delimiter=" ", names=["MZ", "INT"])
plt.plot(scan.MZ, scan.INT, linewidth=0.5)
plt.title("0. Raw Spectrum")

def Xcorr(seq, theo_spec, exp_spec):
    ################
    ## 1. BINNING ##   # Xcorr without binning would be better # but then what do we use as offset?
    ################
    bin_width = 0.02 # TODO: calculate bin_width in m/z from sequence and ppm threshold (40 ppm in PD)
    bins = list(np.arange(int(round(min(scan.MZ))),
                          int(round(max(scan.MZ)))+bin_width,
                          bin_width))
    bins = [round(x, _decimal_places(bin_width)) for x in bins]
    scan['BIN'] = pd.cut(scan.MZ, bins=bins)
    bins_df = pd.DataFrame(scan.groupby("BIN")["INT"].sum())
    bins_df.reset_index(drop=False, inplace=True)
    bins_df.insert(1, 'MZ', bins_df['BIN'].apply(lambda x: x.mid))
    plt.plot(bins_df.MZ, bins_df.INT, linewidth=0.5)
    plt.title("1. Binning: " + str(bin_width) + " bin width")
    
    ###################################
    ## 2. SQUARE ROOT OF INTENSITIES ##
    ###################################
    bins_df["SQRT_INT"] = np.sqrt(bins_df.INT) # TODO: sqrt before or after binning?
    plt.plot(bins_df.MZ, bins_df.SQRT_INT, linewidth=0.5)
    plt.title("2. Square Root of Intensities")
    
    ######################
    ## 3. NORMALIZATION ##
    ######################
    mz_windows = 10 # TODO: how many windows?
    # mz_range = np.linspace(min(bins_df.MZ), max(bins_df.MZ), num=mz_windows)
    windows = np.array_split(bins_df, mz_windows)
    normalized = []
    for w in windows:
        normarray = np.array(w.SQRT_INT).reshape(-1, 1)
        if normarray.max() > 0:
            norm = (normarray - normarray.min()) / (normarray.max() - normarray.min())
        else: # nothing to normalize
            norm = np.array(w.SQRT_INT).reshape(-1, 1)
        normalized.append(norm * 50) # max. intensity always 50
    bins_df["NORM_INT"] = np.concatenate(normalized)
    plt.plot(bins_df.MZ, bins_df.NORM_INT, linewidth=0.5)
    plt.title("3. Normalization of Intensities: " + str(mz_windows) + " m/z windows")
    
    #########################
    ## 4. GENERATE OFFSETS ##
    #########################
    offset_by = 1 # bin
    offset_n = 75 # ± bins
    # Similarity at no offset - Mean of similarities at all offsets
    offset_df = 
    return(xcorr)

