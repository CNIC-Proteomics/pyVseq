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

def Xcorr(seq, charge, assign, ions, m_proton):
    bin_width = 0.02 # TODO: calculate bin_width in m/z from sequence and ppm threshold (40 ppm in PD)
    offset_by = 1 # bin
    offset_n = 75 # ± bins
    exp_spec = ions
    # Prepare theoretical mass spectrum
    theo_spec = assign
    theo_spec.sort_values(by=['FRAGS'], inplace=True)
    # theo_spec = theo_spec.head(1).T
    # theo_spec.columns = ["MZ"]
    # theo_spec.MZ = (theo_spec.MZ + charge*m_proton) / charge
    theo_spec["INT"] = list(int(len(theo_spec)/2)*[1]) + list(int(len(theo_spec)/2)*[1]) # NO Half INT for b-series
    theo_spec.sort_values(by=['MZ'], inplace=True)
    theo_spec.reset_index(drop=True, inplace=True)
    ################
    ## 1. BINNING ##   # Xcorr without binning would be better # but then what do we use as offset?
    ################
    min_MZ = min(exp_spec.MZ) if min(exp_spec.MZ)<=min(theo_spec.MZ) else min(theo_spec.MZ)
    max_MZ = max(exp_spec.MZ) if max(exp_spec.MZ)>=min(theo_spec.MZ) else max(theo_spec.MZ)
    bins = list(np.arange(int(round(min_MZ)-(offset_n*bin_width)),
                          int(round(max(exp_spec.MZ))+(offset_n*(bin_width+1))),
                          bin_width))
    bins = [round(x, _decimal_places(bin_width)) for x in bins]
    exp_spec['BIN'] = pd.cut(exp_spec.MZ, bins=bins)
    bins_df = pd.DataFrame(exp_spec.groupby("BIN")["INT"].sum())
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
    mz_windows = 10 # Comet MS/MS uses 10 windows
    # mz_range = np.linspace(min(bins_df.MZ), max(bins_df.MZ), num=mz_windows)
    windows = np.array_split(bins_df, mz_windows)
    normalized = []
    for w in windows:
        normarray = np.array(w.SQRT_INT).reshape(-1, 1)
        if normarray.max() > 0:
            norm = (normarray - normarray.min()) / (normarray.max() - normarray.min())
        else: # nothing to normalize
            norm = np.array(w.SQRT_INT).reshape(-1, 1)
        normalized.append(norm * 1) # max. intensity always 50
    bins_df["NORM_INT"] = np.concatenate(normalized)
    plt.plot(bins_df.MZ, bins_df.NORM_INT, linewidth=0.5)
    plt.title("3. Normalization of Intensities: " + str(mz_windows) + " m/z windows")
    
    #########################
    ## 4. GENERATE OFFSETS ##
    #########################
    offsets = np.arange(-offset_n, offset_n+1, offset_by)
    # Similarity at no offset - Mean of similarities at all offsets
    offset_df = []
    bins_df.MZ = bins_df.MZ.astype('float')
    for o in offsets:
        # if o != 0:
        temp_df = bins_df.copy()
        temp_df.MZ = temp_df.MZ + (o*bin_width)
        # TODO: re-bin
        offset_df.append(temp_df)
            
    ##########################
    ## 5. CROSS CORRELATION ##
    ##########################
    theo_spec['BIN'] = pd.cut(theo_spec.MZ, bins=bins)
    theo_spec['MZ'] = theo_spec.BIN.apply(lambda x: x.mid)
    theo_spec = theo_spec[['MZ','INT']]
    # Dot product at each offset
    p_xcorrs = []
    for o_df in offset_df:
        o_df = pd.merge(o_df, theo_spec, on ='MZ', how ='left')
        o_df.INT_y = o_df.INT_y.fillna(0)
        p_xcorr = np.dot(o_df.NORM_INT, o_df.INT_y)
        p_xcorrs.append(p_xcorr)
    plt.plot(offsets, p_xcorrs, linewidth=0.5)
    plt.title("4. Cross-correlation at each offset")
    mean_sim = p_xcorrs[:75] + p_xcorrs[76:]
    mean_sim = sum(mean_sim) / len(mean_sim)
    xcorr = p_xcorrs[75] - mean_sim
    # TODO take offset 0 spectrum substract mean_sim and recalculate dot product
    temp_df = bins_df.copy()
    temp_df.NORM_INT = temp_df.NORM_INT - mean_sim
    temp_df = pd.merge(temp_df, theo_spec, on ='MZ', how ='left')
    temp_df.INT_y = temp_df.INT_y.fillna(0)
    p_xcorr = np.dot(temp_df.NORM_INT, temp_df.INT_y)
    # xcorr_corr
    # plt.xcorr
    # np.correlate
    return(xcorr)

