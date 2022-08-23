import argparse
import concurrent.futures
import configparser
import itertools
import logging
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyopenms
import re
from scipy.stats import chi2_contingency, poisson
import sys
from tqdm import tqdm
from bisect import bisect_left
pd.options.mode.chained_assignment = None  # default='warn'

def findFULL(fulls, scan, scanrange):
    pos = int(bisect_left(list(fulls), int(scan)))
    if pos == 0:
        return fulls[:pos+1+scanrange]
    if pos == len(fulls):
        return fulls[pos-1-scanrange:]
    if int(fulls[pos]) <= int(scan):
        return fulls[pos-scanrange:pos+1+scanrange]
    else:
        return fulls[pos-1-scanrange:pos+scanrange]
    
def readMZML(mzmlpath, scan, scanrange):
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzmlpath, exp)
    # Keep only full scans
    fulls = []
    for s in exp.getSpectra():
        if s.getMSLevel() == 1: # Keep only full scans
            fulls.append(int(s.getNativeID().split(' ')[-1][5:]))
    # query = np.arange(scan-scanrange,scan+scanrange+1,1)
    query = findFULL(fulls, scan, scanrange)
    spec = []
    for s in exp.getSpectra():
        # Keep only scans in range
        if s.getMSLevel() == 1 and int(s.getNativeID().split(' ')[-1][5:]) in query:
            df = pd.DataFrame([s.get_peaks()[0], s.get_peaks()[1]]).T
            df.columns = ["MZ", "INT"]
            spec.append(df)
    spec = pd.concat(spec)
    spec = spec.sort_values(by="MZ", ignore_index=True)
    return(spec)

def _decimal_places(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

def SumInt(row, prevrow):
    sumint = 0
    if isinstance(prevrow, pd.Series):
        sumint = prevrow.SUMINT + 1
    return(sumint)

def InInt(row, dtas):
    peakmz = dtas.iloc[row.name].MZ
    peakint = dtas.iloc[row.name].INT
    sigma = 4*10**-7 * peakmz**1.5027
    row = row.apply(lambda x: peakint * math.e**(-0.5*(x-peakmz)**2 / (sigma**2)))
    # inintlist = []
    # for i, j in enumerate(row.PEAKMZS):
    #     sigma = 4*10**-7 * row.PEAKMZS[i]**1.5027
    #     inint = row.PEAKINTS[i] * math.e**(-0.5*(row.MIDPOINT-row.PEAKMZS[i])**2 / (sigma**2))
    #     inintlist.append(inint)
    return(row)

def Integrate(scan, mz, scanrange, mzrange, bin_width, mzmlpath, n_workers):
    srange = int(scanrange)
    drange = float(mzrange)
    # Read MZML file #
    dtas = readMZML(mzmlpath, scan, srange)
    # Binning #
    bins = list(np.arange(mz-drange, mz+drange, bin_width))
    bins = [round(x, _decimal_places(bin_width)) for x in bins]
    bins_df = pd.DataFrame([bins])
    bins_df = bins_df.loc[bins_df.index.repeat(len(dtas))]
    bins_df.reset_index(drop=True, inplace=True)
    # Calculate intensity #
    indices, row_series = zip(*bins_df.iterrows())
    with concurrent.futures.ProcessPoolExecutor(n_workers) as executor:
        temp_bins_df = list(tqdm(executor.map(InInt, row_series, itertools.repeat(dtas), chunksize=500),
                                 total=len(row_series)))
    bins_df = pd.concat(temp_bins_df, axis=1).T
    apex_list = pd.DataFrame(bins_df.sum() / (srange*2+1), columns=["SUMINT"])
    apex_list["APEX"] = apex_list.SUMINT.diff(-1)
    apex_list.APEX[apex_list.APEX>0] = True
    apex_list.APEX[apex_list.APEX<=0] = False
    apex_list["APEX_B"] = apex_list["APEX"]
    for i, j in apex_list.iterrows():
        try:
            if j.APEX_B == True and apex_list.iloc[i-1].APEX_B == True:
                apex_list.at[i, "APEX"] = False
        except KeyError:
            continue
    apex_list.at[len(apex_list)-1, "APEX"] = False
    apex_list = apex_list.drop("APEX_B", axis=1)
    apex_list["BIN"] = bins
    # Filter apex #
    apexonly = pd.concat([apex_list[apex_list.APEX==True], apex_list[apex_list.SUMINT==0]])
    apexonly.sort_values(by=['BIN'], inplace=True)
    apexonly.reset_index(drop=True, inplace=True)
    for index, row in apexonly.iterrows(): # DUMMY MZ VALUES
        before = pd.Series([0]*row.shape[0], index=row.index)
        after = pd.Series([0]*row.shape[0], index=row.index)
        before.BIN = row.BIN - bin_width/10
        after.BIN = row.BIN + bin_width/10
        apexonly.loc[apexonly.shape[0]] = before
        apexonly.loc[apexonly.shape[0]] = after
    apexonly.sort_values(by=['BIN'], inplace=True)
    apexonly.reset_index(drop=True, inplace=True)
    return(apex_list, apexonly)

def PlotIntegration(theo_dist, mz, apex_list, apexonly, outplot, mz2=None, theo_dist2=None, out=False):
    apexannot = apexonly[apexonly.APEX==True].copy()
    apexannot = apexannot[apexannot.SUMINT>=apexannot.SUMINT.max()*0.1] # don't annotate small peaks to reduce clutter
    fig = plt.figure()
    if mz2:
        fig.set_size_inches(20, 15)
    else:
        fig.set_size_inches(20, 7.5)
    
    custom_lines = [Line2D([0], [0], color="darkblue", lw=2),
                Line2D([0], [0], color="salmon", lw=2),
                Line2D([0], [0], color="orange", lw=2, ls="--")]
    
    cont_table = pd.crosstab(index=list(theo_dist.exp_int),columns=list(theo_dist.P_compare))
    # chi2, p, dof, ex = chi2_contingency(np.array([list(theo_dist.exp_int), list(theo_dist.P_compare)]))
    chi2, p, dof, ex = chi2_contingency(cont_table)

    ax1 = fig.add_subplot(2,1,1)
    apex_list["COLOR"] = 'darkblue'
    apex_list.loc[apex_list.APEX == True, 'COLOR'] = 'red'
    plt.xlim(apex_list.BIN.min(), apex_list.BIN.max())
    plt.xlabel("M/Z", fontsize=15)
    plt.ylabel(r'$\sum_{n=0}^{n_{peaks}} Intensity_n \times e^{-\frac{1}{2}\times\frac{(BinMZ-PeakMZ)^2}{\sigma^2}} $', fontsize=15)
    plt.title("Integrated Scans", fontsize=20)
    plt.plot(apex_list.BIN, apex_list.SUMINT, linewidth=1, color="darkblue", zorder=4)
    plt.bar(theo_dist.theomz, theo_dist.P_compare, width=0.008, color="salmon", zorder=3)
    plt.axvline(x=mz, color='orange', ls="--", zorder=2) # Chosen peak
    ax1.annotate(str(round(mz,3)) + " Th", (mz,max(apex_list.SUMINT)-0.05*max(apex_list.SUMINT)),
                 style='italic', color='black', backgroundcolor='orange', fontsize=10, ha="left")
    for i,j in apexannot.iterrows():
        ax1.annotate(str(round(j.BIN,3)), (j.BIN, j.SUMINT))
    text_box = AnchoredText("Chi2:     " + str(round(chi2, 4)) + "\nDoF:       " + str(dof) + "\nP-value: " + str(round(p, 4)),
                            frameon=True, loc='upper left', pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    ax1.add_artist(text_box) # TODO check
    ax1.legend(custom_lines, ['Experimental peaks', 'Theoretical peaks', 'Chosen peak'],
               loc="upper right")
    
    ## RECOM GRAPH ##
    if mz2: # TODO chi2 not working bc it needs contingency table
        custom_lines = [Line2D([0], [0], color="darkblue", lw=2),
                    Line2D([0], [0], color="salmon", lw=2),
                    Line2D([0], [0], color="green", lw=2, ls="dotted")]
        cont_table2 = pd.crosstab(index=list(theo_dist2.exp_int),columns=list(theo_dist2.P_compare))
        # chi2_alt_peak, p_alt_peak, dof, ex = chi2_contingency(np.array([list(theo_dist2.exp_int), list(theo_dist2.P_compare)]))
        chi2_alt_peak, p_alt_peak, dof, ex = chi2_contingency(cont_table2)
        ax2 = fig.add_subplot(2,1,2)
        plt.xlim(apex_list.BIN.min(), apex_list.BIN.max())
        plt.xlabel("M/Z", fontsize=15)
        plt.ylabel(r'$\sum_{n=0}^{n_{peaks}} Intensity_n \times e^{-\frac{1}{2}\times\frac{(BinMZ-PeakMZ)^2}{\sigma^2}} $', fontsize=15)
        plt.title("Integrated Scans (corrected precursor)", fontsize=20)
        # plt.plot(apexonly.BIN, apexonly.SUMINT, linewidth=1, color="darkblue", zorder=4)
        plt.plot(apex_list.BIN, apex_list.SUMINT, linewidth=1, color="darkblue", zorder=4)
        plt.bar(theo_dist2.theomz, theo_dist2.P_compare, width=0.008, color="salmon", zorder=3)
        plt.axvline(x=theo_dist2.theomz.min(), color='green', ls="dotted", zorder=1) # Corrected peak
        ax2.annotate(str(round(mz2,3)) + " Th", (mz2,max(apex_list.SUMINT)-0.05*max(apex_list.SUMINT)),
                     style='italic', color='black', backgroundcolor='lightgreen', fontsize=10, ha="left")
        for i,j in apexannot.iterrows():
            ax2.annotate(str(round(j.BIN,3)), (j.BIN, j.SUMINT))
        text_box = AnchoredText("Chi2:     " + str(round(chi2_alt_peak, 4)) + "\nDoF:       " + str(dof) + "\nP-value: " + str(round(p_alt_peak, 4)),
                                frameon=True, loc='upper left', pad=0.5)
        plt.setp(text_box.patch, facecolor='white', alpha=0.5)
        ax2.add_artist(text_box) # TODO check
        ax2.legend(custom_lines, ['Experimental peaks', 'Theoretical peaks', 'Corrected peak'],
                   loc="upper right")
    
    fig.savefig(outplot)
    fig.clear()
    plt.close(fig)
    
    if out:
        if mz2:
            return(chi2, p, chi2_alt_peak, p_alt_peak)
        else:
            return(chi2, p)
    else:
        return

def getTheoMH(charge, sequence, mods, pos, nt, ct, massconfig, standalone):
    '''    
    Calculate theoretical MH using the PSM sequence.
    '''
    mass = massconfig
    AAs = dict(mass._sections['Aminoacids'])
    MODs = dict(mass._sections['Fixed Modifications'])
    m_proton = mass.getfloat('Masses', 'm_proton')
    m_hydrogen = mass.getfloat('Masses', 'm_hydrogen')
    m_oxygen = mass.getfloat('Masses', 'm_oxygen')
    total_aas = 2*m_hydrogen + m_oxygen
    total_aas += charge*m_proton
    #total_aas += float(MODs['nt']) + float(MODs['ct'])
    if nt:
        total_aas += float(MODs['nt'])
    if ct:
        total_aas += float(MODs['ct'])
    for i, aa in enumerate(sequence):
        if aa.lower() in AAs:
            total_aas += float(AAs[aa.lower()])
        if aa.lower() in MODs:
            total_aas += float(MODs[aa.lower()])
        # if aa.islower():
        #     total_aas += float(MODs['isolab'])
        if i in pos:
            total_aas += float(mods[pos.index(i)])
    MH = total_aas - (charge-1)*m_proton
    return MH



def main(args):
    '''
    Main function
    '''
    ## PARAMETERS ##    
    srange = int(mass._sections['Parameters']['int_scanrange'])
    drange = float(mass._sections['Parameters']['int_mzrange'])
    bin_width = float(mass._sections['Parameters']['int_binwidth'])
    t_poisson = float(mass._sections['Parameters']['poisson_threshold'])

    logging.info("Scan range: ±" + str(srange))
    logging.info("MZ range: ±" + str(drange) + " Th")
    logging.info("Bin width: " + str(bin_width) + " Th")
    logging.info("Reading input table...")
    query = pd.read_table(Path(args.infile), index_col=None, delimiter="\t")
    query["chi2"] = None
    query["p_value"] = None
    if 'alt_peak' in query.columns: # Recom
        query["chi2_alt_peak"] = None
        query["p_value_alt_peak"] = None
    logging.info("Looking for .mzML files...")
    mzmlfiles = os.listdir(Path(args.raw))
    mzmlfiles = [i for i in mzmlfiles if i[-5:].lower()=='.mzml']
    check = [i[:-5] for i in mzmlfiles if i[-5:].lower()=='.mzml']
    logging.info(str(len(mzmlfiles)) + " mzML files found.")
    missing = list(set(list(query.Raw.unique())) - set(check))
    if missing:
        logging.info("mzML files missing!" + str(missing))
        sys.exit()

    outpath = os.path.join(args.outpath, "Integration.tsv")

    for mzml in mzmlfiles:
        # GetSubQuery
        sub = query[query.Raw==mzml[:-5]].copy()
        if len(sub) <= 0: # Skip reading mzml if there are no scans to search
            continue
        logging.info("Reading file " + str(mzml))
        msdata = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(os.path.join(Path(args.raw), mzml), msdata)
        mzml = mzml[:-5]
        ref = []
        for r in msdata.getSpectra():
            if r.getMSLevel() == 1: # FULL
                ref.append(int(r.getNativeID().split(' ')[-1][5:]))
        for i, q in sub.iterrows():
            logging.info("\tIntegrating SCAN=" + str(q.FirstScan) + "...")
            qfull = min(ref, key=lambda x:abs(x-q.FirstScan))
            qpos = ref.index(qfull)
            if qfull > q.FirstScan:
                qpos -= 1
                qfull = ref[qpos]
            mz = msdata.getSpectrum(int(q.FirstScan)-1).getPrecursors()[0].getMZ() # Experimental
            ## GET FULL SCANS ##
            spectra = pd.DataFrame(columns=["MZ", "INT"])
            fulls = ref[qpos-srange:qpos+srange+1]
            for f in fulls:
                s = msdata.getSpectrum(f-1)
                ions = pd.DataFrame([s.get_peaks()[0], s.get_peaks()[1]]).T
                ions.columns = ["MZ", "INT"]
                spectra = pd.concat([spectra, ions])
            spectra =  spectra.sort_values(by="MZ", ignore_index=True)
            ## BINNING ##
            bins = list(np.arange(mz-drange, mz+drange, bin_width))
            bins = [round(x, _decimal_places(bin_width)) for x in bins]
            bins_df = pd.DataFrame([bins])
            bins_df = bins_df.loc[bins_df.index.repeat(len(spectra))]
            bins_df.reset_index(drop=True, inplace=True)
            ## CALCULATE INTENSITY ##
            indices, row_series = zip(*bins_df.iterrows())
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
                temp_bins_df = list(tqdm(executor.map(InInt, row_series, itertools.repeat(spectra), chunksize=500),
                                         total=len(row_series)))
            bins_df = pd.concat(temp_bins_df, axis=1).T
            apex_list = pd.DataFrame(bins_df.sum() / (srange*2+1), columns=["SUMINT"])
            apex_list["APEX"] = apex_list.SUMINT.diff(-1)
            apex_list.APEX[apex_list.APEX>0] = True
            apex_list.APEX[apex_list.APEX<=0] = False
            apex_list["APEX_B"] = apex_list["APEX"]
            for h, j in apex_list.iterrows():
                try:
                    if j.APEX_B == True and apex_list.iloc[h-1].APEX_B == True:
                        apex_list.at[h, "APEX"] = False
                except KeyError:
                    continue
            apex_list.at[len(apex_list)-1, "APEX"] = False
            apex_list = apex_list.drop("APEX_B", axis=1)
            apex_list["BIN"] = bins
            ## FILTER APEX ##
            apexonly = pd.concat([apex_list[apex_list.APEX==True], apex_list[apex_list.SUMINT==0]])
            apexonly.sort_values(by=['BIN'], inplace=True)
            ## DUMMY MZ VALUES ##
            apexonly.reset_index(drop=True, inplace=True)
            for index, row in apexonly.iterrows():
                before = pd.Series([0]*row.shape[0], index=row.index)
                after = pd.Series([0]*row.shape[0], index=row.index)
                before.BIN = row.BIN - bin_width/10
                after.BIN = row.BIN + bin_width/10
                apexonly.loc[apexonly.shape[0]] = before
                apexonly.loc[apexonly.shape[0]] = after
            apexonly.sort_values(by=['BIN'], inplace=True)
            apexonly.reset_index(drop=True, inplace=True)
            outplot = os.path.join(args.outpath, str(q.Raw) + "_" + str(q.FirstScan)
                                   + "_ch" + str(q.Charge) + "_Integration.pdf")
            ## THEORETICAL ISOTOPIC ENVELOPE (POISSON) ##
            plainseq = ''.join(re.findall("[A-Z]+", q.Sequence))
            mods = [round(float(i),6) for i in re.findall("\d*\.?\d*", q.Sequence) if i]
            pos = [int(j)-1 for j, k in enumerate(q.Sequence) if k.lower() == '[']
            parental = getTheoMH(q.Charge, plainseq, mods, pos, True, True, mass, False)
            mim = q.MH
            dm = mim - parental
            theomh = parental + dm + (q.Charge-1)*mass.getfloat('Masses', 'm_proton')
            C13 = 1.003355 # Dalton
            est_C13 = (0.000594 * theomh) - 0.03091
            poisson_df = pd.DataFrame(list(range(0,9)))
            poisson_df.columns = ["n"]
            poisson_df["theomh"] = np.arange(theomh, theomh+8.5*C13, C13)
            poisson_df["theomz"] = poisson_df.theomh / q.Charge
            poisson_df["Poisson"] = poisson_df.apply(lambda x: poisson.pmf(x.n, est_C13), axis=1)
            poisson_df["cumsum"] = poisson_df.Poisson.cumsum()
            poisson_df = pd.concat([poisson_df[poisson_df["cumsum"]<t_poisson], poisson_df[poisson_df["cumsum"]>=t_poisson].head(1)])
            poisson_df["n_poisson"] = poisson_df.Poisson/poisson_df.Poisson.sum()
            # Select experimental peaks within tolerance
            apexonly2 = apexonly[apexonly.APEX==True].copy() 
            poisson_df["closest"] = [min(apexonly2.BIN, key=lambda x:abs(x-i)) for i in list(poisson_df.theomz)] # filter only those close to n_poisson
            poisson_df["dist"] = abs(poisson_df.theomz - poisson_df.closest)
            # poisson_df["match"] = poisson_df.apply(lambda x: True if x.dist<=bin_width*4 else False, axis=1) 
            poisson_filtered = poisson_df[poisson_df.dist<=bin_width*4].copy() # TODO: make 4 an adjustable param
            if len(apexonly2) <= 0:
                logging.info("\t\t\t\tNot enough information in the spectrum! 0 apexes found.")
                return
            poisson_filtered["exp_peak"] = poisson_filtered.apply(lambda x: min(list(apexonly2.BIN), key=lambda y:abs(y-x.theomz)), axis=1)
            poisson_filtered = poisson_filtered[poisson_filtered.exp_peak>=0]
            poisson_filtered["exp_int"] = poisson_filtered.apply(lambda x: float(apexonly2[apexonly2.BIN==x.exp_peak].SUMINT), axis=1)
            int_total = poisson_filtered.exp_int.sum()
            poisson_df["P_compare"] = poisson_df.apply(lambda x: x.n_poisson*int_total, axis=1)
            # TODO exp_peak is missing
            poisson_df["exp_peak"] = poisson_df.apply(lambda x: min(list(apexonly2.BIN), key=lambda y:abs(y-x.theomz)), axis=1)
            poisson_df["exp_int"] = poisson_df.apply(lambda x: float(apexonly2[apexonly2.BIN==x.exp_peak].SUMINT) if x.dist<=bin_width*4 else 0, axis=1)
            if 'alt_peak' in query.columns: # Recom
                q.alt_peak = (mz*q.Charge-q.DeltaMass+q.alt_peak)/q.Charge
                poisson_df2 = pd.DataFrame(list(range(0,9)))
                poisson_df2.columns = ["n"]
                poisson_df2["theomz"] = np.arange(q.alt_peak, q.alt_peak+(8.5*C13)/q.Charge, C13/q.Charge)
                poisson_df2["Poisson"] = poisson_df2.apply(lambda x: poisson.pmf(x.n, est_C13), axis=1)
                poisson_df2["cumsum"] = poisson_df2.Poisson.cumsum()
                poisson_df2 = pd.concat([poisson_df2[poisson_df2["cumsum"]<t_poisson], poisson_df2[poisson_df2["cumsum"]>=t_poisson].head(1)])
                poisson_df2["n_poisson"] = poisson_df2.Poisson/poisson_df2.Poisson.sum()
                poisson_df2["closest"] = [min(apexonly2.BIN, key=lambda x:abs(x-i)) for i in list(poisson_df2.theomz)] # filter only those close to n_poisson
                poisson_df2["dist"] = abs(poisson_df2.theomz - poisson_df2.closest) 
                poisson_filtered2 = poisson_df2[poisson_df2.dist<=bin_width*4].copy() # TODO: make 4 an adjustable param
                poisson_filtered2["exp_peak"] = poisson_df2.apply(lambda x: min(list(apexonly2.BIN), key=lambda y:abs(y-x.theomz)), axis=1)
                poisson_filtered2 = poisson_filtered2[poisson_filtered2.exp_peak>=0]
                poisson_filtered2["exp_int"] = poisson_filtered2.apply(lambda x: float(apexonly2[apexonly2.BIN==x.exp_peak].SUMINT), axis=1)
                int_total2 = poisson_filtered2.exp_int.sum()
                poisson_df2["P_compare"] = poisson_df2.apply(lambda x: x.n_poisson*int_total2, axis=1)
                poisson_df2["exp_peak"] = poisson_df2.apply(lambda x: min(list(apexonly2.BIN), key=lambda y:abs(y-x.theomz)), axis=1)
                poisson_df2["exp_int"] = poisson_df2.apply(lambda x: float(apexonly2[apexonly2.BIN==x.exp_peak].SUMINT) if x.dist<=bin_width*4 else 0, axis=1)
                chi2, p, chi2_alt_peak, p_alt_peak = PlotIntegration(poisson_df, mz, apex_list, apexonly, outplot, q.alt_peak, poisson_df2, out=True)
                sub.loc[i, 'chi2'] = chi2
                sub.loc[i, 'p_value'] = p
                sub.loc[i, 'chi2_alt_peak'] = chi2_alt_peak
                sub.loc[i, 'p_value_alt_peak'] = p_alt_peak
            else: # TODO fix list of INT given to chi2
                chi2, p = PlotIntegration(poisson_df, mz, apex_list, apexonly, outplot, out=True)
                sub.loc[i, 'chi2'] = chi2
                sub.loc[i, 'p_value'] = p
        # Save stats to table
        sub.to_csv(outpath, index=False, sep='\t', encoding='utf-8',
                   mode='a', header=not os.path.exists(outpath))
    return
        
    # for i, q in query.iterrows():
    #     ## SAVE OUTPUT ##
    #     outplot = Path(args.infile[:-4] + "_"+ str(int(q.SCAN)) + "_" + str(q.MZ) + ".pdf")
    #     fig.savefig(outplot)
    #     fig.clear()
    #     plt.close(fig)
    #     apex_list = apex_list.rename(columns={"SUMINT": "INTENSITY", "BIN": "MZ"})
    #     apex_list = apex_list.drop("COLOR", axis=1)
    #     apex_list = apex_list[apex_list.columns.tolist()[-1:] + apex_list.columns.tolist()[:-1]]
    #     outfile = Path(args.infile[:-4] + "_"+ str(int(q.SCAN)) + "_" + str(q.MZ) + ".tsv")
    #     apex_list.to_csv(outfile, index=False, sep='\t', encoding='utf-8')
    # return

if __name__ == '__main__':

    # multiprocessing.freeze_support()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='VseqExplorer',
        epilog='''
        Example:
            python VseqExplorer.py

        ''')
        
    defaultconfig = os.path.join(os.path.dirname(__file__), "ScanIntegrator.ini")
    
    parser.add_argument('-i',  '--infile', required=True, help='Table of scans to search')
    parser.add_argument('-r',  '--raw', required=True, help='Directory containing .mzML files')
    parser.add_argument('-s',  '--scanrange', default=6, help='± full scans to use')
    parser.add_argument('-m',  '--mzrange', default=2, help='± MZ window to use')
    parser.add_argument('-b',  '--bin', default=0.001, help='Bin width to use')
    parser.add_argument('-p',  '--poisson', default=0.8, help='Poisson coverage threshold')
    parser.add_argument('-c', '--config', default=defaultconfig, help='Path to custom config.ini file')
    parser.add_argument('-o', '--outpath', required=True, help='Path to save results')
    parser.add_argument('-w',  '--n_workers', type=int, default=4, help='Number of threads/n_workers (default: %(default)s)')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()
    
    # parse config
    mass = configparser.ConfigParser(inline_comment_prefixes='#')
    mass.read(args.config)
    if args.scanrange is not None:
        mass.set('Parameters', 'int_scanrange', str(args.scanrange))
    if args.mzrange is not None:
        mass.set('Parameters', 'int_mzrange', str(args.mzrange))
    if args.bin is not None:
        mass.set('Parameters', 'int_binwidth', str(args.bin))
    if args.poisson is not None:
        mass.set('Parameters', 'poisson_threshold', str(args.poisson))

    # logging debug level. By default, info level
    Path(args.outpath).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(args.outpath, 'ScanIntegrator_log.txt')
    log_file_debug = os.path.join(args.outpath, 'ScanIntegrator_log_debug.txt')
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            handlers=[logging.FileHandler(log_file_debug),
                                      logging.StreamHandler()])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            handlers=[logging.FileHandler(log_file),
                                      logging.StreamHandler()])

    # start main function
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    main(args)
    logging.info('end script')