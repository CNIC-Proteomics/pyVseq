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
    bins_df = pd.DataFrame([bins]*len(dtas))
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

def PlotIntegration(theo_dist, mz, apex_list, apexonly, outplot):
    fig = plt.figure()
    fig.set_size_inches(20, 15)
    
    custom_lines = [Line2D([0], [0], color="darkblue", lw=2),
                Line2D([0], [0], color="salmon", lw=2),
                Line2D([0], [0], color="orange", lw=2, ls="--"),
                Line2D([0], [0], color="green", lw=2, ls="dotted")]
    
    chi2, p, dof, ex = chi2_contingency(np.array([list(theo_dist.exp_int), list(theo_dist.P_compare)]))
    
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
    plt.axvline(x=theo_dist.theomz.min(), color='green', ls="dotted", zorder=1) # Monoisotopic peak
    ax1.annotate(str(mz) + " Th", (mz,max(apex_list.SUMINT)-0.05*max(apex_list.SUMINT)), color='black', fontsize=10, ha="left")
    ax1.legend(custom_lines, ['Experimental peaks', 'Theoretical peaks', 'Chosen peak', 'Monoisotopic peak'],
               loc="upper right")

    ax2 = fig.add_subplot(2,1,2)
    plt.xlim(apex_list.BIN.min(), apex_list.BIN.max())
    plt.xlabel("M/Z", fontsize=15)
    plt.ylabel(r'$\sum_{n=0}^{n_{peaks}} Intensity_n \times e^{-\frac{1}{2}\times\frac{(BinMZ-PeakMZ)^2}{\sigma^2}} $', fontsize=15)
    plt.title("Integrated Scans (apexes only)", fontsize=20)
    plt.plot(apexonly.BIN, apexonly.SUMINT, linewidth=1, color="darkblue", zorder=4)
    plt.bar(theo_dist.theomz, theo_dist.P_compare, width=0.008, color="salmon", zorder=3)
    plt.axvline(x=mz, color='orange', ls="--", zorder=2)
    plt.axvline(x=theo_dist.theomz.min(), color='green', ls="dotted", zorder=1)
    ax2.annotate(str(mz) + " Th", (mz,max(apex_list.SUMINT)-0.05*max(apex_list.SUMINT)), color='black', fontsize=10, ha="left")
    text_box = AnchoredText("Chi2:     " + str(round(chi2, 2)) + "\nP-value: " + str(p) + "\nDoF:       " + str(dof),
                            frameon=True, loc='upper left', pad=0.5)
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    ax2.add_artist(text_box) # TODO check
    ax2.legend(custom_lines, ['Experimental peaks', 'Theoretical peaks', 'Chosen peak', 'Monoisotopic peak'],
               loc="upper right")
    
    fig.savefig(outplot)
    fig.clear()
    plt.close(fig)
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

def prePlotIntegration(sub, mz, scanrange, mzrange, bin_width, t_poisson, mzmlpath, out, n_workers):
    ''' Integrate and save apex list and plot to files. '''
    outpath = os.path.join(out, str(sub.Raw) +
                           "_" + str(sub.Sequence) + "_" + str(sub.FirstScan)
                           + "_ch" + str(sub.Charge) + "_Integration.csv")
    outplot = os.path.join(out, str(sub.Raw) +
                           "_" + str(sub.Sequence) + "_" + str(sub.FirstScan)
                           + "_ch" + str(sub.Charge) + "_Integration.pdf")
    apex_list, apexonly = Integrate(sub.FirstScan, mz, scanrange, mzrange,
                                    bin_width, mzmlpath, n_workers)
    apex_list.to_csv(outpath, index=False, sep=',', encoding='utf-8')
    
    # Isotopic envelope theoretical distribution (Poisson)
    massconfig = configparser.ConfigParser(inline_comment_prefixes='#')
    massconfig.read(args.config)
    plainseq = ''.join(re.findall("[A-Z]+", sub.Sequence))
    mods = [round(float(i),6) for i in re.findall("\d*\.?\d*", sub.Sequence) if i]
    pos = [int(j)-1 for j, k in enumerate(sub.Sequence) if k.lower() == '[']
    parental = getTheoMH(sub.Charge, plainseq, mods, pos, True, True, massconfig, False)
    mim = sub.MH
    dm = mim - parental
    theomh = parental + dm + (sub.Charge-1)*massconfig.getfloat('Masses', 'm_proton')
    avg_aa = 111.1254 # Dalton
    C13 = 1.003355 # Dalton
    est_C13 = (0.000594 * theomh) - 0.03091
    poisson_df = pd.DataFrame(list(range(0,9)))
    poisson_df.columns = ["n"]
    poisson_df["theomh"] = np.arange(theomh, theomh+8.5*C13, C13)
    poisson_df["theomz"] = poisson_df.theomh / sub.Charge
    poisson_df["Poisson"] = poisson_df.apply(lambda x: poisson.pmf(x.n, est_C13), axis=1)
    poisson_df["cumsum"] = poisson_df.Poisson.cumsum()
    poisson_df = pd.concat([poisson_df[poisson_df["cumsum"]<t_poisson], poisson_df[poisson_df["cumsum"]>=t_poisson].head(1)])
    poisson_df["n_poisson"] = poisson_df.Poisson/poisson_df.Poisson.sum()
    # Select experimental peaks within tolerance
    # apexonly2 = apexonly[apexonly.SUMINT>0]
    apexonly2 = apexonly[apexonly.APEX==True].copy()
    if len(apexonly2) <= 0:
        logging.info("\t\t\t\tNot enough information in the spectrum! 0 apexes found.")
        return
    poisson_df["exp_peak"] = poisson_df.apply(lambda x: min(list(apexonly2.BIN), key=lambda y:abs(y-x.theomz)), axis=1)
    # poisson_df.exp_peak = poisson_df.apply(lambda x: -1 if abs(x.exp_peak-x.theomz)>2*bin_width else x.exp_peak, axis=1)
    poisson_df = poisson_df[poisson_df.exp_peak>=0]
    poisson_df["exp_int"] = poisson_df.apply(lambda x: float(apexonly2[apexonly2.BIN==x.exp_peak].SUMINT), axis=1)
    int_total = poisson_df.exp_int.sum()
    poisson_df["P_compare"] = poisson_df.apply(lambda x: x.n_poisson*int_total, axis=1) # TODO check
    # poisson_df["P_compare"] = poisson_df.apply(lambda x: (x.Poisson/poisson_df.Poisson.max())*apexonly.SUMINT.max(), axis=1)
    # Plots
    PlotIntegration(poisson_df, mz, apex_list, apexonly, outplot)
    return

def main(args):
    '''
    Main function
    '''
    ## PARAMETERS ##    
    srange = int(mass._sections['Parameters']['int_scanrange'])
    drange = float(mass._sections['Parameters']['int_mzrange'])
    bin_width = float(mass._sections['Parameters']['int_binwidth'])
    t_poisson = float(mass._sections['Parameters']['int_poisson_threshold'])
    
    # infile = r"\\Tierra\SC\U_Proteomica\UNIDAD\DatosCrudos\JorgeAlegreCebollada\Glyco_Titin\experiment_Oct21\8870\Titin_glyco.51762.51762.0.dta"
    logging.info("Scan range: ±" + str(srange))
    logging.info("MZ range: ±" + str(drange) + " Th")
    logging.info("Bin width: " + str(bin_width) + " Th")
    logging.info("Reading input table...")
    query = pd.read_table(Path(args.infile), index_col=None, delimiter="\t")
    logging.info("Looking for .mzML files...")
    mzmlfiles = os.listdir(Path(args.raw))
    mzmlfiles = [i[:-5] for i in mzmlfiles if i[-5:].lower()=='.mzml']
    logging.info(str(len(mzmlfiles)) + " mzML files found.")
    missing = list(set(list(query.Raw.unique())) - set(mzmlfiles))
    if missing:
        logging.info("mzML files missing!" + str(missing))
        sys.exit()

    for mzml in mzmlfiles:
        logging.info("Reading file " + str(mzml) + ".mzML...")
        msdata = pyopenms.MSExperiment()
        pyopenms.MzMLFile().load(mzml, msdata)
        ref = []
        for r in msdata.getSpectra():
            if r.getMSLevel() == 1: # FULL
                ref.append(int(r.getNativeID().split(' ')[-1][5:]))
        # GetSubQuery
        sub = query[query.Raw==mzml].copy()
        for i, q in sub.iterrows():
            logging.info("\tIntegrating SCAN=" + str(q.FirstScan) + "...")
            qfull = min(ref, key=lambda x:abs(x-q.FirstScan))
            qpos = ref.index(qfull)
            if qfull > q.FirstScan:
                qpos -= 1
                qfull = ref[qpos]
            s = msdata.getSpectrum(int(qfull)-1)
            mz = s.getPrecursors()[0].getMZ() # Experimental
            ions = pd.DataFrame([s.get_peaks()[0], s.get_peaks()[1]]).T
            ions.columns = ["MZ", "INT"]
            ## Get spectrum and adjacents (srange)
            spectra = pd.DataFrame([q.FirstScan, mz, ions]).T
            spectra.columns = ["SCAN", "MZ", "IONS"]
            
            
        
    for i, q in query.iterrows():
        logging.info("QUERY=" + str(i+1) + " SCAN=" + str(int(q.SCAN)) + " MZ=" + str(q.MZ)+"Th")
        ## JOIN DTAS ##
        qfull = min(list(dtadf.SCAN), key = lambda x : abs(x - q.SCAN))
        dta = dtadf[dtadf.SCAN == qfull].index.values[0]
        dta = dtadf.iloc[dtadf[dtadf.SCAN == qfull].index.values[0]-srange:dtadf[dtadf.SCAN == qfull].index.values[0]+srange+1]
        dtas = pd.concat([pd.read_table(os.path.join(args.dta, f.FILENAME), index_col=None, header=0, delimiter=" ", names=["MZ", "INT"]) for i, f in dta.iterrows()])
        dtas = dtas.sort_values(by="MZ", ignore_index=True)
        
        ## BINNING ##
        bins = list(np.arange(q.MZ-drange, q.MZ+drange, bin_width))
        bins = [round(x, _decimal_places(bin_width)) for x in bins]
        # bins_df = pd.DataFrame([[0]*len(bins)]*len(dtas), columns=bins)
        bins_df = pd.DataFrame([bins]*len(dtas))
        # dtas = pd.concat([dtas, temp_bins], axis=1)
        
        ## CALCULATE INTENSITY ##
        logging.info("Integrating scans...")
        indices, row_series = zip(*bins_df.iterrows())
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            temp_bins_df = list(tqdm(executor.map(InInt, row_series, itertools.repeat(dtas), chunksize=500),
                                     total=len(row_series)))
        bins_df = pd.concat(temp_bins_df, axis=1).T
        # bins_df = bins_df.apply(lambda x: InInt(x, dtas), axis=1)
        # TODO in bins_df do cumsum and divide by nscans (srange*2+1) then pick the apex to save to dtas
        logging.info("Finding peaks...")
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
        # dtas["SUMINT"] = bins_df.sum(axis=1)
        
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
        
        ## PLOTS ##
        fig = plt.figure()
        fig.set_size_inches(20, 15)
        
        ax1 = fig.add_subplot(2,1,1)
        apex_list["COLOR"] = 'darkblue'
        apex_list.loc[apex_list.APEX == True, 'COLOR'] = 'red'
        plt.xlabel("M/Z", fontsize=15)
        plt.ylabel(r'$\sum_{n=0}^{n_{peaks}} Intensity_n \times e^{-\frac{1}{2}\times\frac{(BinMZ-PeakMZ)^2}{\sigma^2}} $', fontsize=15)
        plt.title("Integrated Scans", fontsize=20)
        plt.plot(apex_list.BIN, apex_list.SUMINT, linewidth=1, color="darkblue")
        plt.axvline(x=q.MZ, color='orange', ls="--")
        ax1.annotate(str(q.MZ) + " Th", (q.MZ,max(apex_list.SUMINT)-0.05*max(apex_list.SUMINT)), color='black', fontsize=10, ha="left")

        ax2 = fig.add_subplot(2,1,2)
        plt.xlabel("M/Z", fontsize=15)
        plt.ylabel(r'$\sum_{n=0}^{n_{peaks}} Intensity_n \times e^{-\frac{1}{2}\times\frac{(BinMZ-PeakMZ)^2}{\sigma^2}} $', fontsize=15)
        plt.title("Integrated Scans (apexes only)", fontsize=20)
        plt.plot(apexonly.BIN, apexonly.SUMINT, linewidth=1, color="darkblue")
        plt.axvline(x=q.MZ, color='orange', ls="--")
        ax2.annotate(str(q.MZ) + " Th", (q.MZ,max(apex_list.SUMINT)-0.05*max(apex_list.SUMINT)), color='black', fontsize=10, ha="left")
        # ax2.annotate(r'$\sum_{n=0}^{n_{peaks}} Intensity_n \times e^{-\frac{1}{2}\times\frac{(BinMZ-PeakMZ)^2}{\sigma^2}} $', (391.5,max(apex_list.FINALINT)-0.1*max(apex_list.FINALINT)), color='black', fontsize=20, ha="left")
        # apex_list.astype(str).to_csv(r"output.tsv", index=False, sep='\t', encoding='utf-8')

        ## SAVE OUTPUT ##
        outplot = Path(args.infile[:-4] + "_"+ str(int(q.SCAN)) + "_" + str(q.MZ) + ".pdf")
        fig.savefig(outplot)
        fig.clear()
        plt.close(fig)
        apex_list = apex_list.rename(columns={"SUMINT": "INTENSITY", "BIN": "MZ"})
        apex_list = apex_list.drop("COLOR", axis=1)
        apex_list = apex_list[apex_list.columns.tolist()[-1:] + apex_list.columns.tolist()[:-1]]
        outfile = Path(args.infile[:-4] + "_"+ str(int(q.SCAN)) + "_" + str(q.MZ) + ".tsv")
        apex_list.to_csv(outfile, index=False, sep='\t', encoding='utf-8')
    return

if __name__ == '__main__':

    # multiprocessing.freeze_support()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='VseqExplorer',
        epilog='''
        Example:
            python VseqExplorer.py

        ''')
    
    parser.add_argument('-i',  '--infile', required=True, help='Table of scans to search')
    parser.add_argument('-r',  '--raw', required=True, help='Directory containing .mzML files')
    parser.add_argument('-s',  '--scanrange', default=6, help='± full scans to use')
    parser.add_argument('-m',  '--mzrange', default=2, help='± MZ window to use')
    parser.add_argument('-b',  '--bin', default=0.001, help='Bin width to use')
    parser.add_argument('-p',  '--poisson', default=0.8, help='Poisson coverage threshold')
    parser.add_argument('-o', '--outpath', help='Path to save results')
    parser.add_argument('-w',  '--n_workers', type=int, default=4, help='Number of threads/n_workers (default: %(default)s)')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()
    
    # parse config
    mass = configparser.ConfigParser(inline_comment_prefixes='#')
    mass.read(args.config)
    if args.scanrange is not None:
        mass.set('Parameters', 'scanrange', str(args.scanrange))
    if args.mzrange is not None:
        mass.set('Parameters', 'mzrange', str(args.mzrange))
    if args.binwidth is not None:
        mass.set('Parameters', 'binwidth', str(args.binwidth))
    if args.t_poisson is not None:
        mass.set('Parameters', 't_poisson', str(args.t_poisson))

    # logging debug level. By default, info level
    log_file = outfile = args.infile[:-4] + 'ScanIntegrator_log.txt'
    log_file_debug = outfile = args.infile[:-4] + 'ScanIntegrator_log_debug.txt'
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