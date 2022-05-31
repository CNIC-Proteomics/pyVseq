import pyopenms
import argparse
import concurrent.futures
import itertools
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

def readMZML(mzmlpath, scan, scanrange):
    exp = pyopenms.MSExperiment()
    pyopenms.MzMLFile().load(mzmlpath, exp)
    # Keep only full scans
    spec = []
    query = np.arange(scan-scanrange,scan+scanrange+1,1)
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
    dtas = readMZML(mzmlpath, scan, scanrange)
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
    return(mz, apex_list, apexonly)

def PlotIntegration(mz, apex_list, apexonly, outplot):
    fig = plt.figure()
    fig.set_size_inches(20, 15)
    
    ax1 = fig.add_subplot(2,1,1)
    apex_list["COLOR"] = 'darkblue'
    apex_list.loc[apex_list.APEX == True, 'COLOR'] = 'red'
    plt.xlabel("M/Z", fontsize=15)
    plt.ylabel(r'$\sum_{n=0}^{n_{peaks}} Intensity_n \times e^{-\frac{1}{2}\times\frac{(BinMZ-PeakMZ)^2}{\sigma^2}} $', fontsize=15)
    plt.title("Integrated Scans", fontsize=20)
    plt.plot(apex_list.BIN, apex_list.SUMINT, linewidth=1, color="darkblue")
    plt.axvline(x=mz, color='orange', ls="--")
    ax1.annotate(str(mz) + " Th", (mz,max(apex_list.SUMINT)-0.05*max(apex_list.SUMINT)), color='black', fontsize=10, ha="left")

    ax2 = fig.add_subplot(2,1,2)
    plt.xlabel("M/Z", fontsize=15)
    plt.ylabel(r'$\sum_{n=0}^{n_{peaks}} Intensity_n \times e^{-\frac{1}{2}\times\frac{(BinMZ-PeakMZ)^2}{\sigma^2}} $', fontsize=15)
    plt.title("Integrated Scans (apexes only)", fontsize=20)
    plt.plot(apexonly.BIN, apexonly.SUMINT, linewidth=1, color="darkblue")
    plt.axvline(x=mz, color='orange', ls="--")
    ax2.annotate(str(mz) + " Th", (mz,max(apex_list.SUMINT)-0.05*max(apex_list.SUMINT)), color='black', fontsize=10, ha="left")
    
    fig.savefig(outplot)
    fig.clear()
    plt.close(fig)
    return

def main(args):
    '''
    Main function
    '''
    ## PARAMETERS ##
    srange = int(args.scanrange)
    drange = float(args.mzrange)
    bin_width = float(args.bin)
    
    # infile = r"\\Tierra\SC\U_Proteomica\UNIDAD\DatosCrudos\JorgeAlegreCebollada\Glyco_Titin\experiment_Oct21\8870\Titin_glyco.51762.51762.0.dta"
    logging.info("Scan range: ±" + str(srange))
    logging.info("MZ range: ±" + str(drange) + " Th")
    logging.info("Bin width: " + str(bin_width) + " Th")
    logging.info("Reading input table...")
    query = pd.read_table(Path(args.infile), index_col=None, header=0, delimiter="\t", names=["SCAN", "MZ"])
    logging.info("Looking for .dta files...")
    dtafiles = os.listdir(Path(args.dta))
    dtafiles = [i for i in dtafiles if i[-6:]=='.0.dta'] # Full Scans
    logging.info(str(len(dtafiles)) + " Full Scan .dta files found.")
    dtadf = pd.DataFrame(dtafiles)
    dtadf.columns = ["FILENAME"]
    dtadf["SCAN"] = dtadf.FILENAME.str.split(".").str[-3].astype(int)
    dtadf = dtadf.sort_values(by="SCAN", ignore_index=True)

    for i, q in query.iterrows():
        logging.info("QUERY=" + str(i+1) + " SCAN=" + str(int(q.SCAN)) + " MZ=" + str(q.MZ)+"Th")
        ## JOIN DTAS ##
        qfull = min(list(dtadf.SCAN), key = lambda x : abs(x - q.SCAN))
        dta = dtadf[dtadf.SCAN == qfull].index.values[0]
        dta = dtadf.iloc[dtadf[dtadf.SCAN == qfull].index.values[0]-srange:dtadf[dtadf.SCAN == qfull].index.values[0]+srange+1]
        # dtas = [[i, pd.read_table(os.path.join(args.infile, f.FILENAME), index_col=None, header=0, delimiter=" ", names=["MZ", "INT"])] for i, f in dta.iterrows()]
        dtas = pd.concat([pd.read_table(os.path.join(args.dta, f.FILENAME), index_col=None, header=0, delimiter=" ", names=["MZ", "INT"]) for i, f in dta.iterrows()])
        dtas = dtas.sort_values(by="MZ", ignore_index=True)
        # dtas["PW"] = 10**-6 * dtas.MZ**1.5027
        # dtas["SIGMA"] = 40**-7 * dtas.MZ**1.5027
        
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
    parser.add_argument('-d',  '--dta', required=True, help='Directory containing .DTA files')
    parser.add_argument('-s',  '--scanrange', default=6, help='± full scans to use')
    parser.add_argument('-m',  '--mzrange', default=2, help='± MZ window to use')
    parser.add_argument('-b',  '--bin', default=0.001, help='Bin width to use')
    parser.add_argument('-o', '--outpath', help='Path to save results')
    parser.add_argument('-w',  '--n_workers', type=int, default=4, help='Number of threads/n_workers (default: %(default)s)')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()

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