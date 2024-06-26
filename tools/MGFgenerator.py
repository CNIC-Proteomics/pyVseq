# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:55:08 2022

@author: alaguillog
"""

import argparse
import logging
import itertools
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import sys
pd.options.mode.chained_assignment = None  # default='warn'

def getTheoMZ(AAs, charge, sequence, series):
    '''    
    Calculate theoretical MZ using the PSM sequence.
    '''
    m_proton = 1.007276
    m_hydrogen = 1.007825
    m_oxygen = 15.994915
    total_aas = charge*m_proton
    if series == "y":
        total_aas += 2*m_hydrogen + m_oxygen
    for i, aa in enumerate(sequence):
        if aa.upper() in AAs:
            total_aas += float(AAs[aa.upper()])
    MH = total_aas - (charge-1)*m_proton
    #MZ = (total_aas + int(charge)*m_proton) / int(charge)
    if charge > 0:
        MZ = total_aas / int(charge)
        return MZ, MH
    else:
        return MH

def makeFrags(seq):
    '''
    Name all fragments.
    '''
    frags = pd.DataFrame(np.nan, index=list(range(0,len(seq)*2)),
                         columns=["series", "by"])
    frags.series = ["b" for i in list(range(1,len(seq)+1))] + ["y" for i in list(range(1,len(seq)+1))[::-1]]
    frags.by = ["b" + str(i) for i in list(range(1,len(seq)+1))] + ["y" + str(i) for i in list(range(1,len(seq)+1))[::-1]]
    ## REGIONS ##
    rsize = int(round(len(seq)/4,0))
    regions = [i+1 for i in range(0,len(seq),rsize)]
    if len(seq) % 2 == 0:
        regions += [i+len(seq)+1 for i in range(0,len(seq),rsize)][::-1]
    else:
        regions += [i+len(seq) for i in range(0,len(seq),rsize)][::-1]
        regions[-1] = regions[-1] + 1
    regions.sort()
    rregions = []
    for r in regions:
        if regions[regions.index(r)]<regions[-1]:
            rregions.append(range(r,regions[regions.index(r)+1],1))
    rregions.append(range(regions[-1],regions[-1]+rsize,1))
    rrregions = []
    counter = 0
    for i in rregions:
        counter += 1
        for j in i:
            rrregions.append(counter)
    frags["region"] = rrregions
      ## SEQUENCES ##
    frags["seq"] = None
    for index, row in frags.iterrows():
        series = row.by[0]
        num = int(row.by[1:])
        if series == "b":
            frags.seq.iloc[index] = seq[0:num]
        if series == "y":
            frags.seq.iloc[index] = seq[len(seq)-num:len(seq)]
    return(frags)

def errorize(subset, error, etype, charge):
    if etype == 0:
        subset.MZ = subset.apply(lambda x: round(x.MZ + ((error*x.MZ)/1000000),6), axis=1)
    else:
        subset.MZ = subset.MZ + (error/charge)
    return(subset)

def noiseMaker(subset, n_peaks):
    noises = pd.DataFrame(np.random.uniform(subset.MZ.min(), subset.MZ.max(), n_peaks), columns=["MZ"]) 
    noises["INT"] = np.random.uniform(10, 100, n_peaks)
    noises["seq"] = "noise"
    noises = round(noises, 6)
    check = subset.copy()
    check = pd.concat([check,noises])
    check.drop_duplicates("MZ", keep="first", inplace=True) # remove duplicates of real peaks
    noises = check[check.seq == "noise"]
    subset = pd.concat([subset,noises])
    return(subset)
    
def makeMGFentry(mzs, i, pepmass, charge):
    mgfentry = []
    mgfentry.append("BEGIN IONS\n")
    mgfentry.append("TITLE=") # TODO put case description here
    mgfentry.append("SCANS=" + str(i) + "\n")
    mgfentry.append("RTINSECONDS=" + str(i) + "\n")
    mgfentry.append("PEPMASS=" + str(pepmass) + "\n")
    mgfentry.append("CHARGE=" + str(charge) +  "+\n")
    mzs.sort_values(by=['MZ'], inplace=True)
    mgfentry = mgfentry + list(mzs.apply(lambda x: ' '.join([str(x.MZ), str(x.INT)]) + "\n", axis=1))
    mgfentry.append("END IONS\n")
    return(mgfentry)

def main(args):
    if not os.path.exists(Path(args.outpath)):
        os.mkdir(Path(args.outpath))
    AAs = {"A":71.037114, "R":156.101111, "N":114.042927, "D":115.026943,
           "C":103.009185, "E":129.042593, "Q":128.058578, "G":57.021464,
           "H":137.058912, "I":113.084064, "L":113.084064, "K":128.094963,
           "M":131.040485, "F":147.068414, "P":97.052764, "S":87.032028,
           "T":101.047679, "U":150.953630, "W":186.079313, "Y":163.063329,
           "V":99.068414, "O":132.089878, "Z":129.042594}
    combos = []
    for i in range(1,9):
        combo = itertools.combinations(range(1,9), i)
        for c in combo:
            combos.append(c)
    sequences = pd.read_csv(args.sequence, header=None)
    intensities = [10, 100, 1000]
    if int(args.error) == 0:
        errors = ppmerrors = [0, 10, 40]
    else:
        errors = mherrors = [0, 0.01, 0.025]
    n_peaks = 100 # number of noise peaks to introduce
    logging.info("Combinations of 8 regions: " + str(len(combos)))
    logging.info("Intensities: " + str(intensities))
    if int(args.error) == 0:
        logging.info("PPM errors: " + str(ppmerrors))
    else:
        logging.info("Da errors: " + str(mherrors))
    logging.info("Noisy peaks: " + str(n_peaks))
    for index, sequence in sequences.iterrows():
        sequence = str(sequence[0])
        logging.info("Generating combinations for sequence: " + sequence)
        frags = makeFrags(sequence)
        frags["MZ"] = frags.apply(lambda x: round(getTheoMZ(AAs, 1, x.seq, x.series)[0],6), axis=1)
        pepmass = round(getTheoMZ(AAs, 1, sequence, "y")[0],6)
        
        mgfdata = []
        scan_number = 1
        # titles = []
        for combo in combos:
            subset = frags[frags.region.isin(combo)]
            for inten in intensities:
                subset["INT"] = inten
                for error in errors:
                    errorset = subset.copy()
                    errorset = errorize(errorset, error, int(args.error), 1)
                    noiseset = noiseMaker(errorset, n_peaks)
                    # Noiseless entry #
                    mgfentry = makeMGFentry(errorset, scan_number, pepmass, 1)
                    mgfentry[1] += " " + sequence + " combo" + str(combo) + " int" + str(inten) + " error" + str(error) + "\n"
                    mgfdata += mgfentry
                    # titles.append([str(combo), str(inten), str(error), "NO"])
                    # Noise entry #
                    mgfentry = makeMGFentry(noiseset, scan_number+1, pepmass, 1)
                    mgfentry[1] += " " + sequence + " combo" + str(combo) + " int" + str(inten) + " error" + str(error) + " with noise" + "\n"
                    mgfdata += mgfentry
                    # titles.append([str(combo), str(inten), str(error), "YES"])
                    scan_number += 2
        # titles = pd.DataFrame(titles)
        # titles.columns=['REGIONS', 'INTENSITY', 'PPM_ERROR', 'NOISE']
        # titles.reset_index(inplace=True)
        # titles = titles.rename(columns = {'index':'SCAN'})
        # titles.SCAN = titles.SCAN + 1
        outfile = os.path.join(Path(args.outpath), sequence + ".mgf")
        logging.info("Writing " + str(outfile))
        with open(outfile, 'a') as f:
            for line in mgfdata:
                f.write(line)
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
    parser.add_argument('-s',  '--sequence', required=True, help='List of aminoacid sequences')
    parser.add_argument('-e',  '--error', required=True, help='0=ppm, 1=Da')
    parser.add_argument('-o',  '--outpath', required=True, help='Path to save MGF files')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()
    
    # logging debug level. By default, info level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            handlers=[logging.StreamHandler()])
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p',
                            handlers=[logging.StreamHandler()])

    # start main function
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    main(args)
    logging.info('end script')