# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:42:30 2022

@author: alaguillog
"""

# import modules
import argparse
from bisect import bisect_left
import concurrent.futures
import configparser
import glob
import io
import itertools
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyopenms
from PyPDF2 import PdfMerger
import re
import shutup
shutup.please()
import sys
from tqdm import tqdm
# import custom modules
from Vseq import doVseq
from tools.Hyperscore import locateScan
# module config
matplotlib.use('pdf')
pd.options.mode.chained_assignment = None  # default='warn'

def read_csv_with_progress(file_path, sep, mode="mgf"):
    chunk_size = 50000  # Number of lines to read in each iteration # TODO: add to INI
    # Get the total number of lines in the CSV file
    # logging.info("Calculating average line length + getting file size")
    counter = 0
    total_length = 0
    num_to_sample = 10
    for line in open(file_path, 'r'):
        counter += 1
        if counter > 1:
            total_length += len(line)
        if counter == num_to_sample + 1:
            break
    file_size = os.path.getsize(file_path)
    avg_line_length = total_length / num_to_sample
    avg_number_of_lines = int(file_size / avg_line_length)
    chunks = []
    with tqdm(total=avg_number_of_lines, desc='Reading MGF') as pbar:
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, sep=sep, header=None):
            chunks.append(chunk)
            pbar.update(chunk.shape[0])
    logging.info("Joining chunks...")
    df = pd.concat(chunks, ignore_index=True)
    return(df)

def read_mzml_with_progress(inputfile):
    ondisc_exp = pyopenms.OnDiscMSExperiment()
    ondisc_exp.openFile(inputfile)
    # mgf = pyopenms.MSExperiment()
    # for i in tqdm(range(ondisc_exp.getNrSpectra()), desc="Loading spectra"):
    #     spectrum = ondisc_exp.getSpectrum(i)
    #     mgf.addSpectrum(spectrum)
    mgf = ondisc_exp.getMetaData()
    return(mgf, ondisc_exp)

def checkMGFs(mgfs, mgflist):
    checklist = list(mgfs.groups.keys())
    checklist = [i + ".mgf" for i in checklist]
    stock = [os.path.basename(j) for j in mgflist]
    missing = 0
    for i in checklist:
        if i not in stock:
            logging.info("Missing path for file: " + str(i))
            missing += 1
    if missing == 0:
        return(True)
    else:
        return(False)
    
def makeOutpath(outpath3, prot, sequence, firstscan, charge, cand):
    outplot = os.path.join(str(outpath3), str(prot) + "_" + str(sequence) +
                           "_" + str(firstscan) + "_ch" + str(charge) +
                           "_cand" + str(cand) + ".pdf")
    if len(str(outplot)) >= 250:
        outplot = os.path.join(str(outpath3), str(prot) + "_" +
                               str(sequence)[:len(str(sequence))//2] +
                               "_trunc_" + str(firstscan) + "_ch" + str(charge) +
                               "_cand" + str(cand) + ".pdf")
        counter = 0
        while os.path.isfile(outplot):
            counter += 1
            outplot = os.path.join(str(outpath3), str(prot) + "_" +
                                   str(sequence)[:len(str(sequence))//2] +
                                   "_trunc_" + str(firstscan) + "_ch" + str(charge) +
                                   "_cand" + str(cand) + "_" + str(counter) + ".pdf")
    return(outplot)

def getTquery(fr_ns, mode, rawpath, int_perc):
    if mode == "mgf":
        fr_ns2 = fr_ns.copy()
        flag = True
        # Check if index exists
        if os.path.exists(os.path.join(os.path.split(rawpath)[0], os.path.split(rawpath)[1].split(".")[0]+"_index.tsv")):
            logging.info("Existing index found")
            tindex = pd.read_csv(os.path.join(os.path.split(rawpath)[0], os.path.split(rawpath)[1].split(".")[0]+"_index.tsv"), sep="\t")
            squery = [str(i) for i in list(tindex.squery)]
            sindex = np.array(tindex.sindex)
            eindex = np.array(tindex.eindex)
        else:
            fr_ns = fr_ns.to_numpy()
            fr_ns = fr_ns.flatten()
            flag = False
            sindex = np.array([i for i, si in enumerate(fr_ns) if si.startswith('SCANS=')])
            eindex = np.array([i for i, si in enumerate(fr_ns) if si.startswith('END IONS')])
            squery = [i.replace("SCANS=","") for i in fr_ns[sindex]]
        if os.path.exists(os.path.join(os.path.split(rawpath)[0], os.path.split(rawpath)[1].split(".")[0]+"_tquery.tsv")):
            logging.info("Existing tquery found")
            tquery = pd.read_csv(os.path.join(os.path.split(rawpath)[0], os.path.split(rawpath)[1].split(".")[0]+"_tquery.tsv"), sep="\t")
        else:
            if flag:
                fr_ns = fr_ns.to_numpy()
                fr_ns = fr_ns.flatten()
            mquery = [i.replace("PEPMASS=","") for i in fr_ns[sindex-3]]
            cquery = [i.replace("CHARGE=","") for i in fr_ns[sindex-2]]
            rquery = [i.replace("RTINSECONDS=","") for i in fr_ns[sindex-1]]
            tquery = pd.DataFrame([squery, mquery, cquery, rquery]).T
            tquery.columns = ["SCANS", "PEPMASS", "CHARGE", "RT"]
            try:
                tquery[['MZ','INT']] = tquery.PEPMASS.str.split(" ",expand=True,)
            except ValueError:
                tquery['MZ'] = tquery.PEPMASS
            tquery['CHARGE'] = tquery.CHARGE.str[:-1]
            tquery = tquery.drop("PEPMASS", axis=1)
            index1 = (fr_ns2.to_numpy() == 'BEGIN IONS').flatten()
            index2 = (fr_ns2.to_numpy() == 'END IONS').flatten()
            index3 = np.array([i for i in range(0,len(fr_ns2))])
            index4 = index3[index1]+6
            index5 = index3[index2]
            fr_ns2 = fr_ns2.to_numpy()
            allspectra = [fr_ns2[index4[i]:index5[i]] for i in range(0,len(index4))]
            allspectra = [[j[0].split(' ') for j in i] for i in allspectra]
            allspectra = [np.transpose(np.array([[float(k) for k in j] for j in i])) for i in allspectra]
            # Normalize intensity
            allspectra0 = [np.array(i[0]) for i in allspectra]
            allspectra1 = [np.array(i[1]) for i in allspectra]
            allspectra1 = [(allspectra1[i]/max(allspectra1[i]))*100 for i in range(len(allspectra))]
            # Filter by ratio
            if int_perc > 0:
                cutoff1 = [i/max(i) >= int_perc for i in allspectra1]
                allspectra0 = [allspectra0[i][cutoff1[i]] for i in range(len(allspectra))]
                allspectra1 = [allspectra1[i][cutoff1[i]] for i in range(len(allspectra))]
            allspectra = [np.array([allspectra0[i], allspectra1[i]]) for i in range(len(allspectra))]
            # Duplicate m/z measurement
            check = [len(np.unique(i)) != len(i) for i in allspectra0]
            for i in range(len(check)):
                if check[i] == True:
                    temp = allspectra[i].copy()
                    temp = pd.DataFrame(temp).T
                    temp = temp[temp.groupby(0)[1].rank(ascending=False)<2]
                    temp.drop_duplicates(subset=0, inplace=True)
                    allspectra[i] = np.array(temp.T)
            tquery["SPECTRUM"] = allspectra
        try:
            tquery[['SCANS', 'CHARGE', 'RT', 'MZ', 'INT']] = tquery[['SCANS', 'CHARGE', 'RT', 'MZ', 'INT']].apply(pd.to_numeric)
        except KeyError:
            tquery[['SCANS', 'CHARGE', 'RT', 'MZ']] = tquery[['SCANS', 'CHARGE', 'RT', 'MZ']].apply(pd.to_numeric)
        if not os.path.exists(os.path.join(os.path.split(rawpath)[0], os.path.split(rawpath)[1].split(".")[0]+"_tquery.tsv")):
            tquery.to_csv(os.path.join(os.path.split(rawpath)[0],
                                       os.path.split(rawpath)[1].split(".")[0]+"_tquery.tsv"),
                          index=False, sep='\t', encoding='utf-8')
        if not os.path.exists(os.path.join(os.path.split(rawpath)[0], os.path.split(rawpath)[1].split(".")[0]+"_index.tsv")):  
            tindex = pd.DataFrame([squery, sindex, eindex], index=["squery","sindex","eindex"]).T
            tindex.to_csv(os.path.join(os.path.split(rawpath)[0],
                                       os.path.split(rawpath)[1].split(".")[0]+"_index.tsv"),
                          index=False, sep='\t', encoding='utf-8')
    elif mode == "mzml":
        spectra = fr_ns.getSpectra()
        # spectra_n = [int(s.getNativeID().split("=")[-1]) for s in spectra]
        # tquery = []
        # for s in spectra:
        #     if s.getMSLevel() == 2:
        #         df = pd.DataFrame([int(s.getNativeID().split(' ')[-1][5:]), # Scan
        #                   s.getPrecursors()[0].getCharge(), # Precursor Charge
        #                   s.getRT(), # Precursor Retention Time
        #                   s.getPrecursors()[0].getMZ(), # Precursor MZ
        #                   s.getPrecursors()[0].getIntensity()]).T # Precursor Intensity
        #         df.columns = ["SCANS", "CHARGE", "RT", "MZ", "INT"]
        #         tquery.append(df)
        # tquery = pd.concat(tquery)
        rows = []
        for s in spectra:
            if s.getMSLevel() == 2:
                native_id = s.getNativeID()
                scan = int(native_id.rsplit('scan=', 1)[-1])
                precursor = s.getPrecursors()[0]
        
                rows.append([
                    scan,
                    precursor.getCharge(),
                    s.getRT(),
                    precursor.getMZ(),
                    precursor.getIntensity()
                ])
        tquery = pd.DataFrame(rows, columns=["SCANS", "CHARGE", "RT", "MZ", "INT"])
        tquery = tquery.apply(pd.to_numeric)
        tquery.SCANS = tquery.SCANS.astype(int)
        tquery.CHARGE = tquery.CHARGE.astype(int)
        # tquery["SPECTRUM"] = tquery.apply(lambda x: locateScan(x.SCANS, mode, fr_ns, spectra, spectra_n, 0, int_perc),
        #                                   axis=1)
        squery = sindex = eindex = 0
        if not os.path.exists(os.path.join(os.path.split(rawpath)[0], os.path.split(rawpath)[1].split(".")[0]+"_tquery.tsv")):
            tquery.to_csv(os.path.join(os.path.split(rawpath)[0],
                                       os.path.split(rawpath)[1].split(".")[0]+"_tquery.tsv"),
                          index=False, sep='\t', encoding='utf-8')
        if not os.path.exists(os.path.join(os.path.split(rawpath)[0], os.path.split(rawpath)[1].split(".")[0]+"_index.tsv")):  
            tindex = pd.DataFrame([squery, sindex, eindex], index=["squery","sindex","eindex"]).T
            tindex.to_csv(os.path.join(os.path.split(rawpath)[0],
                                       os.path.split(rawpath)[1].split(".")[0]+"_index.tsv"),
                          index=False, sep='\t', encoding='utf-8')
    return tquery, squery, sindex, eindex

def getOffset(fr_ns):
    def _check(can):
        try:
            list(map(float, can))
            return 0
        except ValueError:
            return 1
    fs = fr_ns[fr_ns[0].str.contains("SCANS=")].iloc[0].name
    ls = fr_ns[fr_ns[0].str.contains("END IONS")].iloc[0].name
    for i in range(fs, ls+1):
        can = fr_ns[0].iloc[i].split(" ")
        if len(can)==2 and _check(can)==0:
            fi = i
            break
    index_offset = fi - fs
    return index_offset

def getTheoMZH(charge, sequence, mods, pos, nt, ct, mass):
    '''    
    Calculate theoretical MZ using the PSM sequence.
    '''
    AAs = dict(mass._sections['Aminoacids'])
    MODs = dict(mass._sections['Fixed Modifications'])
    m_proton = mass.getfloat('Masses', 'm_proton')
    m_hydrogen = mass.getfloat('Masses', 'm_hydrogen')
    m_oxygen = mass.getfloat('Masses', 'm_oxygen')
    total_aas = 2*m_hydrogen + m_oxygen
    total_aas += charge*m_proton
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
    #MZ = (total_aas + int(charge)*m_proton) / int(charge)
    if charge > 0:
        MZ = total_aas / int(charge)
        return MZ, MH
    else:
        return MH
    
def takeClosest(myNumber, myList):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before
    
def _parallelProcessSpectrum(x, parlist, pbar):
    '''
    Get experimental spectrum.
    '''
    fr_ns = parlist[0]
    index_offset = parlist[1]
    scan = int(x.SCANS)
    index2 = parlist[2]
    mode = parlist[3]
    ftol = parlist[4]
    int_perc = parlist[5]
    squery = parlist[6]
    sindex = parlist[7]
    eindex = parlist[8]
    if mode == "mgf":
        place = squery.index(str(scan))
        ions = fr_ns.iloc[sindex[place]+1:eindex[place]]
        ions[['MZ','INT']] = ions[0].str.split(" ",expand=True,)
        ions = ions.drop(ions.columns[0], axis=1)
        ions = ions.apply(pd.to_numeric)
    elif mode == "mzml":
        s = fr_ns.getSpectrum(scan-1)
        ions = pd.DataFrame([s.get_peaks()[0], s.get_peaks()[1]]).T
        ions.columns = ["MZ", "INT"]
    ions.reset_index(drop=True)
    # DIA: Filter by intensity ratio
    ions = ions[ions.INT>=ions.INT.max()*int_perc]
    pbar.update(1)
    return(ions)

def _parallelExpSpectrum(x, parlist):
    relist = expSpectrum(parlist[0], parlist[1], x.FirstScan, parlist[2], parlist[3], parlist[4], parlist[5],
                     parlist[6], parlist[7], parlist[8], parlist[9], parlist[10], x.Diagnostic_data)
    return(relist)
    
def expSpectrum(fr_ns, index_offset, scan, index2, mode, frags_diag, ftol,
                int_perc, squery=0, sindex=0, eindex=0, preprocessmsdata=False,
                diagnostic_data=0):
    '''
    Get experimental spectrum.
    '''
    if preprocessmsdata:
        ions = diagnostic_data
    else:
        if mode == "mgf":
            place = squery.index(str(scan))
            ions = fr_ns.iloc[sindex[place]+1:eindex[place]]
            # index1 = fr_ns.loc[fr_ns[0]=='SCANS='+str(scan)].index[0] + index_offset
            # index3 = np.where(index2)[0]
            # index3 = index3[np.searchsorted(index3,[index1,],side='right')[0]]
            # ions = fr_ns.iloc[index1:index3,:]
            # ions[0] = ions[0].str.strip()
            ions[['MZ','INT']] = ions[0].str.split(" ",expand=True,)
            ions = ions.drop(ions.columns[0], axis=1)
            ions = ions.apply(pd.to_numeric)
        elif mode == "mzml":
            s = fr_ns.getSpectrum(scan-1)
            ions = pd.DataFrame([s.get_peaks()[0], s.get_peaks()[1]]).T
            ions.columns = ["MZ", "INT"]
        ions.reset_index(drop=True)
        # DIA: Filter by intensity ratio
        ions = ions[ions.INT>=ions.INT.max()*int_perc]
    # DIA: Filter by number diagnostic ions (if there is a tie, use total intensity)
    frags_diag = list(frags_diag)
    ions["FRAG"] = ions.MZ.apply(takeClosest, myList=frags_diag)
    ions["PPM"] = (((ions.MZ - ions.FRAG)/ions.FRAG)*1000000).abs()
    nions = len(ions[ions.PPM<=ftol])
    iions = ions[ions.PPM<=ftol].INT.sum()
    return(nions, iions)

def theoSpectrum(seq, mods, pos, len_ions, dm, mass):
    '''
    Prepare theoretical fragment matrix.

    '''
    m_hydrogen = mass.getfloat('Masses', 'm_hydrogen')
    m_oxygen = mass.getfloat('Masses', 'm_oxygen')
    ## Y SERIES ##
    #ipar = list(range(1,len(seq)))
    outy = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        yn = list(seq[i:])
        if i > 0: nt = False
        else: nt = True
        fragy = getTheoMZH(0,yn,mods,pos,nt,True,mass) + dm
        outy[i:] = fragy
        
    ## B SERIES ##
    outb = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        bn = list(seq[::-1][i:])
        if i > 0: ct = False
        else: ct = True
        fragb = getTheoMZH(0,bn,mods,pos,True,ct,mass) - 2*m_hydrogen - m_oxygen + dm
        outb[i:] = fragb
    
    ## FRAGMENT MATRIX ##
    yions = outy.T
    bions = outb.iloc[::-1].T
    spec = pd.concat([bions, yions], axis=1)
    spec.columns = range(spec.columns.size)
    spec.reset_index(inplace=True, drop=True)
    return(spec)

def eScore(ppmfinal, int2, err):
    int2.reset_index(inplace=True, drop=True)
    ppmfinal["minv"] = ppmfinal.min(axis=1)
    qscore = pd.DataFrame(ppmfinal["minv"])
    qscore[qscore > err] = 0
    qscore["INT"] = int2
    qscoreFALSE = pd.DataFrame([[21,21],[21,21]])
    qscore = qscore[(qscore>0).all(1)]
    if qscore.shape[0] == 2:
        qscore = qscoreFALSE
    escore = (qscore.INT/1000000).sum()
    return(escore)

def errorMatrix(mz, theo_spec):
    '''
    Prepare ppm-error and experimental mass matrices.
    '''
    m_proton = mass.getfloat('Masses', 'm_proton')
    exp = pd.DataFrame(np.tile(pd.DataFrame(mz), (1, len(theo_spec.columns)))) 
    
    ## EXPERIMENTAL MASSES FOR CHARGE 2 ##
    mzs2 = pd.DataFrame(mz)*2 - m_proton
    mzs2 = pd.DataFrame(np.tile(pd.DataFrame(mzs2), (1, len(exp.columns)))) 
    
    ## EXPERIMENTAL MASSES FOR CHARGE 3 ##
    mzs3 = pd.DataFrame(mz)*3 - m_proton*2 # WRONG
    mzs3 = pd.DataFrame(np.tile(pd.DataFrame(mzs3), (1, len(exp.columns)))) 
    
    ## PPM ERRORS ##
    terrors = (((exp - theo_spec)/theo_spec)*1000000).abs()
    terrors2 =(((mzs2 - theo_spec)/theo_spec)*1000000).abs()
    terrors3 = (((mzs3 - theo_spec)/theo_spec)*1000000).abs()
    return(terrors, terrors2, terrors3, exp)

def _parallelGetIons(x, parlist, pbar):
    relist = getIons(x, parlist[0], parlist[1], parlist[2], parlist[3], parlist[4], parlist[5],
                     parlist[6], parlist[7], parlist[8], parlist[9], parlist[10], parlist[11],
                     parlist[12], parlist[13], parlist[14], parlist[15], parlist[16], parlist[17],
                     parlist[18], parlist[19], parlist[20], parlist[21])
    pbar.update(1)
    return([relist, x.FirstScan])

def getIons(x, tquery, mgf, index2, min_dm, min_match, ftol, outpath,
            standalone, massconfig, dograph, min_hscore, ppm_plot,
            index_offset, mode, int_perc, squery, sindex, eindex,
            spectra, spectra_n, fsort_by, od):
    ions_exp = []
    b_ions = []
    y_ions = []
    vscore, escore, hscore, nions, bions, yions, ppmfinal, frags = doVseq(mode, index_offset, x, tquery, mgf, index2, spectra,
                                                                          spectra_n, min_dm, min_match, ftol, outpath, standalone,
                                                                          massconfig, dograph, 0, ppm_plot, int_perc,
                                                                          squery, sindex, eindex, sortby=fsort_by,
                                                                          od=od)
    ppmfinal = ppmfinal.drop("minv", axis=1)
    ppmfinal.columns = frags.by
    ppmfinal[ppmfinal>ftol] = 0
    ppmfinal = ppmfinal.astype('bool').T
    ppmfinal = ppmfinal[(ppmfinal == True).any(axis=1)]
    if ppmfinal.any().any():
        b_ions = b_ions + [x for x in list(ppmfinal.index.values) if "b" in x]
        y_ions = y_ions + [x for x in list(ppmfinal.index.values) if "y" in x]
    # ions_matched = len(b_ions) + len(y_ions)
    return([nions, ions_exp, bions, yions, vscore, escore, hscore])

def plotRT(subtquery, outpath, prot, charge, startRT, endRT):
    titleseq = str(subtquery.Sequence.loc[0])
    titledm = str(round(subtquery.DeltaMass.loc[0],6))
    outgraph = str(prot) + "_" + titleseq + "_M" + str(subtquery.MH.loc[0]) + "_ch" + str(charge) + "_RT_plots.pdf"
    ## DUMMY RT VALUES ##  
    subtquery.sort_values(by=['RetentionTime'], inplace=True)
    subtquery.RetentionTime = subtquery.RetentionTime / 60
    subtquery.reset_index(drop=True, inplace=True)
    for index, row in subtquery.iterrows():
        before = pd.Series([0]*row.shape[0], index=row.index)
        after = pd.Series([0]*row.shape[0], index=row.index)
        before.RetentionTime = row.RetentionTime - 0.1
        after.RetentionTime = row.RetentionTime + 0.1
        before.Sequence = row.Sequence
        after.Sequence = row.Sequence
        before.DeltaMass = row.DeltaMass
        after.DeltaMass = row.DeltaMass
        subtquery.loc[subtquery.shape[0]] = before
        subtquery.loc[subtquery.shape[0]] = after
    subtquery.sort_values(by=['RetentionTime'], inplace=True)
    subtquery.reset_index(drop=True, inplace=True)
    ## PLOTS ##
    fig = plt.figure()
    fig.set_size_inches(15, 20)
    fig.suptitle(titleseq + '+' + titledm, fontsize=30)
    ## RT vs E-SCORE ##
    ax1 = fig.add_subplot(3,1,1)
    plt.xlim(startRT, endRT)
    plt.xlabel("Retention Time (minutes)", fontsize=15)
    plt.ylabel("E-score", fontsize=15)
    plt.plot(subtquery.RetentionTime, subtquery.e_score, linewidth=1, color="darkblue")
    ## RT vs MATCHED IONS ##
    ax2 = fig.add_subplot(3,1,2)
    plt.xlim(startRT, endRT)
    plt.xlabel("Retention Time (minutes)", fontsize=15)
    plt.ylabel("Matched Ions", fontsize=15)
    plt.plot(subtquery.RetentionTime, subtquery.ions_matched, linewidth=1, color="darkblue")
    ## RT vs MATCHED IONS * E-SCORE ##
    ax3 = fig.add_subplot(3,1,3)
    plt.xlim(startRT, endRT)
    plt.xlabel("Retention Time (minutes)", fontsize=15)
    plt.ylabel("Matched Ions * E-score", fontsize=15)
    plt.plot(subtquery.RetentionTime, subtquery.ions_matched*subtquery.e_score, linewidth=1, color="darkblue")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(Path(outpath), outgraph))
    fig.clear()
    plt.close(fig)
    return

def processSeqTable(query, raw, tquery, ptol, ftol, fsort_by, bestn, fullprot,
                    prot, mgf, index2, min_dm, min_match, min_hscore, outpath3,
                    mass, n_workers, parallelize, ppm_plot, outfile, index_offset,
                    mode, int_perc, m_proton, diag_ions, keep_n, spectra, spectra_n,
                    od):
    # logging.info("\tExploring sequence " + str(query.Sequence) + ", "
    #              + str(query.MH) + " Th, Charge "
    #              + str(query.Charge))
    ## SEQUENCE ##
    query.Sequence = str(query.Sequence).upper()
    plainseq = ''.join(re.findall("[A-Z]+", query.Sequence))
    mods = [round(float(i),6) for i in re.findall("\d*\.?\d*", query.Sequence) if i]
    pos = [int(j)-1 for j, k in enumerate(query.Sequence) if k.lower() == '[']
    acc_pos = 0
    for i, p in enumerate(pos):
        if i > 0:
            pos[i] = p - 2 - len(str(mods[i-1])) - acc_pos
            acc_pos += len(str(mods[i-1])) + 2
    ## MZ and MH ##
    query['expMH'] = query.MH
    query['expMZ'] = round((query.expMH + (m_proton * (query.Charge-1))) / query.Charge, 6)
    query['MZ'] = getTheoMZH(query.Charge, plainseq, mods, pos, True, True, mass)[0]
    query['MH'] = getTheoMZH(query.Charge, plainseq, mods, pos, True, True, mass)[1]
    ## DM ##
    mim = query.expMH
    dm = mim - query.MH
    dm_theo_spec = theoSpectrum(plainseq, mods, pos, len(plainseq), dm, mass).loc[0]
    frags = ["b" + str(i) for i in list(range(1,len(plainseq)+1))] + ["y" + str(i) for i in list(range(1,len(plainseq)+1))[::-1]]
    # frags_diag = [i for i in frags if i[0]=="b"][len([i for i in frags if i[0]=="b"])//2-diag_ions//2:len([i for i in frags if i[0]=="b"])//2-diag_ions//2+diag_ions] + [i for i in frags if i[0]=="y"][len([i for i in frags if i[0]=="y"])//2-diag_ions//2:len([i for i in frags if i[0]=="y"])//2-diag_ions//2+diag_ions]
    frags_diag = [i for i in frags if i[0]=="b" and int(i[1])>=diag_ions]+[i for i in frags if i[0]=="y" and int(i[1])>=diag_ions]
    dm_theo_spec.index = frags
    if keep_n > 0:
        frags_diag = dm_theo_spec[frags_diag]
        frags_diag = (frags_diag+(m_proton*query.Charge))/query.Charge
    ## TOLERANCE ##
    upper = query.expMZ + ptol
    lower = query.expMZ - ptol
    ## OPERATIONS ##
    # subtquery = tquery[(tquery.CHARGE==query.Charge) & (tquery.MZ>=lower) & (tquery.MZ<=upper)]
    subtquery = tquery[(tquery.MZ>=lower) & (tquery.MZ<=upper)]
    # logging.info("\t" + str(subtquery.shape[0]) + " scans found within ±"
    #              + str(ptol) + " Th")
    if subtquery.shape[0] == 0:
        return # TODO can this be nothing or do we need a dummy DF
    # logging.info("\tComparing...")
    subtquery['Protein'] = prot
    subtquery['Sequence'] = query.Sequence
    subtquery['MH'] = query.expMH
    subtquery['DeltaMassLabel'] = query.DeltaMassLabel
    subtquery['DeltaMass'] = dm
    subtquery.rename(columns={'SCANS': 'FirstScan', 'CHARGE': 'Charge', 'RT':'RetentionTime'}, inplace=True)
    subtquery["RawCharge"] = subtquery.Charge
    subtquery.Charge = query.Charge
    parlist = [tquery, mgf, index2, min_dm, min_match, ftol, Path(outpath3), False, mass, False, min_hscore, ppm_plot, index_offset, mode, int_perc, spectra, spectra_n, fsort_by, od]
    # # DIA: Filter by diagnostic ions
    # logging.info("Filtering by diagnostic ions...")
    if keep_n > 0:
        subtquery['temp_diagnostic'] = subtquery.apply(lambda x: expSpectrum(mgf, index_offset, x.FirstScan, index2, mode, frags_diag, ftol, int_perc), axis=1)
        subtquery['Diagnostic_Ions'] = pd.DataFrame(subtquery.temp_diagnostic.tolist()).iloc[:, 0]. tolist()
        subtquery['Diagnostic_Intensity'] = pd.DataFrame(subtquery.temp_diagnostic.tolist()).iloc[:, 1]. tolist()
        subtquery = subtquery.drop('temp_diagnostic', axis = 1)
        subtquery = subtquery.nlargest(keep_n, ['Diagnostic_Ions', 'Diagnostic_Intensity'])
        subtquery = subtquery.sort_index()
    if parallelize == "both":
        indices, rowSeries = zip(*subtquery.iterrows())
        rowSeries = list(rowSeries)
        tqdm.pandas(position=0, leave=True)
        chunks = 100
        if len(rowSeries) <= 500:
            chunks = 50
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            vseqs = list(executor.map(_parallelGetIons, rowSeries, itertools.repeat(parlist), chunksize=chunks))
        subtquery['templist'] = vseqs
    else:
        subtquery['templist'] = subtquery.apply(lambda x: getIons(x,
                                                                  tquery,
                                                                  mgf,
                                                                  index2,
                                                                  min_dm,
                                                                  min_match,
                                                                  ftol,
                                                                  Path(outpath3),
                                                                  False,
                                                                  mass,
                                                                  False,
                                                                  min_hscore,
                                                                  ppm_plot,
                                                                  index_offset,
                                                                  mode,
                                                                  int_perc,
                                                                  od)
                                                #if x.b_series and x.y_series else 0
                                                , axis = 1)
    subtquery['ions_matched'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 0]. tolist()
    #subtquery['ions_exp'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 1]. tolist()
    subtquery['ions_total'] = len(plainseq) * 2
    subtquery['b_series'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 2]. tolist()
    subtquery['y_series'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 3]. tolist()
    subtquery['Raw'] = str(raw)
    subtquery['v_score'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 4]. tolist()
    subtquery['e_score'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 5]. tolist()
    subtquery['hyperscore'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 6]. tolist()
    subtquery['product'] = subtquery['ions_matched'] * subtquery['e_score']
    subtquery = subtquery.drop('templist', axis = 1)
    ## SORT BY ions_matched ##
    try:
        subtquery.sort_values(by=['INT'], inplace=True, ascending=False)
        subtquery.sort_values(by=[fsort_by], inplace=True, ascending=False)
    except KeyError:
        subtquery.sort_values(by=[fsort_by], inplace=True, ascending=False)
    subtquery.sort_values(by=[fsort_by], inplace=True, ascending=False)
    subtquery.reset_index(drop=True, inplace=True)
    f_subtquery = subtquery.iloc[0:bestn]
    f_subtquery.reset_index(drop=True, inplace=True)
    f_subtquery["outpath"] = str(outpath3) + "/" + str(prot) + "_" + f_subtquery.Sequence.astype(str) + "_" + f_subtquery.FirstScan.astype(str) + "_ch" + f_subtquery.Charge.astype(str) + "_cand" + (f_subtquery.index.values+1).astype(str) + ".pdf"
    if f_subtquery.shape[0] > 0:
        # logging.info("\tRunning Vseq on " + str(bestn) + " best candidates...")
        f_subtquery = f_subtquery[f_subtquery[fsort_by]>min_hscore]
        if not os.path.exists(Path(outpath3)):
            os.mkdir(Path(outpath3))
        f_subtquery.apply(lambda x: doVseq(mode,
                                           index_offset,
                                           x,
                                           tquery,
                                           mgf,
                                           index2,
                                           spectra,
                                           spectra_n,
                                           min_dm,
                                           min_match,
                                           ftol,
                                           Path(x.outpath),
                                           False,
                                           mass,
                                           True,
                                           0,
                                           ppm_plot,
                                           od=od), axis = 1)
    allpagelist = list(map(Path, list(f_subtquery["outpath"])))
    pagelist = []
    for f in allpagelist:
        if os.path.isfile(f):
            pagelist.append(f)
    merger = PdfMerger()
    for page in pagelist:
        merger.append(io.FileIO(page,"rb"))
    # logging.info("\tFound " + str(len(pagelist)) + " candidates with v-score > " + str(min_hscore))
    if len(pagelist) > 0:
        outmerge = os.path.join(Path(outpath3), str(prot) + "_" + str(query.Sequence) + "_M" + str(round(query.expMH,4)) + "_ch" + str(query.Charge) + "_best" + str(bestn) + ".pdf")
        with open(outmerge, 'wb') as f:
            merger.write(f)
        for page in pagelist:
            os.remove(page)
        #if len(x.b_series)>1 and len(x.y_series)>1 else logging.info("\t\tSkipping one candidate with empty fragmentation series...")
        ## PLOT RT vs E-SCORE and MATCHED IONS ##
        subtquery.loc[len(subtquery)] = 0
        subtquery.iloc[-1].RetentionTime = tquery.iloc[0].RT/60
        subtquery.loc[len(subtquery)] = 0
        subtquery.iloc[-1].RetentionTime = tquery.iloc[-1].RT/60
        plotRT(subtquery, outpath3, prot, query.Charge, tquery.iloc[0].RT/60, tquery.iloc[-1].RT/60)
    subtquery = subtquery[subtquery.Charge != 0]
    subtquery.sort_values(by=[fsort_by], inplace=True, ascending=False)
    subtquery.to_csv(outfile, index=False, sep='\t', encoding='utf-8',
                     mode='a', header=not os.path.exists(outfile))
    return(subtquery)

def _parallelSeqTable(x, parlist):
    result = processSeqTable(query=x[0], raw=parlist[0], tquery=parlist[1], ptol=parlist[2], ftol=parlist[3],
                             fsort_by=parlist[4], bestn=parlist[5], fullprot=parlist[6], prot=parlist[7],
                             mgf=parlist[8], index2=parlist[9], min_dm=parlist[10], min_match=parlist[11],
                             min_hscore=parlist[12], outpath3=parlist[13], mass=parlist[14], n_workers=parlist[15],
                             parallelize=parlist[16], ppm_plot=parlist[17], outfile=parlist[18], index_offset=parlist[19],
                             mode=parlist[20], int_perc=parlist[21], m_proton=parlist[22], diag_ions=parlist[23], keep_n=parlist[24],
                             spectra=parlist[25], spectra_n=parlist[26])
    return(result)

def main(args):
    '''
    Main function
    '''
    ## PARAMETERS ##
    m_proton = mass.getfloat('Masses', 'm_proton')
    ptol = float(mass._sections['Parameters']['precursor_tolerance'])
    ftol = float(mass._sections['Parameters']['fragment_tolerance'])
    bestn = int(mass._sections['Parameters']['best_n'])
    min_dm = float(mass._sections['Parameters']['min_dm'])
    min_match = int(mass._sections['Parameters']['min_ions_matched'])
    fsort_by = str(mass._sections['Parameters']['sort_by'])
    min_hscore = float(mass._sections['Parameters']['vseq_threshold'])
    ppm_plot = float(mass._sections['Parameters']['ppm_plot'])
    parallelize = str(mass._sections['Parameters']['parallelize'])
    diag_ions = int(mass._sections['Parameters']['diagnostic_ions'])
    keep_n = int(mass._sections['Parameters']['keep_n'])
    int_perc = float(mass._sections['Parameters']['intensity_percent_threshold'])
    score_mode = bool(int(mass._sections['Parameters']['score_mode']))
    full_y = bool(int(mass._sections['Parameters']['full_y']))
    outpath = Path(args.outpath)
    preprocessmsdata = False
    if ptol > 100:
        preprocessmsdata = True
    ## INPUT ##
    logging.info("Reading sequence table")
    seqtable = pd.read_csv(args.table, sep='\t')
    seqtable = seqtable[seqtable.Sequence.notna()]
    prots = seqtable.groupby("q")
    #raws = seqtable.groupby("Raw")
    logging.info("Reading MS file(s)")
    if '*' in args.infile: # wildcard
        mgftable = pd.DataFrame(glob.glob(args.infile))
    else:
        mgftable = pd.read_csv(args.infile, header=None)
    raws = mgftable.groupby(0)
    # if not checkMGFs(raws, list(mgftable[0])):
    #     sys.exit()
    all_outfiles = []
    for raw, rawtable in raws:
        if raw[-4:].lower() == "mzml":
            logging.info("MZML: " + str(os.path.split(raw)[-1][:-5]))
            mode = "mzml"
            # mgf = pyopenms.MSExperiment()
            # pyopenms.MzMLFile().load(raw, mgf)
            mgf, od = read_mzml_with_progress(raw)
            spectra = mgf.getSpectra()
            spectra_n = [int(s.getNativeID().split("=")[-1]) for s in spectra]
            index_offset = 0
            index2 = 0
            tquery, squery, sindex, eindex = getTquery(mgf, mode, raw, int_perc)
            tquery = tquery.drop_duplicates(subset=['SCANS'])
        else:
            logging.info("MGF: " + str(os.path.split(raw)[-1][:-4]))
            mode = "mgf"
            # mgf = pd.read_csv(Path(raw), header=None, sep="\t")
            mgf = read_csv_with_progress(Path(raw), "\t")
            od = None
            logging.info("Getting index offset...")
            index_offset = getOffset(mgf.head(10000)) # Only the first scan is needed
            # logging.info("Building index...")
            index2 = mgf.to_numpy() == 'END IONS'
            spectra = 0
            spectra_n = 0
            logging.info("Building index...")
            tquery, squery, sindex, eindex = getTquery(mgf, mode, raw, int_perc)
            tquery = tquery.drop_duplicates(subset=['SCANS'])
        
        logging.info("Extracting scan data...")
        if preprocessmsdata:
            eparlist = [mgf, index_offset, index2, mode, ftol, int_perc, squery, sindex, eindex]
            indices, rowSeries = zip(*tquery.iterrows())
            rowSeries = list(rowSeries)
            tqdm.pandas(position=0, leave=True)
            tquery_diagnostic = []
            with tqdm(total=tquery.shape[0]) as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as executor:
                    futures = [executor.submit(_parallelProcessSpectrum, row, eparlist, pbar) for row in rowSeries]
                    for future in concurrent.futures.as_completed(futures):
                        tquery_diagnostic.append(future.result())
            tquery["Diagnostic_data"] = tquery_diagnostic
            
        raw = Path(raw).stem
        outpath2 = os.path.join(outpath, str(raw))
        if not os.path.exists(Path(outpath2)):
            os.mkdir(Path(outpath2))
        for fullprot, seqtable in prots:
            try:
                prot = re.search(r'(?<=\|)[a-zA-Z0-9-_]+(?=\|)', fullprot).group(0)
            except AttributeError:
                prot = fullprot
            logging.info("\tPROTEIN: " + str(prot))
            outpath3 = os.path.join(outpath, str(raw), str(prot))
            outfile = os.path.join(outpath3, str(Path(raw).stem) + "_" + str(prot) + "_EXPLORER.tsv")
            all_outfiles += [outfile]
            # if not os.path.exists(Path(outpath3)):
            #     os.mkdir(Path(outpath3))
                
            if parallelize == "protein" or parallelize == "both":
                indices, rowSeqs = zip(*seqtable.iterrows())
                rowSeqs = list(rowSeqs)
                tqdm.pandas(position=0, leave=True)
                parlist = [raw, tquery, ptol, ftol, fsort_by, bestn, fullprot, prot,
                           mgf, index2, min_dm, min_match, min_hscore, outpath3,
                           mass, args.n_workers, parallelize, ppm_plot, outfile, index_offset,
                           mode, int_perc, m_proton, diag_ions, keep_n, spectra, spectra_n]
                # chunks = 100
                # if len(rowSeqs) <= 500:
                #     chunks = 50
                subtqueries = []
                with tqdm(total=len(rowSeqs)) as pbar:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as executor:
                        futures = [executor.submit(_parallelSeqTable, rowSeqs, parlist)
                                   for row in rowSeqs]
                        for future in concurrent.futures.as_completed(futures):
                            pbar.update(1)
                            subtqueries.append(future.result())
                # with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
                #     exploredseqs = list(tqdm(executor.map(_parallelSeqTable,
                #                                           rowSeqs,
                #                                           itertools.repeat(parlist),
                #                                           chunksize=chunks),
                #                       total=len(rowSeqs)))
            elif parallelize == "peptide":
                ## COMPARE EACH SEQUENCE ##
                for index, query in seqtable.iterrows(): # TODO: parallelize
                    ## SEQUENCE ##
                    query.Sequence = str(query.Sequence).upper()
                    plainseq = ''.join(re.findall("[A-Z]+", query.Sequence))
                    mods = [round(float(i),6) for i in re.findall("\d*\.?\d*", query.Sequence) if i]
                    pos = [int(j)-1 for j, k in enumerate(query.Sequence) if k.lower() == '[']
                    ## MZ and MH ##
                    query['expMH'] = query.MH
                    query['expMZ'] = round((query.expMH + (m_proton * (query.Charge-1))) / query.Charge, 6)
                    query['MZ'] = getTheoMZH(query.Charge, plainseq, mods, pos, True, True, mass)[0]
                    query['MH'] = getTheoMZH(query.Charge, plainseq, mods, pos, True, True, mass)[1]
                    logging.info("\tExploring sequence " + str(query.Sequence) + ", "
                                 + str(query.expMH) + " Da, Charge "
                                 + str(query.Charge) + ", " + str(query.expMZ) + "Th")
                    ## DM ##
                    mim = query.expMH
                    dm = mim - query.MH
                    dm_theo_spec = theoSpectrum(plainseq, mods, pos, len(plainseq), dm, mass).loc[0]
                    frags = ["b" + str(i) for i in list(range(1,len(plainseq)+1))] + ["y" + str(i) for i in list(range(1,len(plainseq)+1))[::-1]]
                    # frags_diag = [i for i in frags if i[0]=="b"][len([i for i in frags if i[0]=="b"])//2-diag_ions//2:len([i for i in frags if i[0]=="b"])//2-diag_ions//2+diag_ions] + [i for i in frags if i[0]=="y"][len([i for i in frags if i[0]=="y"])//2-diag_ions//2:len([i for i in frags if i[0]=="y"])//2-diag_ions//2+diag_ions]
                    frags_diag = [i for i in frags if i[0]=="b" and int(i[1])>=diag_ions]+[i for i in frags if i[0]=="y" and int(i[1])>=diag_ions]
                    dm_theo_spec.index = frags
                    if keep_n > 0:
                        frags_diag = dm_theo_spec[frags_diag]
                        frags_diag = (frags_diag+(m_proton*query.Charge))/query.Charge
                    ## TOLERANCE ##
                    upper = query.expMZ + ptol
                    lower = query.expMZ - ptol
                    ## OPERATIONS ##
                    # subtquery = tquery[(tquery.CHARGE==query.Charge) & (tquery.MZ>=lower) & (tquery.MZ<=upper)]
                    subtquery = tquery[(tquery.MZ>=lower) & (tquery.MZ<=upper)]
                    logging.info("\t" + str(subtquery.shape[0]) + " scans found within +-"
                                 + str(ptol) + " Th")
                    if subtquery.shape[0] == 0:
                        continue
                    subtquery['Protein'] = prot
                    subtquery['Sequence'] = query.Sequence
                    subtquery['MH'] = query.expMH
                    subtquery['DeltaMassLabel'] = query.DeltaMassLabel
                    subtquery['DeltaMass'] = dm
                    subtquery.rename(columns={'SCANS': 'FirstScan', 'CHARGE': 'Charge', 'RT':'RetentionTime'}, inplace=True)
                    subtquery["RawCharge"] = subtquery.Charge
                    subtquery.Charge = query.Charge
                    parlist = [tquery, mgf, index2, min_dm, min_match, ftol, Path(outpath3),
                               False, mass, False, min_hscore, ppm_plot, index_offset, mode,
                               int_perc, squery, sindex, eindex, spectra, spectra_n, fsort_by,
                               od]
                    # DIA: Filter by diagnostic ions
                    logging.info("\tFiltering by diagnostic ions...")
                    if keep_n > 0:
                        if preprocessmsdata:
                            chunks = math.ceil(len(subtquery)/args.n_workers)
                            eparlist = [0, index_offset, index2, mode, frags_diag, ftol, int_perc, squery, sindex, eindex, preprocessmsdata]
                            indices, rowSeries = zip(*subtquery.iterrows())
                            rowSeries = list(rowSeries)
                            with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
                                diag = list(tqdm(executor.map(_parallelExpSpectrum,
                                                                 rowSeries,
                                                                 itertools.repeat(eparlist),
                                                                 chunksize=chunks),
                                                    total=len(rowSeries)))
                            subtquery['temp_diagnostic'] = diag
                        else:
                            subtquery['temp_diagnostic'] = subtquery.apply(lambda x: expSpectrum(mgf, index_offset, x.FirstScan, index2,
                                                                                            mode, frags_diag, ftol, int_perc,
                                                                                            squery, sindex, eindex, preprocessmsdata,
                                                                                            0), axis=1)
                        subtquery['Diagnostic_Ions'] = pd.DataFrame(subtquery.temp_diagnostic.tolist()).iloc[:, 0]. tolist()
                        subtquery['Diagnostic_Intensity'] = pd.DataFrame(subtquery.temp_diagnostic.tolist()).iloc[:, 1]. tolist()
                        subtquery = subtquery.drop('temp_diagnostic', axis = 1)
                        subtquery = subtquery.nlargest(keep_n, ['Diagnostic_Ions', 'Diagnostic_Intensity'])
                        subtquery = subtquery.sort_index()
                    logging.info("\tKept " + str(subtquery.shape[0]) + " scans with more diagnostic ions and highest diagnostic ion intensity")
                    indices, rowSeries = zip(*subtquery.iterrows())
                    rowSeries = list(rowSeries)
                    tqdm.pandas(position=0, leave=True)
                    logging.info("\tComparing...")
                    # chunks = 100
                    # if len(rowSeries) <= 500:
                    #     chunks = 50
                    # with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:
                    #     # with tqdm(total=len(rowSeries)) as progress_bar:
                    #     #     futures = {}
                    #     #     for idx, dt in enumerate(rowSeries):
                    #     #         future = executor.submit(_parallelGetIons, dt, itertools.repeat(parlist))
                    #     #         futures[future] = idx
                    #     #     vseqs = [None] * len(rowSeries)
                    #     #     for future in concurrent.futures.as_completed(futures):
                    #     #         idx = futures[future]
                    #     #         vseqs[idx] = future.result()
                    #     #         progress_bar.update(1)
                    #     vseqs = list(tqdm(executor.map(_parallelGetIons, rowSeries, itertools.repeat(parlist), chunksize=chunks),
                    #                       total=len(rowSeries)))
                    vseqs = []
                    scans = []
                    with tqdm(total=len(rowSeries)) as pbar:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_workers) as executor:
                            futures = [executor.submit(_parallelGetIons, row, parlist, pbar) for row in rowSeries]
                            for future in concurrent.futures.as_completed(futures):
                                vseqs.append(future.result()[0])
                                scans.append(future.result()[1])
                    order = pd.DataFrame([vseqs, scans]).T
                    order.columns = ['vseqs', 'FirstScan']
                    order = order.sort_values(by='FirstScan')
                    subtquery = subtquery.sort_values(by='FirstScan')
                    subtquery['templist'] = list(order.vseqs)
                    # subtquery['templist'] = vseqs
                    subtquery['ions_matched'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 0]. tolist()
                    #subtquery['ions_exp'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 1]. tolist()
                    subtquery['ions_total'] = len(plainseq) * 2
                    subtquery['b_series'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 2]. tolist()
                    subtquery['y_series'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 3]. tolist()
                    subtquery['Raw'] = str(raw)
                    subtquery['v_score'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 4]. tolist()
                    subtquery['e_score'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 5]. tolist()
                    subtquery['hyperscore'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 6]. tolist()
                    subtquery['product'] = subtquery['ions_matched'] * subtquery['e_score']
                    subtquery = subtquery.drop('templist', axis = 1)
                    ## SORT BY ions_matched ##
                    logging.info("\tSorting by " +  str(fsort_by) + "...")
                    try:
                        subtquery.sort_values(by=['INT'], inplace=True, ascending=False)
                        subtquery.sort_values(by=[fsort_by], inplace=True, ascending=False)
                    except KeyError:
                        subtquery.sort_values(by=[fsort_by], inplace=True, ascending=False)
                    # subtquery.sort_values(by=[fsort_by], inplace=True, ascending=False)
                    subtquery.reset_index(drop=True, inplace=True)
                    f_subtquery = subtquery.iloc[0:bestn]
                    f_subtquery.reset_index(drop=True, inplace=True)
                    # f_subtquery["shortseq"] = f_subtquery.apply(lambda x: x.Sequence if len(x.Sequence)>= else x.Sequence[:len(x.Sequence)//2] + "_trunc", axis=1)
                    f_subtquery["outpath"] = str(outpath3) + "/" + str(prot) + "_" + f_subtquery.Sequence.astype(str) + "_" + f_subtquery.FirstScan.astype(str) + "_ch" + f_subtquery.Charge.astype(str) + "_cand" + (f_subtquery.index.values+1).astype(str) + ".pdf"
                    # f_subtquery["outpath"] = makeOutpath(outpath3, prot, f_subtquery.Sequence.astype(str), f_subtquery.FirstScan.astype(str), f_subtquery.Charge.astype(str), (f_subtquery.index.values+1).astype(str))
                    if f_subtquery.shape[0] > 0:
                        logging.info("\tRunning Vseq on " + str(len(f_subtquery)) + " best candidates...")
                        f_subtquery = f_subtquery[f_subtquery[fsort_by]>min_hscore]
                        if not os.path.exists(Path(outpath3)):
                            os.mkdir(Path(outpath3))
                        f_subtquery.apply(lambda x: doVseq(mode,
                                                           index_offset,
                                                           x,
                                                           tquery,
                                                           mgf,
                                                           index2,
                                                           spectra,
                                                           spectra_n,
                                                           min_dm,
                                                           min_match,
                                                           ftol,
                                                           Path(x.outpath),
                                                           False,
                                                           mass,
                                                           True,
                                                           0,
                                                           ppm_plot,
                                                           int_perc,
                                                           squery,
                                                           sindex,
                                                           eindex,
                                                           calc_hs=0,
                                                           hs=x.hyperscore,
                                                           sortby=fsort_by,
                                                           od=od), axis = 1)
                    allpagelist = list(map(Path, list(f_subtquery["outpath"])))
                    pagelist = []
                    for f in allpagelist:
                        if os.path.isfile(f):
                            pagelist.append(f)
                    merger = PdfMerger()
                    for page in pagelist:
                        merger.append(io.FileIO(page,"rb"))
                    logging.info("\tFound " + str(len(pagelist)) + " candidates with " + str(fsort_by) + " > " + str(min_hscore))
                    if len(pagelist) > 0:
                        outmerge = os.path.join(Path(outpath3), str(prot) + "_" + str(query.Sequence) + "_M" + str(round(query.expMH,4)) + "_ch" + str(query.Charge) + "_best" + str(bestn) + ".pdf")
                        with open(outmerge, 'wb') as f:
                            merger.write(f)
                        for page in pagelist:
                            os.remove(page)
                        #if len(x.b_series)>1 and len(x.y_series)>1 else logging.info("\t\tSkipping one candidate with empty fragmentation series...")
                        ## PLOT RT vs E-SCORE and MATCHED IONS ##
                        subtquery.loc[len(subtquery)] = 0
                        subtquery.iloc[-1].RetentionTime = tquery.iloc[0].RT/60
                        subtquery.loc[len(subtquery)] = 0
                        subtquery.iloc[-1].RetentionTime = tquery.iloc[-1].RT/60
                        plotRT(subtquery, outpath3, prot, query.Charge, tquery.iloc[0].RT/60, tquery.iloc[-1].RT/60)
                    #exploredseqs.append(subtquery)
                    subtquery = subtquery[subtquery.Charge != 0]
                    subtquery.sort_values(by=[fsort_by], inplace=True, ascending=False)
                    subtquery.drop("SPECTRUM", axis=1, inplace=True)
                    subtquery.to_csv(outfile, index=False, sep='\t', encoding='utf-8',
                                     mode='a', header=not os.path.exists(outfile))
    logging.info("Creating joined results file...")
    all_data = []
    for f in all_outfiles:
        try:
            temp = pd.read_csv(os.path.join(f), delimiter='\t')
            all_data.append(temp)
        except pd.errors.EmptyDataError:
            continue
    all_data = pd.concat(all_data)
    all_data['QC_Plot'] = all_data.apply(lambda x: os.path.join(outpath, str(x.Raw), str(x.Protein),
                                                                str(x.Protein)+"_"+x.Sequence+"_M"+str(x.MH)+"_ch"+str(x.Charge)+"_best5.pdf"), axis=1)
    all_data['RT_Plot'] = all_data.apply(lambda x: os.path.join(outpath, str(x.Rraw), str(x.Protein),
                                                                str(x.Protein)+"_"+x.Sequence+"_M"+str(x.MH)+"_ch"+str(x.Charge)+"_RT_plots.pdf"), axis=1)
    all_data.to_csv(os.path.join(outpath, "VSEQ_EXPLORER_RESULTS.tsv"), sep='\t', index=False)
        # if exploredseqs:    
        #     logging.info("Writing output table")
        #     # outfile = os.path.join(os.path.split(Path(args.table))[0],
        #     #                        os.path.split(Path(args.table))[1][:-4] + "_EXPLORER.csv")
        #     outfile = os.path.join(outpath2, str(Path(raw).stem) + "_EXPLORER.tsv")
        #     bigtable = pd.concat(exploredseqs, ignore_index=True, sort=False)
        #     bigtable = bigtable[bigtable.Charge != 0]
        #     bigtable.to_csv(outfile, index=False, sep='\t', encoding='utf-8')
    # files = []
    # for r, d, f in os.walk(outpath):
    #     for file in f:
            # if file[-3:]=="tsv":
                # files += [os.path.join(r, file)]
    # dfs = [pd.read_csv(i, sep="\t") for i in files]
    # dfs = [pd.read_csv(i, sep="\t").assign(**{"Raw": "_".join(os.path.basename(i).split("_")[:-2])}) for i in files]
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
        
    defaultconfig = os.path.join(os.path.dirname(__file__), "Vseq.ini")
    
    parser.add_argument('-i',  '--infile', required=True, help='Table of MGFs to search')
    parser.add_argument('-t',  '--table', required=True, help='Table of sequences to compare')
    parser.add_argument('-c', '--config', default=defaultconfig, help='Path to custom config.ini file')
    parser.add_argument('-o', '--outpath', help='Path to save results')
    parser.add_argument('-w',  '--n_workers', type=int, default=4, help='Number of threads/n_workers (default: %(default)s)')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()
    
    if args.verbose:
        #warnings.filterwarnings('ignore')
        shutup.jk()
    
    # parse config
    mass = configparser.ConfigParser(inline_comment_prefixes='#')
    with io.open(args.config, "r", encoding="utf-8") as my_config:
        mass.read_file(my_config)
    # if something is changed, write a copy of ini
    if mass.getint('Logging', 'create_ini') == 1:
        with open(os.path.dirname(args.table) + '/Vseq.ini', 'w') as newconfig:
            mass.write(newconfig)
    
    # make outdir
    if args.outpath:
        args.outpath = os.path.join(Path(args.outpath),"Vseq_Results")
    else:
        args.outpath = os.path.join(os.path.dirname(Path(args.table)),"Vseq_Results")
    if not os.path.exists(args.outpath):
        Path(args.outpath).mkdir(parents=True, exist_ok=True)

    # logging debug level. By default, info level
    log_file = os.path.join(Path(args.outpath), 'VseqExplorer_log.txt')
    log_file_debug = os.path.join(Path(args.outpath), 'VseqExplorer_log_debug.txt')
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
    logging._defaultFormatter = logging.Formatter(u"%(message)s")

    # start main function
    logging.info('start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    try:
        main(args)
    except:
        logging.exception('An error occurred')
    logging.info('end script')
