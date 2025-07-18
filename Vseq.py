# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:10:14 2022

@author: alaguillog
"""

# import modules
import argparse
import configparser
import io
import itertools
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pyopenms
import random
import re
import seaborn as sns
import scipy.stats
import statistics
import sys
# import custom modules
from tools.Hyperscore import scoreVseq, locateScan
import tools.ScanIntegrator as ScanIntegrator
from tqdm import tqdm
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
    return df

def read_mzml_with_progress(inputfile):
    ondisc_exp = pyopenms.OnDiscMSExperiment()
    ondisc_exp.openFile(inputfile)
    mgf = pyopenms.MSExperiment()
    for i in tqdm(range(ondisc_exp.getNrSpectra()), desc="Loading spectra"):
        spectrum = ondisc_exp.getSpectrum(i)
        mgf.addSpectrum(spectrum)
    return mgf

def prepareWorkspace(exp, msdatapath, outpath):
    msdata = Path(msdatapath)
    outpath = Path(outpath)
    # Get dta path for the experiment
    var_name_path = os.path.join(outpath, exp)
    # Create output directory
    if not os.path.exists(outpath):
        os.mkdir(outpath)
        os.mkdir(var_name_path)
    logging.info("Experiment: " + exp)
    logging.info("msdatapath: " + str(msdata))
    logging.info("outpath: " + str(outpath))
    logging.info("varNamePath: " + str(var_name_path))
    pathdict = {"exp": exp,
                "msdata": msdatapath,
                "out": outpath,
                "var_name": var_name_path}
    return pathdict

def getOffset(fr_ns):
    def _check(can):
        try:
            a = list(map(float, can))
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

def getTquery(fr_ns, mode):
    if mode == "mgf":
        squery = fr_ns.loc[fr_ns[0].str.contains("SCANS=")]
        squery = squery[0].str.replace("SCANS=","")
        squery.reset_index(inplace=True, drop=True)
        mquery = fr_ns.loc[fr_ns[0].str.contains("PEPMASS=")]
        mquery = mquery[0].str.replace("PEPMASS=","")
        mquery.reset_index(inplace=True, drop=True)
        cquery = fr_ns.loc[fr_ns[0].str.contains("CHARGE=")]
        cquery = cquery[0].str.replace("CHARGE=","")
        cquery.reset_index(inplace=True, drop=True)
        tquery = pd.concat([squery.rename('SCANS'),
                            mquery.rename('PEPMASS'),
                            cquery.rename('CHARGE')],
                           axis=1)
        try:
            tquery[['MZ','INT']] = tquery.PEPMASS.str.split(" ",expand=True,)
        except ValueError:
            tquery['MZ'] = tquery.PEPMASS
        tquery['CHARGE'] = tquery.CHARGE.str[:-1]
        tquery = tquery.drop("PEPMASS", axis=1)
        tquery = tquery.apply(pd.to_numeric)
    elif mode == "mzml":
        tquery = []
        for s in fr_ns.getSpectra(): # TODO this is slow
            if s.getMSLevel() == 2:
                df = pd.DataFrame([int(s.getNativeID().split(' ')[-1][5:]), # Scan
                          s.getPrecursors()[0].getCharge(), # Precursor Charge
                          s.getRT(), # Precursor Retention Time
                          s.getPrecursors()[0].getMZ(), # Precursor MZ
                          s.getPrecursors()[0].getIntensity()]).T # Precursor Intensity
                df.columns = ["SCANS", "CHARGE", "RT", "MZ", "INT"]
                tquery.append(df)
        tquery = pd.concat(tquery)
        tquery = tquery.apply(pd.to_numeric)
    return tquery

def getTheoMH(charge, sequence, mods, pos, nt, ct, massconfig, standalone):
    '''    
    Calculate theoretical MH using the PSM sequence.
    '''
    if not standalone:
        mass = massconfig
    else:
        mass = configparser.ConfigParser(inline_comment_prefixes='#')
        mass.read(args.config)
        if args.error is not None:
            mass.set('Parameters', 'ppm_error', str(args.error))
        if args.deltamass is not None:
            mass.set('Parameters', 'min_dm', str(args.deltamass))
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
    return MH

def expSpectrum(fr_ns, index_offset, scan, index2, mode, int_perc, spectra,
                spectra_n, squery=0, sindex=0, eindex=0, od=None):
    '''
    Prepare experimental spectrum.
    '''
    if mode == "mgf":
        if squery != 0:
            place = squery.index(str(scan))
            ions = fr_ns.iloc[sindex[place]+1:eindex[place]]
        else:
            index1 = fr_ns.loc[fr_ns[0]=='SCANS='+str(scan)].index[0] + index_offset
            index3 = np.where(index2)[0]
            index3 = index3[np.searchsorted(index3,[index1,],side='right')[0]]
            # try:
            ions = fr_ns.iloc[index1:index3,:]
            ions[0] = ions[0].str.strip()
        ions[['MZ','INT']] = ions[0].str.split(" ",expand=True,)
        ions = ions.drop(ions.columns[0], axis=1)
        ions = ions.apply(pd.to_numeric)
    elif mode == "mzml":
        # s = spectra[spectra_n.index(scan)]
        nativeid = '='.join(spectra[0].getNativeID().split('=')[:-1]) + '=' + str(scan)
        s = od.getSpectrumByNativeId(nativeid)
        ions = pd.DataFrame([s.get_peaks()[0], s.get_peaks()[1]]).T
        ions.columns = ["MZ", "INT"]
    # DIA: Filter by intensity ratio
    if int_perc > 0:
        ions=ions[ions.INT>=ions.INT.max()*int_perc]
    ions["ZERO"] = 0
    #ions["CCU"] = 0.01
    ions["CCU"] = ions.MZ - 0.01
    ions.reset_index(drop=True)
    bind = pd.DataFrame(list(itertools.chain.from_iterable(zip(list(ions['CCU']),list(ions['MZ'])))), columns=["MZ"])
    bind["REL_INT"] = list(itertools.chain.from_iterable(zip(list(ions['ZERO']),list(ions['INT']))))
    bind["ZERO"] = 0
    bind["CCU"] = bind.MZ + 0.01
    
    spec = pd.DataFrame(list(itertools.chain.from_iterable(zip(list(bind['MZ']),list(bind['CCU'])))), columns=["MZ"])
    spec["REL_INT"] = list(itertools.chain.from_iterable(zip(list(bind['REL_INT']),list(bind['ZERO']))))
    
    median_rel_int = statistics.median(ions.INT)
    std_rel_int = np.std(ions.INT, ddof = 1)
    ions["NORM_REL_INT"] = (ions.INT - median_rel_int) / std_rel_int
    ions["P_REL_INT"] = scipy.stats.norm.cdf(ions.NORM_REL_INT) #, 0, 1)
    normspec = ions.loc[ions.P_REL_INT>0.81]
    if len(ions) > 0 and len(normspec) > 0:
        spec_correction = max(ions.INT)/statistics.mean(normspec.INT)
    else: spec_correction = 0
    spec["CORR_INT"] = spec.REL_INT*spec_correction
    spec.loc[spec['CORR_INT'].idxmax()]['CORR_INT'] = max(spec.REL_INT)
    spec["CORR_INT"] = spec.apply(lambda x: max(ions.INT)-13 if x["CORR_INT"]>max(ions.INT) else x["CORR_INT"], axis=1)
    return(spec, ions, spec_correction)

def theoSpectrum(seq, mods, pos, len_ions, dm, massconfig, standalone):
    '''
    Prepare theoretical fragment matrix.

    '''
    if not standalone:
        mass = massconfig
    else:
        mass = configparser.ConfigParser(inline_comment_prefixes='#')
        mass.read(args.config)
        if args.error is not None:
            mass.set('Parameters', 'ppm_error', str(args.error))
        if args.deltamass is not None:
            mass.set('Parameters', 'min_dm', str(args.deltamass))
    m_hydrogen = mass.getfloat('Masses', 'm_hydrogen')
    m_oxygen = mass.getfloat('Masses', 'm_oxygen')
    ## Y SERIES ##
    #ipar = list(range(1,len(seq)))
    outy = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    pos_y = [len(seq)-1-i for i in pos]
    for i in range(0,len(seq)):
        yn = list(seq[i:])
        if i > 0: nt = False
        else: nt = True
        fragy = getTheoMH(0,yn,mods,pos_y,nt,True, massconfig, standalone) + dm
        outy[i:] = fragy
        
    ## B SERIES ##
    outb = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        bn = list(seq[::-1][i:])
        if i > 0: ct = False
        else: ct = True
        fragb = getTheoMH(0,bn,mods,pos,True,ct, massconfig, standalone) - 2*m_hydrogen - m_oxygen + dm
        outb[i:] = fragb
    
    ## FRAGMENT MATRIX ##
    yions = outy.T
    bions = outb.iloc[::-1].T
    spec = pd.concat([bions, yions], axis=1)
    spec.columns = range(spec.columns.size)
    spec.reset_index(inplace=True, drop=True)
    return(spec)

def errorMatrix(mz, theo_spec, massconfig, standalone):
    '''
    Prepare ppm-error and experimental mass matrices.
    '''
    if not standalone:
        mass = massconfig
    else:
        mass = configparser.ConfigParser(inline_comment_prefixes='#')
        mass.read(args.config)
        if args.error is not None:
            mass.set('Parameters', 'ppm_error', str(args.error))
        if args.deltamass is not None:
            mass.set('Parameters', 'min_dm', str(args.deltamass))
    m_proton = mass.getfloat('Masses', 'm_proton')
    exp = pd.DataFrame(np.tile(pd.DataFrame(mz), (1, len(theo_spec.columns)))) 
    
    ## EXPERIMENTAL MASSES FOR CHARGE 2 ##
    mzs2 = pd.DataFrame(mz)*2 - m_proton
    mzs2 = pd.DataFrame(np.tile(pd.DataFrame(mzs2), (1, len(exp.columns)))) 
    
    ## EXPERIMENTAL MASSES FOR CHARGE 3 ##
    mzs3 = pd.DataFrame(mz)*3 - m_proton*2
    mzs3 = pd.DataFrame(np.tile(pd.DataFrame(mzs3), (1, len(exp.columns)))) 
    
    ## PPM ERRORS ##
    terrors = (((exp - theo_spec)/theo_spec)*1000000).abs()
    terrors2 =(((mzs2 - theo_spec)/theo_spec)*1000000).abs()
    terrors3 = (((mzs3 - theo_spec)/theo_spec)*1000000).abs()
    return(terrors, terrors2, terrors3, exp)

def makeFrags(seq_len):
    '''
    Name all fragments.
    '''
    frags = pd.DataFrame(np.nan, index=list(range(0,seq_len*2)),
                         columns=["by", "by2", "by3", "bydm", "bydm2", "bydm3"])
    frags.by = ["b" + str(i) for i in list(range(1,seq_len+1))] + ["y" + str(i) for i in list(range(1,seq_len+1))[::-1]]
    frags.by2 = frags.by + "++"
    frags.by3 = frags.by + "+++"
    frags.bydm = frags.by + "*"
    frags.bydm2 = frags.by + "*++"
    frags.bydm3 = frags.by + "*+++"
    return(frags)

def assignIons(theo_spec, dm_theo_spec, frags, dm, arg_dm, massconfig, standalone):
    if not standalone:
        mass = massconfig
    else:
        mass = configparser.ConfigParser(inline_comment_prefixes='#')
        mass.read(args.config)
        if args.error is not None:
            mass.set('Parameters', 'ppm_error', str(args.error))
        if args.deltamass is not None:
            mass.set('Parameters', 'min_dm', str(args.deltamass))
    m_proton = mass.getfloat('Masses', 'm_proton')
    assign = pd.concat([frags.by, theo_spec.iloc[0]], axis=1)
    assign.columns = ['FRAGS', '+']
    assign["++"] = (theo_spec.iloc[0]+m_proton)/2
    assign["+++"] = (theo_spec.iloc[0]+2*m_proton)/3
    assign["*"] = dm_theo_spec.iloc[0]
    assign["*++"] = (dm_theo_spec.iloc[0]+m_proton)/2
    
    c_assign = pd.DataFrame(list(assign["+"]) + list(assign["++"]) + list(assign["+++"]))
    if dm >= arg_dm:
        c_assign = pd.concat([c_assign, pd.DataFrame(list(assign["*"])), pd.DataFrame(list(assign["*++"]))])
    c_assign.columns = ["MZ"]
    c_assign_frags = pd.DataFrame(list(frags.by) + list(frags.by + "++") + list(frags.by + "+++"))
    if dm >= arg_dm:
        c_assign_frags = pd.concat([c_assign_frags, pd.DataFrame(list(frags.by + "*")), pd.DataFrame(list(frags.by + "*++"))])
    c_assign["FRAGS"] = c_assign_frags
    c_assign["ION"] = c_assign.apply(lambda x: re.findall(r'\d+', x.FRAGS)[0], axis=1)
    c_assign["CHARGE"] = c_assign.apply(lambda x: x.FRAGS.count('+'), axis=1).replace(0, 1)
    return(c_assign)

def makeAblines(texp, minv, assign, ions, min_match):
    masses = pd.concat([texp[0], minv], axis = 1)
    matches = masses[(masses < 51).sum(axis=1) >= 0.001]
    matches.reset_index(inplace=True, drop=True)
    if len(matches) <= min_match:
        matches = pd.DataFrame([[1,3],[2,4]])
        proof = pd.DataFrame([[0,0,0,0]])
        proof.columns = ["MZ","FRAGS","PPM","INT"]
        return(proof, False) 
    matches_ions = pd.DataFrame(np.repeat(list(matches[0]), len(assign)))
    matches_ions.columns = ["temp_mi"]
    matches_ions["temp_ci1"] = np.tile(np.array(assign.FRAGS), len(matches))
    matches_ions["temp_mi1"] = np.repeat(list(matches["minv"]), len(assign))
    matches_ions["temp_ci"] = np.tile(np.array(assign.MZ), len(matches))
    matches_ions["check"] = abs(matches_ions.temp_mi-matches_ions.temp_ci)/matches_ions.temp_ci*1000000
    matches_ions = matches_ions[matches_ions.check<=51]
    matches_ions = matches_ions.drop(["temp_ci", "check"], axis = 1)
    matches_ions.columns = ["MZ","FRAGS","PPM"]
    # for mi in list(range(0,len(matches))):
    #     for ci in list(range(0, len(assign))):
    #         if abs(matches.iloc[mi,0]-assign.iloc[ci,0])/assign.iloc[ci,0]*1000000 <= 51:
    #             asign = pd.Series([matches.iloc[mi,0], assign.iloc[ci,1], matches.iloc[mi,1]])
    #             matches_ions = pd.concat([matches_ions, asign], ignore_index=True, axis=1)
    #             #matches.iloc[2,1]
    # matches_ions = matches_ions.T
    if matches_ions.empty:
        proof = pd.DataFrame([[0,0,0,0]])
        proof.columns = ["MZ","FRAGS","PPM","INT"]
        return(proof, False)
    proof = matches_ions.set_index('MZ').join(ions[['MZ','INT']].set_index('MZ'))
    if len(proof)==0:
        mzcycle = itertools.cycle([ions.MZ.iloc[0], ions.MZ.iloc[1]])
        proof = pd.concat([matches_ions, pd.Series([next(mzcycle) for count in range(len(matches_ions))], name="INT")], axis=1)
    proof = proof.reset_index()
    return(proof, True)

def deltaPlot(parcialdm, parcial, ppmfinal):
    deltamplot = pd.DataFrame(np.array([parcialdm, parcial, ppmfinal]).max(0)) # Parallel maxima
    deltamplot = deltamplot[(deltamplot > 0).sum(axis=1) > 0]
    if deltamplot.empty:
        deltamplot = parcial
    #deltamplot.reset_index(inplace=True, drop=True)
    rplot = []
    cplot = []
    for ki in list(range(0,deltamplot.shape[0])): #rows
        for kj in list(range(0,deltamplot.shape[1])): #columns
            if deltamplot.iloc[ki,kj] == 3:
                rplot.append(deltamplot.index.values[ki]) 
                cplot.append(deltamplot.columns.values[kj]) 
    deltaplot = pd.DataFrame([pd.Series(rplot,dtype='int'), pd.Series(cplot,dtype='str')]).T
    if deltaplot.shape[0] != 0:
        deltaplot.columns = ["row", "deltav2"]
        deltaplot["deltav1"] = deltamplot.shape[0] - deltaplot.row
    else:
        deltaplot = pd.DataFrame([[0,0,0]], columns=["row","deltav2","deltav1"])
    deltaplot = deltaplot.apply(pd.to_numeric)
    return(deltamplot, deltaplot)

def qeScore(ppmfinal, int2, err):
    int2.reset_index(inplace=True, drop=True)
    ppmfinal["minv"] = ppmfinal.min(axis=1)
    qscore = pd.DataFrame(ppmfinal["minv"])
    qscore[qscore > err] = 0
    qscore["INT"] = int2
    qscoreFALSE = pd.DataFrame([[21,21],[21,21]])
    qscore = qscore[(qscore>0).all(1)]
    if qscore.shape[0] == 2:
        qscore = qscoreFALSE
        qscore.columns = ["minv", "INT"]
    escore = (qscore.INT/1000000).sum()
    return(qscore, escore)

def asBY(deltaplot, sub, sublen):
    asB = pd.DataFrame()
    asY = pd.DataFrame()
    for i in list(range(0,deltaplot.shape[0])):
        if deltaplot.deltav2[i] <= sublen-1:
            asB = pd.concat([asB, deltaplot.iloc[i]], axis=1)
        if deltaplot.deltav2[i] > sublen-1:
            asY = pd.concat([asY, deltaplot.iloc[i]], axis=1)
    if asB.empty:
        asB = pd.DataFrame([[0],[0],[0]])
        asB.index = ["row","deltav2","deltav1"]
    if asY.empty:
        asY = pd.DataFrame([[0],[0],[0]])
        asY.index = ["row","deltav2","deltav1"]
    asB = asB.T.drop("row", axis=1)
    #asB = asB.iloc[::-1]
    asB.columns = ["row","col"]
    asB = asB.sort_values(by="row")
    asB.reset_index(inplace=True, drop=True)
    asY = asY.T.drop("row", axis=1)
    asY.columns = ["row","col"]
    asY = asY.sort_values(by="row")
    asY.reset_index(inplace=True, drop=True)
    if asB.empty:
        asB = pd.DataFrame([[0,0]], columns=["row", "col"])
    if asY.empty:
        asY = pd.DataFrame([[0,0]], columns=["row", "col"])
        
    BDAG = pd.DataFrame([[0,0,0,0,0,0,0,0]] * asB.shape[0],
                        columns=["row", "col", "dist", "value", "consec", "int", "error", "counts"])
    BDAG.row = asB.row
    BDAG.col = asB.col
    BDAG.counts = pd.Series(BDAG.index.values + 1)
    YDAG = pd.DataFrame([[0,0,0,0,0,0,0,0]] * asY.shape[0],
                        columns=["row", "col", "dist", "value", "consec", "int", "error", "counts"])
    YDAG.row = asY.row
    YDAG.col = asY.col
    YDAG.counts = pd.Series(YDAG.index.values + 1)
    YDAG.sort_values(by="counts")
    
    for asBY, BYDAG in [[asB, BDAG], [asY, YDAG]]:
        if len(asBY) > 1:
            for i in list(range(1,BYDAG.shape[0]))[::-1]:
                BYDAG.dist.iloc[-1] = BYDAG.col.iloc[-1]
                BYDAG.dist.iloc[i-1] = abs(BYDAG.col.iloc[i] - BYDAG.col.iloc[i-1])
                for j in list(range(1,i+1)):
                    if BYDAG.dist.iloc[i] <= 7: # TODO: adjust this value
                        BYDAG.value.iloc[j] = BYDAG.value.iloc[i] + 1
                    elif BYDAG.dist.iloc[i] > 7:
                        BYDAG.value.iloc[j] = 0
            BYDAG.consec = BYDAG.value + 1
    BDAGmax = BDAG[BDAG.value == BDAG.value.max()]
    if len(BDAG) == 1:
        BDAGmax.row = 0
    YDAGmax = YDAG[YDAG.value == YDAG.value.max()]
    YDAGmax = YDAGmax.row - sublen
    bpos = BDAG.iloc[0].row
    ypos = YDAGmax
    return(BDAGmax, YDAGmax, bpos, ypos)

def sortFrags(proofby_df):
    proofby_df["FRAGS0"] = proofby_df.FRAGS.str.extract(r'([a-zA-Z])', expand=True)
    proofby_df["FRAGS1"] = proofby_df.FRAGS.str.extract(r'([0-9]+)', expand=True)
    proofby_df["FRAGS2"] = proofby_df.FRAGS.str.extract(r'(\++)', expand=True).fillna('')
    proofby_df["FRAGS1"] = pd.to_numeric(proofby_df["FRAGS1"])
    proofby_df.sort_values(by="FRAGS1", inplace=True)
    chargegroups = proofby_df.groupby("FRAGS2")
    results = []
    for chargestate in chargegroups:
        group_index = chargestate[1].index.values
        results.append(proofby_df.loc[group_index])
    results = pd.concat(results)
    return(results)

def vScore(qscore, sub, sublen, proofb, proofy, assign):
    '''
    Calculate vScore.
    '''
    Kerr = 0
    Kv = 0.1
    ## SS1 ##
    if len(qscore) <= (sublen*2)/4:
        SS1 = 1
    if len(qscore) > (sublen*2)/4:
        SS1 = 2
    if len(qscore) > (sublen*2)/3:
        SS1 = 3
    if len(qscore) > (sublen*2)/2:
        SS1 = 4
    
    ## SS2 ##
    proofb_vscore = proofb[proofb.PPM < 20]
    proofb_vscore.reset_index(inplace=True)
    if len(proofb_vscore) == 0:
        SS2 = 0
    if len(proofb_vscore) != 0:
        SS2 = SS1 * 1.5
    if len(proofb_vscore) > sublen/3:
        SS2 = SS1 * 2.5
    if len(proofb_vscore) > (sublen*2)/3:
        SS2 = SS1 * 3.5
    
    ## SS3 ##
    proofy_vscore = proofy[proofy.PPM < 20]
    proofy_vscore.reset_index(inplace=True)
    if len(proofy_vscore) == 0:
        SS3 = 0
    if len(proofy_vscore) != 0:
        SS3 = SS1 * 3
    if len(proofy_vscore) > sublen/3:
        SS3 = SS1 * 5
    if len(proofy_vscore) > (sublen*2)/3:
        SS3 = SS1 * 7

    ## SS4 ##
    SS4b = SS4y = 0
    temp = []
    for proofby_vscore, SS4 in [[proofb_vscore, SS4b], [proofy_vscore, SS4y]]:
        if len(proofby_vscore) > 1:
            proofby_vscore = pd.concat([proofby_vscore, pd.merge(proofby_vscore, assign, on="FRAGS")[["ION", "CHARGE"]]], axis=1)
            #proofby_vscore = proofby_vscore.sort_values(by="CHARGE")
            proofby_vscore = sortFrags(proofby_vscore)
            proofby_vscore["ION_DIFF"] = -pd.to_numeric(proofby_vscore.ION).diff(periods=-1)
            SS4 = len(proofby_vscore[proofby_vscore.ION_DIFF == 1])
            temp.append([proofby_vscore, SS4])
        else:
            temp.append([proofby_vscore, SS4])
    proofb_vscore, proofy_vscore = temp[0][0], temp[1][0]
    SS4b, SS4y = temp[0][1], temp[1][1]
    SS4 = SS4b + SS4y
    
    ## SS5 and SS6 ##
    if len(proofb_vscore) <= 1:
        SS5b = 0
        SS6b = 0
        Kv = 0.1
    if len(proofb_vscore) > 1:
        SS5b = statistics.median(pd.to_numeric(proofb_vscore.ION))
        SS6b = statistics.median(pd.to_numeric(proofb_vscore.PPM))
    if len(proofy_vscore) <= 1:
        SS5y = 0
        SS6y = 0
        Kv = 0.1
    if len(proofy_vscore) > 1:
        SS5y = statistics.median(pd.to_numeric(proofy_vscore.ION))
        SS6y = statistics.median(pd.to_numeric(proofy_vscore.PPM))
    SS5 = SS5b + SS5y
    SS6 = (SS6b + SS6y)/2
    if SS6b == 0 or SS6y == 0:
        SS6 = SS6b + SS6y
    if SS6 < 16:
        Kerr = 3
    if SS6 < 11:
        Kerr = 6
    if SS6 < 6:
        Kerr = 9
    
    ## Kv ##
    if SS4b < 4 and SS4y < 4 and SS5b < 7 and SS5y < 7:
        Kv = 0.01
    if SS4b <= 4 or SS4y <= 4:
        if SS5b >= 8 or SS5y >= 8:
            Kv = 0.8
    if SS4b >=5 or SS4y >= 5:
        if SS5b < 6 or SS5y < 6:
            Kv = 1.2
    if SS4b >=5 or SS4y >= 5:
        if SS5b >= 6 or SS5y >= 6:
            Kv = 1.5
    if SS4b > 7 or SS4y > 7:
        if SS5b >= 8 or SS5y >= 8:
            Kv = 1.8
    if SS4b > 7 and SS4y > 7 and SS5b >= 8 and SS5y >= 8:
        Kv = 2.7
        
    vscore = (SS1 + SS2 + SS3 + Kerr + (SS4 * SS5)) * Kv / sublen
    return(vscore)

def locateFixedMods(proof, plainseq, mods, pos, massconfig, standalone):
    if not standalone:
        mass = massconfig
    else:
        mass = configparser.ConfigParser(inline_comment_prefixes='#')
        mass.read(args.config)
        if args.error is not None:
            mass.set('Parameters', 'ppm_error', str(args.error))
        if args.deltamass is not None:
            mass.set('Parameters', 'min_dm', str(args.deltamass))
    m_proton = mass.getfloat('Masses', 'm_proton')
    def _calcMZ(charge, series, length, plainseq, mods, pos, massconfig, standalone):
        if series == "b":
            seq = plainseq[:length]
            ct = False
            if length == len(plainseq):
                ct = True
            MZ = getTheoMH(charge, seq, mods, pos, True, ct, massconfig, standalone)
            # for m,p in mods,pos:
            #     if p+1 <= length:
            #         MZ += m
        else: # "y"
            seq = plainseq[-length:]
            nt = False
            if length == len(plainseq):
                nt = True
            MZ = getTheoMH(charge, seq, mods, pos, nt, True, massconfig, standalone)
            # for m,p in mods,pos:
            #     if p+1 <= len(plainseq)-length:
            #         MZ += m
        MZ = (MZ + (charge-1)*m_proton) / int(charge)
        return MZ
    proof["CHARGE"] = proof.apply(lambda x: x.FRAGS.count('+') if x.FRAGS.count('+')>1 else 1, axis=1)
    proof["SERIES"] = proof.apply(lambda x: x.FRAGS[0], axis=1)
    proof["LENGTH"] = proof.apply(lambda x: int(re.search(r'\d+', x.FRAGS).group()), axis=1)
    proof["A_LENGTH"] = proof.apply(lambda x: x.LENGTH-1 if x.SERIES=="b" else (len(plainseq)*2)-x.LENGTH, axis=1)
    proof["NM"] = proof.apply(lambda x: _calcMZ(x.CHARGE, x.SERIES, x.LENGTH, plainseq, [0], [0], massconfig, standalone), axis=1)
    proof["MOD"] = proof.apply(lambda x: _calcMZ(x.CHARGE, x.SERIES, x.LENGTH, plainseq, mods, pos, massconfig, standalone), axis=1)
    proof["DIFF"] = (proof.MOD - proof.NM) * proof.CHARGE
    return(proof)

def plotPpmMatrix(sub, plainseq, fppm, dm, frags, zoom, ions, err, specpar, exp_spec,
                  proof, deltamplot, escore, vscore, hscore, BDAGmax, YDAGmax, bpos, ypos,
                  min_dm, outpath, massconfig, standalone, ppm_plot):
    if not standalone:
        mass = massconfig
        outplot = outpath
        if len(str(outplot)) >= 250: # try to shorten very long paths
            fn, ext = os.path.splitext(Path(outplot))
            fn = fn.split("\\")
            fn[-1] = fn[-1][:-len(str(outplot))-250]
            if len(fn[-1]) == 0:
                fn[-1] = "plot"
            counter = 1
            while os.path.exists(Path(outplot)):
                outplot = Path(fn + " (" + str(counter) + ")" + ext)
                counter += 1
    else:
        mass = configparser.ConfigParser(inline_comment_prefixes='#')
        mass.read(args.config)
        if args.error is not None:
            mass.set('Parameters', 'ppm_error', str(args.error))
        if args.deltamass is not None:
            mass.set('Parameters', 'min_dm', str(args.deltamass))
        outplot = os.path.join(outpath, str(sub.Raw) +
                               "_" + str(sub.Sequence) + "_" + str(sub.FirstScan)
                               + "_ch" + str(sub.Charge) + ".pdf")
        if len(str(outplot)) >= 250:
            outplot = os.path.join(outpath, str(sub.Raw) +
                                   "_" + str(sub.Sequence)[:len(str(sub.Sequence))//2] + "_trunc_" + str(sub.FirstScan)
                                   + "_ch" + str(sub.Charge) + ".pdf")
            counter = 0
            while os.path.isfile(outplot):
                counter += 1
                outplot = os.path.join(outpath, str(sub.Raw) +
                                       "_" + str(sub.Sequence)[:len(str(sub.Sequence))//2] + "_trunc_" + str(sub.FirstScan)
                                       + "_ch" + str(sub.Charge) + "_" + str(counter) + ".pdf")
            
    fppm.index = list(frags.by)
    if not hasattr(sub, 'DeltaMassLabel'):
        sub.DeltaMassLabel = "'N/A'"
    dmlabel = ', '.join(re.findall(r'\'(.*?)\'', str(sub.DeltaMassLabel))).split(",")
    mainT = sub.Sequence + "+" + str(round(dm,6))
    mods = [round(float(i),6) for i in re.findall("\d*\.?\d*", sub.Sequence) if i]
    pos = [int(j)-1 for j, k in enumerate(sub.Sequence) if k.lower() == '[']
    for i, p in enumerate(pos):
        if i > 0:
            pos[i] = p - 2 - len(str(mods[i-1]))
    try:
        mainT2 = plainseq
        counter = 0
        for i, p in enumerate(pos):
            if i > 0:
                p = p + sum(len(s)+2 for s in [str(r) for r in dmlabel[0:i-1]])
            mainT2 = mainT2[:p+1] + '[' + dmlabel[counter] + ']' + mainT2[p+1:]
            counter += 1
        mainT2 = mainT2 + "+" + str(round(dm,6))
    except IndexError: # Missing DM Labels, just use mainT
        mainT2 = mainT

    #z  = max(fppm.max())
    
    full_frag_palette = ["#FF0000", "#EA1400", "#D52900", "#C03E00", "#AB5300", "#966800", "#827C00", "#6D9100", "#58A600", "#43BB00",
                    "#2ED000", "#1AE400", "#05F900", "#00EF0F", "#00DA24", "#00C539", "#00B04E", "#009C62", "#008777", "#00728C",
                    "#005DA1", "#0048B6", "#0034CA", "#001FDF", "#000AF4", "#0A06F4", "#1F14DF", "#3421CA", "#482FB6", "#5D3CA1",
                    "#724A8C", "#875777", "#9C6562", "#B0724E", "#C57F39", "#DA8D24", "#EF9A0F", "#FDA503", "#F8A713", "#F3A922",
                    "#EDAB32", "#E8AD41", "#E3AF51", "#DDB160", "#D8B370", "#D3B57F", "#CDB78F", "#C8B99E", "#C3BBAE", "#BEBEBE"]
    
    frag_palette = [full_frag_palette[i] for i in range(len(full_frag_palette)) if i % 2 != 0]
    frag_palette[2] = frag_palette[1]
    frag_palette[1] = "#FF3300" #Substitute ambiguous color
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(22, 26)
    gs = fig.add_gridspec(nrows=10, ncols=6, hspace=1.5)
    #fig.suptitle('VSeq', fontsize=20)
###### INFO TABLE ##
    PTMprob = list(plainseq)
    if not hasattr(sub, 'DeltaMassLabel'):
        sub.DeltaMassLabel = "'N/A'"
    datatable = pd.DataFrame([str(sub.Raw), str(sub.FirstScan), str(sub.Charge), str(sub.RetentionTime), str(round(dm,6)), ', '.join(re.findall(r'\'(.*?)\'', sub.DeltaMassLabel)), str(sub.MH), str(round(escore, 6)), str(round(vscore,6)), str(round(hscore,6))],
                             index=["Raw", "Scan", "Charge", "RT", "DeltaM", "Label", "MH", "Escore", "Vscore", "Hyperscore"])
    #ax2 = fig.add_subplot(3,6,(1,2))
    ax1 = fig.add_subplot(gs[0:3, 0:2])
    ax1.axis('off')
    ax1.axis('tight')
    ytable = plt.table(cellText=datatable.values, rowLabels=datatable.index.to_list(), loc='center', fontsize=15)
    header = [ytable.add_cell(-1,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="lightcoral")]
    header[0].visible_edges = "TLR"
    header[0].get_text().set_text("SCAN INFO")
    header2 = [ytable.add_cell(10,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="lightcoral")]
    header2[0].get_text().set_text("PTM PINPOINTING")
    if abs(dm) >= min_dm:
        ypos = len(plainseq) - (YDAGmax.to_list()[0])
        yaa = PTMprob[::-1][YDAGmax.to_list()[0]]
        if len(plainseq)-YDAGmax.to_list()[0] > len(plainseq):
            ypos = len(plainseq)
            yaa = PTMprob[-1]
        header3 = [ytable.add_cell(11,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="none")]
        header3[0].get_text().set_text(str(PTMprob[bpos])+str(bpos+1))
        header6 = [ytable.add_cell(11,-1, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="none")]
        header6[0].get_text().set_text("B series") 
        header4 = [ytable.add_cell(12,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="none")]
        header4[0].get_text().set_text(str(yaa)+str(ypos))
        header5 = [ytable.add_cell(12,-1, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="none")]
        header5[0].get_text().set_text("Y series")
    else:
        header3 = [ytable.add_cell(11,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="red")]
        header3[0].get_text().set_text("Unmodified Peptide")
    ytable.scale(0.5, 2)
    ytable.set_fontsize(15)
###### FRAGMENTS and DM PINPOINTING##
    # colors = ["red","green","blue","orange","grey"]
    # gradient = []
    # for color in colors:
    #     newcolors = list(Color("red").range_to(Color("green"),12))
    ax2 = fig.add_subplot(gs[0:3, 2:6])
    # ax2.axis('off')
    ax2.axis('tight')
    deltamplot.columns = frags.by
    posmatrix = deltamplot.copy()
    for col in posmatrix.columns:
        posmatrix[col].values[:] = 0
    for row in posmatrix.iterrows():
        posmatrix.loc[row[0]] = deltamplot.loc[row[0]]
    posmatrix[posmatrix<3] = 0
    posmatrix = posmatrix.astype(bool).astype(str)
    posmatrix[posmatrix=='False'] = ''
    posmatrix[posmatrix=='True'] = '⬤'
    posmatrix.columns = list(range(0,posmatrix.shape[1]))
    if not (fppm == 50).all().all():
        posmatrix = posmatrix.loc[list(fppm.T.index.values)]
    # start fixed mod annotation
    ions.reset_index(drop=True,inplace=True)
    ions_check = pd.DataFrame(ions.iloc[posmatrix[(posmatrix=='⬤').any(axis=1)].index.to_list()].MZ.copy())
    ions_check.MZ = ions_check.MZ.astype(float)
    ions_check["ID"] = ions_check.index
    ions_check = pd.merge(proof, ions_check)
    ions_check = ions_check.loc[ions_check.DIFF!=0]
    posmatrix2 = posmatrix.copy() # ☐ ◢
    posmatrix2[posmatrix2=='⬤'] = ''
    for index, row in ions_check.iterrows():
        posmatrix2.at[row.ID, row.A_LENGTH] = '◢'
    # end fixed mod annotation
    # ax2 = fig.add_subplot(3,6,(3,6))
    if abs(dm) >= min_dm and not (fppm == 50).all().all():
        sns.heatmap(fppm.T, annot=posmatrix, fmt='', annot_kws={"size": 40 / np.sqrt(len(fppm.T)), "color": "white", "path_effects":[path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]},
                    cmap=frag_palette, xticklabels=list(frags.by), yticklabels=False, cbar_kws={'label': 'ppm error'})
        sns.heatmap(fppm.T, cmap=frag_palette, cbar=False, annot=posmatrix2, fmt='', yticklabels=False, annot_kws={"size": 40 / np.sqrt(len(fppm.T)), "color": "lightblue", "path_effects":[path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]})
    else:
        sns.heatmap(fppm.T, cmap=frag_palette, xticklabels=list(frags.by), yticklabels=False, cbar_kws={'label': 'ppm error'})
        sns.heatmap(fppm.T, cmap=frag_palette, cbar=False, annot=posmatrix2, fmt='', yticklabels=False, annot_kws={"size": 40 / np.sqrt(len(fppm.T)), "color": "lightblue", "path_effects":[path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]})
    ax2.figure.axes[-1].yaxis.label.set_size(15)
    plt.title(mainT, fontsize=20)
    plt.xlabel("b series --------- y series", fontsize=15)
    plt.ylabel("large--Exp.masses--small", fontsize=15)
    for i, j in enumerate(frags.by):
        plt.axvline(x=frags.by[i], color='white', ls="--")
###### PPM vs INTENSITY(LOG)
    # ax1 = fig.add_subplot(3,6,(7,8))
    ax3 = fig.add_subplot(gs[3:6, 0:2])
    #ax1.plot([1, 1], [15, 15], color='red', transform=ax1.transAxes)  
    plt.yscale("log")
    plt.xlabel("error in ppm______________________ >50", fontsize=15)
    plt.ylabel("'log₁₀(Intensity)'", fontsize=15)
    plt.scatter(zoom, ions.INT, c="lightblue", edgecolors="blue", s=100)
    plt.axvline(x=err, color='tab:blue', ls="--")
###### INTEPRETED V-PLOT ##
    tempfrags = pd.merge(proof, exp_spec)
    tempfrags = tempfrags[tempfrags.REL_INT != 0]
    tempfrags = tempfrags[tempfrags.PPM <= ppm_plot]
    tempfrags.reset_index(inplace=True)
    tempfrags["combFRAGS"] = tempfrags.apply(lambda x: x.FRAGS[0] + str(re.findall(r'\d+', x.FRAGS)[0]), axis=1)
    fragints = {}
    for frag, fragdf in tempfrags.groupby("combFRAGS"):
        fragints[frag] = sum(fragdf.REL_INT)
    ax4 = fig.add_subplot(gs[3:6, 2:6])
    # ax6 = fig.add_subplot(3,6,(9,12))
    interdf = pd.DataFrame(0,index=range(int(len(frags)/2)),columns=range(int(len(frags))))
    interdf.columns = frags.by
    # intlist = []
    for column in fppm.T:
        if (fppm.T[column]<50).any():
            try:
                interdf[column][int(column[1:])-1] = None
                interdf[column][int(column[1:])-1] = int(math.log(fragints[column]))
                # intlist.append(math.log(fragints[column]))
            except KeyError:
                interdf[column][int(column[1:])-1] = 0
                # intlist.append(0)
        # else:
            # intlist.append(0)
    res = sns.heatmap(interdf, cmap="Blues", xticklabels=list(frags.by), yticklabels=False, cbar_kws={'label': 'log₁₀(Relative Intensity)'})
    for _, spine in res.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
    ax4.figure.axes[-1].yaxis.label.set_size(15)
    plt.title(mainT2, fontsize=20)
    plt.xlabel("b series --------- y series", fontsize=15)
    plt.ylabel("large--Exp.masses--small", fontsize=15)
###### SCAN INFO ##
    ## SEQUENCE ##
    ax5 = fig.add_subplot(gs[6:7,0:6])
    # ax3 = fig.add_subplot(4,6,(19,23))
    observed = list(interdf.columns[(interdf > 0).any()])
    fragsb = list(frags.by[0:int(len(frags)/2)])
    fragsy = list(frags.by[int(len(frags)/2):int(len(frags))])
    colordf = pd.DataFrame({"AA":list(plainseq), "B":fragsb, "Y":fragsy})
    colordf["sumB"] = [1 if i in observed else 0 for i in colordf.B]
    colordf["cumsumB"] = colordf.sumB[::-1].cumsum()
    colordf["sumY"] = [1 if i in observed else 0 for i in colordf.Y]
    colordf["cumsumY"] = colordf.sumY.cumsum()
    colordf["cumsumT"] = colordf.cumsumB + colordf.cumsumY
    # if sum(list(map(lambda x: x in observed, fragsb))) >= sum(list(map(lambda x: x in observed, fragsy))):
    mypalette = sns.color_palette("coolwarm", max(colordf.cumsumT)+1)
    colordf["colorT"] = [mypalette[i] for i in colordf.cumsumT] # TODO this won't ever reach maximum
    points = np.ones(len(plainseq))
    ax5.scatter(list(range(len(points))), 2.5*points, facecolors='none', edgecolors='none', marker='o', s=1)
    ax5.scatter(list(range(len(points))), -0.5*points, facecolors='none', edgecolors='none', marker='o', s=1)
    marker_style = dict(color='black', linestyle=' ', marker='o',
                        markersize=35, markerfacecoloralt='tab:red')
    color = ["tab:green" if i in observed else 'white' for i in fragsy]
    ax5.scatter(list(range(len(points))), 1.85*points, c=color, marker='$\u25AC$', s=1500)
    ax5.scatter([s - 0.4 for s in list(range(len(points)))], 2*points-0.2, c=color, marker='$\u2503$', s=600)
    counter = 0
    for x in fragsy:
        weight = 'normal'
        if x in observed: weight = 'bold'
        plt.annotate(x, (counter,2), textcoords="offset points", xytext=(2,10),
                     ha='center', size = 15, color = "black", weight = weight) 
        counter += 1
    # ax5.plot(1 * points, fillstyle="none", **marker_style) 
    color = ['black' if i in pos else 'white' for i,j in enumerate(plainseq)]
    points2 = np.ones(len(plainseq))
    if len(plainseq) < 30:
        color += ['white'] * (30-len(plainseq))
        points2 = np.ones(30)
    if abs(dm) >= min_dm:
        color[bpos] = 'red'
        if len(plainseq)-YDAGmax.to_list()[0] > len(plainseq):
            color[len(plainseq)-1] = 'red'
        else:
            color[abs(len(plainseq)-(YDAGmax.to_list()[0]+1))] = 'red'
    ax5.scatter(list(range(len(points2))), 1*points2, facecolors='none', edgecolors=color, marker='o', s=1200)
    # if len(plainseq) < 30:
    #     extra = np.ones(30 - len(plainseq))
    #     ax5.scatter(list(range(len(extra))), 1*extra, facecolors='none', edgecolors='none', marker='o', s=1000)
    counter = 0
    for x in plainseq:
        color = 'gold'
        if fragsb[counter] in observed:
            color = 'limegreen'
        if fragsy[counter] in observed:
            color = 'limegreen'
        if set([fragsb[counter], fragsy[counter]]).issubset(observed): color = 'forestgreen'
        text = plt.annotate(x, (counter,1), textcoords="offset points", xytext=(0,-10),
                     ha='center', size = 30, color = colordf.colorT[counter], weight='bold')
        text.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='black')])
        counter += 1
    color = ["tab:green" if i in observed else 'white' for i in fragsb]
    ax5.scatter(list(range(len(points))), 0.025*points, c=color, marker='$\u25AC$', s=1500)
    ax5.scatter([s + 0.4 for s in list(range(len(points)))], 0*points+0.2, c=color, marker='$\u2502$', s=600)
    counter = 0
    for x in fragsb:
        weight = 'normal'
        if x in observed:
            weight = 'bold'
        plt.annotate(x, (counter,0), textcoords="offset points", xytext=(2,-22),
                     ha='center', size = 15, color = "black",  weight = weight) 
        counter += 1
    # ax5.scatter([-1.5]*len(mypalette), list(np.linspace(0,2,len(mypalette))),
    #             c=mypalette, marker='$\u25AA$', s=300)
    # ax5.scatter([-1.2]*len(mypalette), list(np.linspace(0,2,len(mypalette))),
    #             c=mypalette, marker='$\u25AA$', s=300)
    # xs, ys, zs = colored_line(np.array([-1.5]*len(mypalette)),
    #                           np.array(list(np.linspace(0,2,len(mypalette)))),
    #                           line_width = .01)
    for i in list(range(0,max(colordf.cumsumT)+1)):
        ax5.plot([-1.5,-1], [list(np.linspace(0,2,len(mypalette)))[i],list(np.linspace(0,2,len(mypalette)))[i]],
                 c=mypalette[i], lw=5)
        if i in np.round(np.linspace(0, len(list(range(0,max(colordf.cumsumT)+1))) - 1, 5)).astype(int):
            plt.annotate(str(i),(-2.2,list(np.linspace(0,2,len(mypalette)))[i]-0.1))
    plt.annotate("N. times\nobserved", (-2.8,0.35), **{'rotation':'vertical', 'ha':'center'})
    ax5.set_axis_off()
###### M/Z vs INTENSITY ##
    proof.FRAGS = proof.apply(lambda x: str(x.FRAGS)+"#" if x.DIFF!=0 else x.FRAGS, axis=1)
    tempfrags = pd.merge(proof, exp_spec)
    tempfrags = tempfrags[tempfrags.REL_INT != 0]
    tempfrags.reset_index(inplace=True)
    ax6 = fig.add_subplot(gs[7:10,0:6])
    # ax4 = fig.add_subplot(3,6,(13,17))
    plt.title(specpar, color="darkblue", fontsize=20)
    plt.xlabel("m/z", fontsize=15)
    plt.ylabel("Relative Intensity", fontsize=15)
    plt.yticks(rotation=90, va="center")
    plt.plot(exp_spec.MZ, exp_spec.CORR_INT, linewidth=0.5, color="darkblue")
    for i, txt in enumerate(tempfrags.FRAGS):
        if "b" in txt:
            txtcolor = "red"
        if "y" in txt:
            txtcolor = "blue"
        ax6.annotate(txt, (tempfrags.MZ[i], tempfrags.CORR_INT[i]), color=txtcolor, fontsize=20, ha="center")
        plt.axvline(x=tempfrags.MZ[i], color='orange', ls="--")
    gs.tight_layout(fig)
    # plt.tight_layout()
    #plt.show()
    fig.savefig(outplot)  
    fig.clear()
    plt.close(fig)
    return

def plotIntegration(sub, mz, scanrange, mzrange, bin_width, t_poisson, mzmlpath, out, n_workers):
    ''' Integrate and save apex list and plot to files. '''
    outpath = os.path.join(out, str(sub.Raw) +
                           "_" + str(sub.Sequence) + "_" + str(sub.FirstScan)
                           + "_ch" + str(sub.Charge) + "_Integration.csv")
    outplot = os.path.join(out, str(sub.Raw) +
                           "_" + str(sub.Sequence) + "_" + str(sub.FirstScan)
                           + "_ch" + str(sub.Charge) + "_Integration.pdf")
    apex_list, apexonly = ScanIntegrator.Integrate(sub.FirstScan, mz, scanrange,
                                                   mzrange, bin_width, mzmlpath,
                                                   n_workers)
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
    theomh = parental + dm + (sub.Charge-1)*mass.getfloat('Masses', 'm_proton')
    # mean_aa = np.mean([float(dict(mass._sections['Aminoacids'])[aa] )for aa in dict(mass._sections['Aminoacids'])])
    avg_aa = 111.1254 # Dalton
    C13 = 1.003355 # Dalton
    est_C13 = (0.000594 * theomh) - 0.03091
    poisson_df = pd.DataFrame(list(range(0,9)))
    poisson_df.columns = ["n"]
    poisson_df["theomh"] = np.arange(theomh, theomh+8.5*C13, C13)
    poisson_df["theomz"] = poisson_df.theomh / sub.Charge
    poisson_df["Poisson"] = poisson_df.apply(lambda x: scipy.stats.poisson.pmf(x.n, est_C13), axis=1)
    poisson_df["cumsum"] = poisson_df.Poisson.cumsum()
    poisson_df = pd.concat([poisson_df[poisson_df["cumsum"]<t_poisson], poisson_df[poisson_df["cumsum"]>=t_poisson].head(1)])
    poisson_df["n_poisson"] = poisson_df.Poisson/poisson_df.Poisson.sum()
    # Select experimental peaks within tolerance
    # apexonly2 = apexonly[apexonly.SUMINT>0]
    apexonly2 = apexonly[apexonly.APEX==True].copy()
    poisson_df["closest"] = [min(apexonly2.BIN, key=lambda x:abs(x-i)) for i in list(poisson_df.theomz)] # filter only those close to n_poisson
    poisson_df["dist"] = abs(poisson_df.theomz - poisson_df.closest)
    poisson_filtered = poisson_df[poisson_df.dist<=bin_width*4].copy()
    if len(apexonly2) <= 0:
        logging.info("\t\t\t\tNot enough information in the spectrum! 0 apexes found.")
        return
    poisson_filtered["exp_peak"] = poisson_filtered.apply(lambda x: min(list(apexonly2.BIN), key=lambda y:abs(y-x.theomz)), axis=1)
    poisson_filtered = poisson_filtered[poisson_filtered.exp_peak>=0]
    poisson_filtered["exp_int"] = poisson_filtered.apply(lambda x: float(apexonly2[apexonly2.BIN==x.exp_peak].SUMINT), axis=1)
    int_total = poisson_filtered.exp_int.sum()
    poisson_df["P_compare"] = poisson_df.apply(lambda x: x.n_poisson*int_total, axis=1)
    poisson_df["exp_int"] = poisson_df.apply(lambda x: float(apexonly2[apexonly2.BIN==x.exp_peak].SUMINT) if x.dist<=bin_width*4 else 0, axis=1)
    # Plots
    ScanIntegrator.PlotIntegration(poisson_df, mz, apex_list, apexonly, outplot)
    return

def doVseq(mode, index_offset, sub, tquery, fr_ns, index2, spectra, spectra_n, min_dm, min_match, err, outpath,
           standalone, massconfig, dograph, min_hscore, ppm_plot, int_perc,
           squery=0, sindex=0, eindex=0, calc_hs=1, hs=0, sortby=None, od=None):
    if not standalone:
        mass = massconfig
    else:
        logging.info("\t\t\tDM Operations...")
        mass = configparser.ConfigParser(inline_comment_prefixes='#')
        with io.open(args.config, "r", encoding="utf-8") as my_config:
            mass.readfp(my_config)
        if args.error is not None:
            mass.set('Parameters', 'ppm_error', str(args.error))
        if args.deltamass is not None:
            mass.set('Parameters', 'min_dm', str(args.deltamass))
    ## SEQUENCE ##
    sub.Sequence = str(sub.Sequence).upper()
    plainseq = ''.join(re.findall("[A-Z]+", sub.Sequence))
    mods = [round(float(i),6) for i in re.findall("\d*\.?\d*", sub.Sequence) if i]
    pos = [int(j)-1 for j, k in enumerate(sub.Sequence) if k.lower() == '[']
    acc_pos = 0
    for i, p in enumerate(pos):
        if i > 0:
            pos[i] = p - 2 - len(str(mods[i-1])) - acc_pos
            acc_pos += len(str(mods[i-1])) + 2
    ## DM ##
    parental = getTheoMH(sub.Charge, plainseq, mods, pos, True, True, massconfig, standalone)
    mim = sub.MH
    if hasattr(sub, 'test_dm'):
        dm = sub.test_dm
    else:
        dm = mim - parental
    #parentaldm = parental + dm
    #dmdm = mim - parentaldm
    #query = tquery[(tquery["CHARGE"]==sub.Charge) & (tquery["SCANS"]==sub.FirstScan)]
    exp_spec, ions, spec_correction = expSpectrum(fr_ns, index_offset, sub.FirstScan, index2, mode, int_perc, spectra, spectra_n, squery, sindex, eindex, od)
    # with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_workers) as executor:   
    #     a
    theo_spec = theoSpectrum(plainseq, mods, pos, len(ions), 0, massconfig, standalone)
    terrors, terrors2, terrors3, texp = errorMatrix(ions.MZ, theo_spec, massconfig, standalone)
    
    ## DM OPERATIONS ##
    dm_theo_spec = theoSpectrum(plainseq, mods, pos, len(ions), dm, massconfig, standalone)
    dmterrors, dmterrors2, dmterrors3, dmtexp = errorMatrix(ions.MZ, dm_theo_spec, massconfig, standalone)
    dmterrorsmin = pd.DataFrame(np.array([dmterrors, dmterrors2, dmterrors3]).min(0)) # Parallel minima
    parcialdm = dmterrorsmin
    dmfppm = dmterrorsmin[(dmterrorsmin < 300).sum(axis=1) >= 0.01*len(dmterrorsmin.columns)]
    dmfppm_fake = pd.DataFrame(50, index=list(range(0,len(plainseq)*2)), columns=list(range(0,len(plainseq)*2)))
    if dmfppm.empty: dmfppm = dmfppm_fake 
    dmfppm = dmfppm.where(dmfppm < 50, 50) # No values greater than 50
    
    ## FRAGMENT NAMES ##
    frags = makeFrags(len(plainseq))
    dmterrors.columns = frags.by
    dmterrors2.columns = frags.by2
    dmterrors3.columns = frags.by3
    
    ## ASSIGN IONS WITHIN SPECTRA ##
    assign = assignIons(theo_spec, dm_theo_spec, frags, dm, min_dm, massconfig, standalone)
    
    ## PPM ERRORS ##
    if sub.Charge == 2:
        ppmfinal = pd.DataFrame(np.array([terrors, terrors2]).min(0))
        parcial = ppmfinal.copy()
        if dm != 0: ppmfinal = pd.DataFrame(np.array([terrors, terrors2, dmterrors, dmterrors2]).min(0))
    elif sub.Charge < 2:
        ppmfinal = pd.DataFrame(np.array([terrors]).min(0))
        parcial = ppmfinal.copy()
        if dm != 0: ppmfinal = pd.DataFrame(np.array([terrors, dmterrors]).min(0))
    elif sub.Charge >= 3:
        ppmfinal = pd.DataFrame(np.array([terrors, terrors2, terrors3]).min(0))
        parcial = ppmfinal.copy()
        if dm != 0: ppmfinal = pd.DataFrame(np.array([terrors, terrors2, terrors3, dmterrors, dmterrors2, dmterrors3]).min(0))
    else:
        sys.exit('ERROR: Invalid charge value!')
    ppmfinal["minv"] = ppmfinal.min(axis=1)
    zoom = ppmfinal.apply(lambda x: random.randint(50, 90) if x.minv > 50 else x.minv , axis = 1)
    minv = ppmfinal["minv"]
    ppmfinal = ppmfinal.drop("minv", axis=1)
    fppm = ppmfinal[(ppmfinal < 50).sum(axis=1) >= 0.001] 
    fppm = fppm.T
    if fppm.empty:
        fppm = pd.DataFrame(50, index=list(range(0,len(plainseq)*2)), columns=list(range(0,len(plainseq)*2)))
        fppm = fppm.T
    
    # if dograph or standalone:
    ## ABLINES ##
    proof, ok = makeAblines(texp, minv, assign, ions, min_match)
    if not ok:
        proofb = proof
        proofy = proof
    else:
        proof.INT = proof.INT * spec_correction
        proof.INT[proof.INT > max(exp_spec.REL_INT)] = max(exp_spec.REL_INT) - 3
        proofb = proof[proof.FRAGS.str.contains("b")]
        proofy = proof[proof.FRAGS.str.contains("y")]
    
    ## SELECT MAXIMUM PPM ERROR TO CONSIDER ## #TODO: Make this a parameter
    fppm[fppm>50] = 50 #TODO: this does not match Rvseq values - too many columns
    parcial[parcial<50] = 2
    parcial[parcial>50] = 0
    parcialdm[parcialdm<50] = 3
    parcialdm[parcialdm>50] = 0
    pppmfinal = ppmfinal.copy()
    pppmfinal[pppmfinal<=300] = 1
    pppmfinal[pppmfinal>300] = 0

    deltamplot, deltaplot = deltaPlot(parcialdm, parcial, pppmfinal)
    #if fppm.empty: fppm = pd.DataFrame(50, index=list(range(0,len(plainseq)*2)), columns=list(range(0,len(plainseq)*2)))
    #z = max(fppm.max())
    
    ## EXPERIMENTAL INTENSITIES MATRIX (TARGET) ##
    #frv2 = ions.INT
    
    ## Q-SCORE AND E-SCORE ##
    qscore, escore = qeScore(ppmfinal, ions.INT, err)
    #matched_ions = qscore.shape[0]
    
    #if dograph or standalone:
    pepmass = tquery[tquery.SCANS == sub.FirstScan].iloc[0]
    specpar = "MZ=" + str(round(pepmass.MZ, 6)) + ", " + "Charge=" + str(int(sub.Charge)) + "+"
    
    BDAGmax, YDAGmax, bpos, ypos = asBY(deltaplot, sub, len(plainseq))
    
    ## SURVEY SCAN INFORMATION ##
    # TODO: dta files required
    
    ## SCORE ##
    vscore = vScore(qscore, sub, len(plainseq), proofb, proofy, assign)
    if "SPECTRUM" not in sub:
        sub['SPECTRUM'] = locateScan(sub.FirstScan, mode, fr_ns, spectra, spectra_n, index2, int_perc, od)
    if calc_hs != 0:
        hscore, nions, bions, yions, intions, dm_pos = scoreVseq(sub, plainseq, mass, err, dm,
                                                                 mass.getfloat('Masses', 'm_proton'),
                                                                 mass.getfloat('Masses', 'm_hydrogen'),
                                                                 mass.getfloat('Masses', 'm_oxygen'),
                                                                 mass.getfloat('Parameters', 'score_mode'),
                                                                 mass.getfloat('Parameters', 'full_y'))
    else: hscore = hs
    if sortby == "ions_matched":
        sortby = int(nions)
    elif sortby == "e_score":
        sortby = float(escore)
    elif sortby == "product":
        sortby = float(nions * escore)
    elif sortby == "v_score":
        sortby = float(vscore)
    elif sortby == "hyperscore":
        sortby = float(hscore)
    else:
        sortby = float(vscore)
        
    ## PLOTS ##
    if standalone:
        logging.info("\t\t\tPlotting...") # TODO should we have min_vscore take effect here or not
    if dograph and sortby >= min_hscore:
        if proof.iloc[0].MZ == 0: return(0, 0, 0, 0, 0)
        else:
            proof = locateFixedMods(proof, plainseq, mods, pos, massconfig, standalone)
            try:
                plotPpmMatrix(sub, plainseq, fppm, dm, frags, zoom, ions, err, specpar, exp_spec,
                              proof, deltamplot, escore, vscore, hscore, BDAGmax, YDAGmax, bpos, ypos,
                              min_dm, outpath, massconfig, standalone, ppm_plot)
            except:
                logging.exception("\t\t\tPlotting failed.", exc_info=1)
    if standalone:
        if not args.integrate:
            logging.info("\t\t\tDone.")
        return(vscore, escore, hscore, dm, intions)
    elif dograph and not standalone:
        return
    elif not dograph and not standalone:
        return(vscore, escore, hscore, nions, bions, yions, ppmfinal, frags)
    else:
        return

def main(args):
    '''
    Main function
    '''
    ## USER PARAMS TO ADD ##
    err = float(mass._sections['Parameters']['fragment_tolerance'])
    min_dm = float(mass._sections['Parameters']['min_dm'])
    min_match = int(mass._sections['Parameters']['min_ions_matched'])
    ppm_plot = float(mass._sections['Parameters']['ppm_plot'])
    min_hscore = float(mass._sections['Parameters']['vseq_threshold'])
    int_scanrange = float(mass._sections['Parameters']['int_scanrange'])
    int_mzrange = float(mass._sections['Parameters']['int_mzrange'])
    int_binwidth = float(mass._sections['Parameters']['int_binwidth'])
    t_poisson = float(mass._sections['Parameters']['poisson_threshold'])
    int_perc = float(mass._sections['Parameters']['intensity_percent_threshold'])
    # try:
    #     arg_dm = float(args.deltamass)
    # except ValueError:
    #     sys.exit("Minimum deltamass (-d) must be a number!")
    # Set variables from input file
    logging.info("Reading input file")
    #scan_info = pd.read_csv(args.infile, sep=r'\,|\t', engine="python")
    scan_info = pd.read_csv(args.infile, sep='\t', engine="python")
    scan_info = scan_info[scan_info.Sequence.notna()]
    scan_info.FirstScan = scan_info.FirstScan.astype(int)
    exps = list(scan_info.Raw.unique())
    for exp in exps:
        #logging.info("Experiment: " + str(exp))
        exp = str(exp).replace(".txt","").replace(".raw","").replace(".mgf","")
        sql = scan_info.loc[scan_info.Raw == exp]
        #data_type = sql.type[0]
        sql.reset_index(inplace=True, drop=True)
        pathdict = prepareWorkspace(exp, sql.msdataDir[0], sql.outDir[0])
        if os.path.isfile(os.path.join(pathdict["msdata"], exp + ".mzML")):
            msdata = os.path.join(pathdict["msdata"], exp + ".mzML")
            mode = "mzml"
            logging.info("\tReading mzML file...")
            # fr_ns = pyopenms.MSExperiment()
            # pyopenms.MzMLFile().load(msdata, fr_ns)
            fr_ns = read_mzml_with_progress(msdata)
            spectra = fr_ns.getSpectra()
            spectra_n = [int(s.getNativeID().split("=")[-1]) for s in spectra]
            index2 = 0
            tquery = getTquery(fr_ns, mode)
            index_offset = 0
        elif os.path.isfile(os.path.join(pathdict["msdata"], exp + ".mgf")):
            msdata = os.path.join(pathdict["msdata"], exp + ".mgf")
            mode = "mgf"
            logging.info("\tReading MGF file...")
            # fr_ns = pd.read_csv(msdata, header=None, sep="\t")
            fr_ns = read_csv_with_progress(msdata, "\t")
            spectra = 0
            spectra_n = 0
            index2 = fr_ns.to_numpy() == 'END IONS'
            tquery = getTquery(fr_ns, mode)
            index_offset = getOffset(fr_ns)
        else:
            logging.info("MGF or mzML file not found in " + str(os.path.join(pathdict["msdata"], exp + ".mgf")) + " or " + str(os.path.join(pathdict["msdata"], exp + ".mzML")))
        tquery.to_csv(os.path.join(pathdict["out"], "tquery_"+ exp + ".csv"), index=False, sep=',', encoding='utf-8')
        for scan in list(sql.FirstScan.unique()):
            subs = sql.loc[sql.FirstScan==scan]
            if len(subs) > 1:
                logging.info("\t\tWarning: " + str(len(subs)) + " entries with the same scan number for this raw were found in input table. Results may be overwritten!")
            for index, sub in subs.iterrows():
                #logging.info(sub.Sequence)
                #seq2 = sub.Sequence[::-1]
                sub2 = sub.copy()
                logging.info("\t\tScan: " + str(scan))
                vscore, escore, hscore, dm, intions = doVseq(mode, index_offset, sub, tquery, fr_ns, index2, spectra, spectra_n, min_dm, min_match, err,
                       pathdict["out"], True, False, True, min_hscore, ppm_plot, int_perc)
                mz = tquery[tquery.SCANS == sub.FirstScan].iloc[0].MZ
                sub.drop("SPECTRUM", inplace=True)
                sub["e-score"] = escore
                sub["v-score"] = vscore
                sub["hyperscore"] = hscore
                sub["DeltaMass"] = dm
                sub["MatchedIonIntensity"] = intions
                sub = pd.DataFrame(sub).T
                outfile = os.path.join(pathdict["out"], "Vseq_Summary.tsv")
                sub.to_csv(outfile, index=False, sep='\t', encoding='utf-8',
                                 mode='a', header=not os.path.exists(outfile))
                if args.integrate:
                    if mode == "mzml":
                        logging.info("\t\t\tIntegrating scans...")
                        plotIntegration(sub2, mz, int_scanrange, int_mzrange,
                                        int_binwidth, t_poisson, msdata,
                                        pathdict["out"], int(args.n_workers)) # outside of doVseq() becuase we don't want it in VseqExplorer
                        logging.info("\t\t\tDone.")
                    elif mode == "mgf":
                        logging.info("\tCannot integrate using MGF files.")
            
if __name__ == '__main__':

    # multiprocessing.freeze_support()
    # parse arguments
    parser = argparse.ArgumentParser(
        description='Vseq',
        epilog='''
        Example:
            python Vseq.py

        ''')
        
    defaultconfig = os.path.join(os.path.dirname(__file__), "Vseq.ini")
    
    parser.add_argument('-i',  '--infile', required=True, help='Input file')
    parser.add_argument('-c', '--config', default=defaultconfig, help='Path to custom config.ini file')
    parser.add_argument('-e', '--error', default=None, help='Maximum ppm error to consider')
    parser.add_argument('-d', '--deltamass', default=None, help='Minimum deltamass to consider')
    parser.add_argument('-n', '--integrate', action="store_true", help='Perform scan integration')
    parser.add_argument('-w',  '--n_workers', type=int, default=4, help='Number of threads/n_workers (default: %(default)s)')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()
    
    # parse config
    mass = configparser.ConfigParser(inline_comment_prefixes='#')
    with io.open(args.config, "r", encoding="utf-8") as my_config:
        mass.read_file(my_config)
    if args.error is not None:
        mass.set('Parameters', 'ppm_error', str(args.error))
    if args.deltamass is not None:
        mass.set('Parameters', 'min_dm', str(args.deltamass))
    # if something is changed, write a copy of ini
    if mass.getint('Logging', 'create_ini') == 1:
        with open(os.path.dirname(args.infile) + '/Vseq.ini', 'w') as newconfig:
            mass.write(newconfig)

    # logging debug level. By default, info level
    log_file = outfile = args.infile[:-4] + 'Vseq_log.txt'
    log_file_debug = outfile = args.infile[:-4] + 'Vseq_log_debug.txt'
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
    try:
        main(args)
    except:
        logging.exception('An error occurred')
    logging.info('end script')
