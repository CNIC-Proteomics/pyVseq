# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:10:14 2022

@author: alaguillog
"""
# mgf = 'S:\\U_Proteomica\\UNIDAD\\DatosCrudos\\LopezOtin\\COVID\\COVID_TMT\\COVID_TMT1_F2.mgf'
# infile = 'C:\\Users\\alaguillog\\GitHub\\vseq_input_data.csv'
# massfile = 'C:\\Users\\alaguillog\\GitHub\\pyVseq\\massfile_original.ini'
# seq2 = kLETEVMq

# import modules
import os
import sys
import argparse
#from colour import Color
import configparser
import itertools
import logging
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
from pathlib import Path
import random
import re
import seaborn as sns
import scipy.stats
import statistics
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

def prepareWorkspace(exp, mgfpath, dtapath, outpath):
    mgfpath = Path(mgfpath)
    dtapath = Path(dtapath)
    outpath = Path(outpath)
    # Get dta path for the experiment
    dtapath = os.path.join(dtapath, exp + ".dta")
    var_name_path = os.path.join(outpath, exp)
    # Create output directory
    if not os.path.exists(outpath):
        os.mkdir(var_name_path)
    logging.info("Experiment: " + exp)
    logging.info("mgfpath: " + str(mgfpath))
    logging.info("dtapath: " + str(dtapath))
    logging.info("outpath: " + str(outpath))
    logging.info("varNamePath: " + str(var_name_path))
    pathdict = {"exp": exp,
                "mgf": mgfpath,
                "dta": dtapath,
                "out": outpath,
                "var_name": var_name_path}
    return pathdict

def getTquery(fr_ns):
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
    tquery[['MZ','INT']] = tquery.PEPMASS.str.split(" ",expand=True,)
    tquery['CHARGE'] = tquery.CHARGE.str[:-1]
    tquery = tquery.drop("PEPMASS", axis=1)
    tquery = tquery.apply(pd.to_numeric)
    return tquery

def getTheoMH(charge, sequence, nt, ct, massconfig, standalone):
    '''    
    Calculate theoretical MH using the PSM sequence.
    '''
    if not standalone:
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
    for aa in sequence:
        if aa.lower() in AAs:
            total_aas += float(AAs[aa.lower()])
        if aa.lower() in MODs:
            total_aas += float(MODs[aa.lower()])
        if aa.islower():
            total_aas += float(MODs['isolab'])
    MH = total_aas - (charge-1)*m_proton
    return MH

def expSpectrum(fr_ns, scan):
    '''
    Prepare experimental spectrum.
    '''
    index1 = fr_ns.loc[fr_ns[0]=='SCANS='+str(scan)].index[0] + 1
    index2 = fr_ns.drop(index=fr_ns.index[:index1], axis=0).loc[fr_ns[0]=='END IONS'].index[0]
    
    ions = fr_ns.iloc[index1:index2,:]
    ions[['MZ','INT']] = ions[0].str.split(" ",expand=True,)
    ions = ions.drop(ions.columns[0], axis=1)
    ions = ions.apply(pd.to_numeric)
    ions["ZERO"] = 0
    #ions["CCU"] = 0.01
    ions["CCU"] = ions.MZ - 0.01
    ions.reset_index(drop=True)
    
    #bind = pd.DataFrame(list(itertools.chain(*set(zip(list(ions['CCU']),list(ions['MZ']))))), columns=["MZ"])
    #bind["REL_INT"] = list(itertools.chain(*set(zip(list(ions['ZERO']),list(ions['INT'])))))
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
    spec_correction = max(ions.INT)/statistics.mean(normspec.INT)
    spec["CORR_INT"] = spec.REL_INT*spec_correction
    spec.loc[spec['CORR_INT'].idxmax()]['CORR_INT'] = max(spec.REL_INT)
    spec["CORR_INT"] = spec.apply(lambda x: max(ions.INT)-13 if x["CORR_INT"]>max(ions.INT) else x["CORR_INT"], axis=1)
    return(spec, ions, spec_correction)

def theoSpectrum(seq, len_ions, dm, massconfig, standalone):
    '''
    Prepare theoretical fragment matrix.

    '''
    if not standalone:
        mass = massconfig
    m_hydrogen = mass.getfloat('Masses', 'm_hydrogen')
    m_oxygen = mass.getfloat('Masses', 'm_oxygen')
    ## Y SERIES ##
    #ipar = list(range(1,len(seq)))
    outy = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        yn = list(seq[i:])
        if i > 0: nt = False
        else: nt = True
        fragy = getTheoMH(0,yn,nt,True, massconfig, standalone) + dm
        outy[i:] = fragy
        
    ## B SERIES ##
    outb = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        bn = list(seq[::-1][i:])
        if i > 0: ct = False
        else: ct = True
        fragb = getTheoMH(0,bn,True,ct, massconfig, standalone) - 2*m_hydrogen - m_oxygen + dm
        outb[i:] = fragb
    
    ## FRAGMENT MATRIX ##
    yions = outy.T
    bions = outb.iloc[::-1].T
    spec = pd.concat([bions, yions], axis=1)
    spec.columns = range(spec.columns.size)
    spec.reset_index(inplace=True, drop=True)
    return(spec)

def errorMatrix(fr_ns, mz, theo_spec, massconfig, standalone):
    '''
    Prepare ppm-error and experimental mass matrices.
    '''
    if not standalone:
        mass = massconfig
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

def makeFrags(seq_len):
    '''
    Name all fragments.
    '''
    frags = pd.DataFrame(np.nan, index=list(range(0,seq_len*2)),
                         columns=["by", "by2", "by3", "bydm", "bydm2"])
    frags.by = ["b" + str(i) for i in list(range(1,seq_len+1))] + ["y" + str(i) for i in list(range(1,seq_len+1))[::-1]]
    frags.by2 = frags.by + "++"
    frags.by3 = frags.by + "+++"
    frags.bydm = frags.by + "*"
    frags.bydm2 = frags.by + "*++"
    return(frags)

def assignIons(theo_spec, dm_theo_spec, frags, dm, arg_dm, massconfig, standalone):
    if not standalone:
        mass = massconfig
    m_proton = mass.getfloat('Masses', 'm_proton')
    assign = pd.concat([frags.by, theo_spec.iloc[0]], axis=1)
    assign.columns = ['FRAGS', '+']
    assign["++"] = (theo_spec.iloc[0]+m_proton)/2
    assign["+++"] = (theo_spec.iloc[0]+2*m_proton)/3
    assign["*"] = dm_theo_spec.iloc[0]
    assign["*++"] = (dm_theo_spec.iloc[0]+m_proton)/2
    
    #c_assign = pd.DataFrame(list(assign["+"]) + list(assign["++"]) + list(assign["+++"]) + list(assign["*"]) + list(assign["*++"]))
    c_assign = pd.DataFrame(list(assign["+"]) + list(assign["++"]) + list(assign["+++"]))
    if dm >= arg_dm:
        c_assign = pd.concat([c_assign, pd.DataFrame(list(assign["*"])), pd.DataFrame(list(assign["*++"]))])
    c_assign.columns = ["MZ"]
    #c_assign["FRAGS"] = list(frags.by) + list(frags.by + "++") + list(frags.by + "+++") + list(frags.by + "*") + list(frags.by + "*++")
    c_assign_frags = pd.DataFrame(list(frags.by) + list(frags.by + "++") + list(frags.by + "+++"))
    if dm >= arg_dm:
        c_assign_frags = pd.concat([c_assign_frags, pd.DataFrame(list(frags.by + "*")), pd.DataFrame(list(frags.by + "*++"))])
    c_assign["FRAGS"] = c_assign_frags
    c_assign["ION"] = c_assign.apply(lambda x: re.findall(r'\d+', x.FRAGS)[0], axis=1)
    c_assign["CHARGE"] = c_assign.apply(lambda x: x.FRAGS.count('+'), axis=1).replace(0, 1)
    return(c_assign)

def makeAblines(texp, minv, assign, ions):
    masses = pd.concat([texp[0], minv], axis = 1)
    matches = masses[(masses < 51).sum(axis=1) >= 0.001]
    matches.reset_index(inplace=True, drop=True)
    if len(matches) == 0 or len(matches) == 2:
        matches = pd.DataFrame([[1,3],[2,4]])
    
    matches_ions = pd.DataFrame()
    for mi in list(range(0,len(matches))):
        for ci in list(range(0, len(assign))):
            if abs(matches.iloc[mi,0]-assign.iloc[ci,0])/assign.iloc[ci,0]*1000000 <= 51:
                asign = pd.Series([matches.iloc[mi,0], assign.iloc[ci,1], matches.iloc[mi,1]])
                matches_ions = pd.concat([matches_ions, asign], ignore_index=True, axis=1)
                #matches.iloc[2,1]
    matches_ions = matches_ions.T
    matches_ions.columns = ["MZ","FRAGS","PPM"]
    proof = pd.merge(matches_ions, ions[['MZ','INT']], how="left", on="MZ")
    if len(proof)==0:
        mzcycle = itertools.cycle([ions.MZ.iloc[0], ions.MZ.iloc[1]])
        proof = pd.concat([matches_ions, pd.Series([next(mzcycle) for count in range(len(matches_ions))], name="INT")], axis=1)
    return(proof)

def deltaPlot(parcialdm, parcial, ppmfinal):
    deltamplot = pd.DataFrame(np.array([parcialdm, parcial, ppmfinal]).max(0)) # Parallel maxima
    deltamplot = deltamplot[(deltamplot > 0).sum(axis=1) >= 0.01*deltamplot.shape[1]]
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
    deltaplot = pd.DataFrame([pd.Series(rplot), pd.Series(cplot)]).T
    if deltaplot.shape[0] != 0:
        deltaplot.columns = ["row", "deltav2"]
        deltaplot["deltav1"] = deltamplot.shape[0] - deltaplot.row
    else:
        deltaplot = pd.concat([deltaplot, pd.Series([0])],axis=0)
        deltaplot.columns = ["row"]
        deltaplot["deltav2"] = 0
        deltaplot["deltav1"] = 0
    return(deltamplot, deltaplot)

def qeScore(ppmfinal, int2, err):
    int2.reset_index(inplace=True, drop=True)
    ppmfinal["minv"] = ppmfinal.apply(lambda x: x.min() , axis = 1)
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

def asBY(deltaplot, sub):
    asB = pd.DataFrame()
    asY = pd.DataFrame()
    for i in list(range(0,deltaplot.shape[0])):
        if deltaplot.deltav2[i] <= len(sub.Sequence)-1:
            asB = pd.concat([asB, deltaplot.iloc[i]], axis=1)
        if deltaplot.deltav2[i] > len(sub.Sequence)-1:
            asY = pd.concat([asY, deltaplot.iloc[i]], axis=1)
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
                    if BYDAG.dist.iloc[i] <= 7:
                        BYDAG.value.iloc[j] = BYDAG.value.iloc[i] + 1
                    elif BYDAG.dist.iloc[i] > 7:
                        BYDAG.value.iloc[j] = 0
            BYDAG.consec = BYDAG.value + 1
    BDAGmax = BDAG[BDAG.value == BDAG.value.max()]
    if len(BDAG) == 1:
        BDAGmax.row = 0
    YDAGmax = YDAG[YDAG.value == YDAG.value.max()]
    YDAGmax = YDAGmax.row - len(sub.Sequence)
    return(BDAGmax, YDAGmax)

def vScore(qscore, sub, proofb, proofy, assign):
    '''
    Calculate vScore.
    '''
    
    ## SS1 ##
    if len(qscore) <= (len(sub.Sequence)*2)/4:
        SS1 = 1
    if len(qscore) > (len(sub.Sequence)*2)/4:
        SS1 = 2
    if len(qscore) > (len(sub.Sequence)*2)/3:
        SS1 = 3
    if len(qscore) > (len(sub.Sequence)*2)/2:
        SS1 = 4
    
    ## SS2 ##
    proofb_vscore = proofb[proofb.PPM < 20]
    proofb_vscore.reset_index(inplace=True)
    if len(proofb_vscore) == 0:
        SS2 = 0
    if len(proofb_vscore) != 0:
        SS2 = SS1 * 1.5
    if len(proofb_vscore) > len(sub.Sequence)/3:
        SS2 = SS1 * 2.5
    if len(proofb_vscore) > (len(sub.Sequence)*2)/3:
        SS2 = SS1 * 3.5
    
    ## SS3 ##
    proofy_vscore = proofy[proofy.PPM < 20]
    proofy_vscore.reset_index(inplace=True)
    if len(proofy_vscore) == 0:
        SS3 = 0
    if len(proofy_vscore) != 0:
        SS3 = SS1 * 3
    if len(proofy_vscore) > len(sub.Sequence)/3:
        SS3 = SS1 * 5
    if len(proofy_vscore) > (len(sub.Sequence)*2)/3:
        SS3 = SS1 * 7

    ## SS4 ##
    SS4b = SS4y = 0
    temp = []
    for proofby_vscore, SS4 in [[proofb_vscore, SS4b], [proofy_vscore, SS4y]]:
        if len(proofby_vscore) > 1:
            proofby_vscore = pd.concat([proofby_vscore, pd.merge(proofby_vscore, assign, on="FRAGS")[["ION", "CHARGE"]]], axis=1)
            proofby_vscore = proofby_vscore.sort_values(by="CHARGE")
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
    SS6 = SS6b + SS6y/2
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
        
    vscore = (SS1 + SS2 + SS3 + Kerr + (SS4 * SS5)) * Kv / len(sub.Sequence)
    return(vscore)

def plotPpmMatrix(sub, fppm, dm, frags, zoom, ions, err, specpar, exp_spec,
                  proof, deltamplot, escore, vscore, BDAGmax, YDAGmax, min_dm,
                  outpath, massconfig, standalone):
    if not standalone:
        mass = massconfig
    fppm.index = list(frags.by)
    mainT = sub.Sequence + "+" + str(round(dm,6)) 
    z  = max(fppm.max())
    outplot = os.path.join(outpath, str(sub.Raw) +
                           "_" + str(sub.Sequence) + "_" + str(sub.FirstScan)
                           + ".pdf")
    
    frag_palette = ["#FF0000", "#EA1400", "#D52900", "#C03E00", "#AB5300", "#966800", "#827C00", "#6D9100", "#58A600", "#43BB00",
                    "#2ED000", "#1AE400", "#05F900", "#00EF0F", "#00DA24", "#00C539", "#00B04E", "#009C62", "#008777", "#00728C",
                    "#005DA1", "#0048B6", "#0034CA", "#001FDF", "#000AF4", "#0A06F4", "#1F14DF", "#3421CA", "#482FB6", "#5D3CA1",
                    "#724A8C", "#875777", "#9C6562", "#B0724E", "#C57F39", "#DA8D24", "#EF9A0F", "#FDA503", "#F8A713", "#F3A922",
                    "#EDAB32", "#E8AD41", "#E3AF51", "#DDB160", "#D8B370", "#D3B57F", "#CDB78F", "#C8B99E", "#C3BBAE", "#BEBEBE"]
    
    fig = plt.figure()
    fig.set_size_inches(22, 15)
    #fig.suptitle('VSeq', fontsize=20)
    ## PPM vs INTENSITY(LOG)
    ax1 = fig.add_subplot(2,6,(1,2))
    #ax1.plot([1, 1], [15, 15], color='red', transform=ax1.transAxes)  
    plt.yscale("log")
    plt.xlabel("error in ppm______________________ >50", fontsize=15)
    plt.ylabel("intensity(log)", fontsize=15)
    plt.scatter(zoom, ions.INT, c="lightblue", edgecolors="blue", s=100)
    plt.axvline(x=err, color='tab:blue', ls="--")
    ## FRAGMENTS and DM PINPOINTING##
    # colors = ["red","green","blue","orange","grey"]
    # gradient = []
    # for color in colors:
    #     newcolors = list(Color("red").range_to(Color("green"),12))
    deltamplot.columns = frags.by
    posmatrix = deltamplot.copy()
    for col in posmatrix.columns:
        posmatrix[col].values[:] = 0
    for row in posmatrix.iterrows():
        posmatrix.loc[row[0]] = deltamplot.loc[row[0]]
    posmatrix[posmatrix<3] = 0
    posmatrix = posmatrix.astype(bool).astype(str)
    posmatrix[posmatrix=='False'] = ''
    posmatrix[posmatrix=='True'] = '★'
    posmatrix.columns = list(range(0,posmatrix.shape[1]))
    posmatrix = posmatrix.loc[list(fppm.T.index.values)]
    ax5 = fig.add_subplot(2,6,(3,6))
    sns.heatmap(fppm.T, annot=posmatrix, fmt='', annot_kws={"color": "white", "path_effects":[path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()]},
                cmap=frag_palette, xticklabels=list(frags.by), yticklabels=False, cbar_kws={'label': 'ppm error'})
    ax5.figure.axes[-1].yaxis.label.set_size(15)
    plt.title(mainT, fontsize=20)
    plt.xlabel("b series --------- y series", fontsize=15)
    plt.ylabel("large--Exp.masses--small", fontsize=15)
    for i, j in enumerate(frags.by):
        plt.axvline(x=frags.by[i], color='white', ls="--")
    # for i, c in enumerate(fppm.T.columns):
    #     for j, v in enumerate(fppm.T[c]):
    #         if v < z:
    #             ax5.text(i + 0.5, j + 0.5, '★', color='white', size=20, ha='center', va='center')
    #posmatrix = fppm.T.copy()
    # for i, c in enumerate(posmatrix.columns):
    #     for j, v in enumerate(posmatrix[c]):
    #         if v == 1:
    #             text = ax5.text(i + 0.5, j - 1.5, '★', color='white', size=20, ha='center', va='center')
    #             text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'), path_effects.Normal()])
    ## M/Z vs INTENSITY ##
    tempfrags = pd.merge(proof, exp_spec)
    tempfrags = tempfrags[tempfrags.REL_INT != 0]
    tempfrags.reset_index(inplace=True)
    ax4 = fig.add_subplot(2,6,(7,9))
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
        ax4.annotate(txt, (tempfrags.MZ[i], tempfrags.CORR_INT[i]), color=txtcolor, fontsize=20, ha="center")
        plt.axvline(x=tempfrags.MZ[i], color='orange', ls="--")
    ## INFO TABLE ##
    PTMprob = list(sub.Sequence)
    datatable = pd.DataFrame([str(sub.Raw), str(sub.FirstScan), str(sub.Charge), str(sub.RetentionTime), str(round(dm,6)), str(sub.ExpNeutralMass + mass.getfloat('Masses', 'm_proton')), str(escore), str(vscore)],
                             index=["Raw", "Scan", "Charge", "RT", "DeltaM", "M.Mass", "Escore", "Vscore"])
    ax2 = fig.add_subplot(2,6,(10,11))
    ax2.axis('off')
    ax2.axis('tight')
    ytable = plt.table(cellText=datatable.values, rowLabels=datatable.index.to_list(), loc='center', fontsize=15)
    header = [ytable.add_cell(-1,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="lightcoral")]
    header[0].visible_edges = "TLR"
    header[0].get_text().set_text("SCAN INFO")
    header2 = [ytable.add_cell(8,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="lightcoral")]
    header2[0].get_text().set_text("PTM PINPOINTING")
    if dm >= min_dm:
        header3 = [ytable.add_cell(9,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="none")]
        header3[0].get_text().set_text(str(PTMprob[BDAGmax.row.iloc[0]])+str(BDAGmax.row.iloc[0]))
        header6 = [ytable.add_cell(9,-1, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="none")]
        header6[0].get_text().set_text("B series") 
        header4 = [ytable.add_cell(10,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="none")]
        header4[0].get_text().set_text(str(PTMprob[YDAGmax.to_list()[0]])+str(YDAGmax.to_list()[0]))
        header5 = [ytable.add_cell(10,-1, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="none")]
        header5[0].get_text().set_text("Y series")
    else:
        header3 = [ytable.add_cell(9,0, ytable.get_celld()[(0,0)].get_width(), ytable.get_celld()[(0,0)].get_height(), loc="center", facecolor="red")]
        header3[0].get_text().set_text("Unmodified Peptide")
    ytable.scale(0.5, 2)
    ytable.set_fontsize(15)
    ## SCAN INFO ##
    # ax2 = fig.add_subplot(2,6,(10,11))
    # plt.axis('off')
    # plt.text(0, 0.5,
    #          'Raw='+str(sub.Raw)+'\n'+
    #          'FirstScan='+str(sub.FirstScan)+'\n'+
    #          'Charge='+str(sub.Charge)+'\n'+
    #          'RT='+str(sub.RetentionTime)+'\n'+
    #          'DeltaM='+str(round(dm,6))+'\n'+
    #          'M.Mass='+str(sub.ExpNeutralMass + mass.getfloat('Masses', 'm_proton'))+'\n'+
    #          'Escore='+str(escore)+'\n'+
    #          'Vscore='+str(vscore)+'\n',
    #          fontsize=20,
    #          horizontalalignment='left',
    #          verticalalignment='center',
    #          transform = ax2.transAxes)
    ## MODIFIED RESIDUE CHARACTERIZATION ##
    # PTMprob = list(sub.Sequence)
    # ax3 = fig.add_subplot(2,6,12)
    # plt.axis('off')
    # if dm >= min_dm:
    #     plt.text(-0.5, 0.5,
    #               'PTM pinpointing:'+'\n'+
    #               'Bseries='+ str(PTMprob[BDAGmax.row.iloc[0]])+str(BDAGmax.row.iloc[0])+'\n'+
    #               'Yseries='+ str(PTMprob[YDAGmax[0]])+str(YDAGmax[0])+'\n',
    #               fontsize=20,
    #               horizontalalignment='left',
    #               verticalalignment='center',
    #               transform = ax3.transAxes)
    # else:
    #     plt.text(-0.5, 0.5,
    #               'Unmodified Peptide',
    #               color="red",
    #               fontsize=20,
    #               horizontalalignment='left',
    #               verticalalignment='center',
    #               transform = ax3.transAxes)
    plt.tight_layout()
    #plt.show()
    fig.savefig(outplot)  
    return

def doVseq(sub, tquery, fr_ns, min_dm, err, outpath, standalone, massconfig, dograph):
    if not standalone:
        logging.info("\t\t\tDM Operations...")
        mass = massconfig
    parental = getTheoMH(sub.Charge, sub.Sequence, True, True, massconfig, standalone)
    mim = sub.ExpNeutralMass + mass.getfloat('Masses', 'm_proton')
    dm = mim - parental
    parentaldm = parental + dm
    dmdm = mim - parentaldm
    #query = tquery[(tquery["CHARGE"]==sub.Charge) & (tquery["SCANS"]==sub.FirstScan)]
    exp_spec, ions, spec_correction = expSpectrum(fr_ns, sub.FirstScan)
    theo_spec = theoSpectrum(sub.Sequence, len(ions), 0, massconfig, standalone)
    terrors, terrors2, terrors3, texp = errorMatrix(fr_ns, ions.MZ, theo_spec, massconfig, standalone)
    
    ## DM OPERATIONS ##
    dm_theo_spec = theoSpectrum(sub.Sequence, len(ions), dm, massconfig, standalone)
    dmterrors, dmterrors2, dmterrors3, dmtexp = errorMatrix(fr_ns, ions.MZ, dm_theo_spec, massconfig, standalone)
    dmterrorsmin = pd.DataFrame(np.array([dmterrors, dmterrors2, dmterrors3]).min(0)) # Parallel minima
    parcialdm = dmterrorsmin
    dmfppm = dmterrorsmin[(dmterrorsmin < 300).sum(axis=1) >= 0.01*len(dmterrorsmin.columns)]
    dmfppm_fake = pd.DataFrame(50, index=list(range(0,len(sub.Sequence)*2)), columns=list(range(0,len(sub.Sequence)*2)))
    if dmfppm.empty: dmfppm = dmfppm_fake 
    dmfppm = dmfppm.where(dmfppm < 50, 50) # No values greater than 50
    
    ## FRAGMENT NAMES ##
    frags = makeFrags(len(sub.Sequence))
    dmterrors.columns = frags.by
    dmterrors2.columns = frags.by2
    dmterrors3.columns = frags.by3
    
    ## ASSIGN IONS WITHIN SPECTRA ##
    assign = assignIons(theo_spec, dm_theo_spec, frags, dm, min_dm, massconfig, standalone)
    
    ## PPM ERRORS ##
    if sub.Charge == 2:
        ppmfinal = pd.DataFrame(np.array([terrors, terrors2]).min(0))
        parcial = ppmfinal
        if dm != 0: ppmfinal = pd.DataFrame(np.array([terrors, terrors2, dmterrors, dmterrors2]).min(0))
    elif sub.Charge >= 3:
        ppmfinal = pd.DataFrame(np.array([terrors, terrors2, terrors3]).min(0))
        parcial = ppmfinal
        if dm != 0: ppmfinal = pd.DataFrame(np.array([terrors, terrors2, terrors3, dmterrors, dmterrors2, dmterrors3]).min(0))
    else:
        sys.exit('ERROR: Charge is not 2 or 3')
    ppmfinal["minv"] = ppmfinal.apply(lambda x: x.min() , axis = 1)
    zoom = ppmfinal.apply(lambda x: random.randint(50, 90) if x.minv > 50 else x.minv , axis = 1)
    minv = ppmfinal["minv"]
    ppmfinal = ppmfinal.drop("minv", axis=1)
    fppm = ppmfinal[(ppmfinal < 50).sum(axis=1) >= 0.001] 
    fppm = fppm.T
    if fppm.empty: fppm = pd.DataFrame(50, index=list(range(0,len(sub.Sequence)*2)), columns=list(range(0,len(sub.Sequence)*2)))
    
    if dograph or standalone:
        ## ABLINES ##
        proof = makeAblines(texp, minv, assign, ions)
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
        if fppm.empty: fppm = pd.DataFrame(50, index=list(range(0,len(sub.Sequence)*2)), columns=list(range(0,len(sub.Sequence)*2)))
        z = max(fppm.max())
    
    ## EXPERIMENTAL INTENSITIES MATRIX (TARGET) ##
    #frv2 = ions.INT
    
    ## Q-SCORE AND E-SCORE ##
    qscore, escore = qeScore(ppmfinal, ions.INT, err)
    #matched_ions = qscore.shape[0]
    
    if dograph or standalone:
        pepmass = tquery[tquery.SCANS == sub.FirstScan].iloc[0]
        specpar = "MZ=" + str(pepmass.MZ) + ", " + "Charge=" + str(int(pepmass.CHARGE)) + "+"
        
        BDAGmax, YDAGmax = asBY(deltaplot, sub)
        
        ## SURVEY SCAN INFORMATION ##
        # TODO: dta files required
        
        ## V-SCORE ##
        vscore = vScore(qscore, sub, proofb, proofy, assign)
    
    ## PLOTS ##
    if standalone:
        logging.info("\t\t\tPlotting...")
    if dograph:
        plotPpmMatrix(sub, fppm, dm, frags, zoom, ions, err, specpar, exp_spec,
                      proof, deltamplot, escore, vscore, BDAGmax, YDAGmax, min_dm,
                      outpath, massconfig, standalone)
    if standalone:
        logging.info("\t\t\tDone.")
        return
    elif dograph and not standalone:
        return
    elif not dograph and not standalone:
        return(escore)
    else:
        return

def main(args):
    '''
    Main function
    '''
    ## USER PARAMS TO ADD ##
    err = float(mass._sections['Parameters']['ppm_error'])
    min_dm = float(mass._sections['Parameters']['min_dm'])
    # try:
    #     arg_dm = float(args.deltamass)
    # except ValueError:
    #     sys.exit("Minimum deltamass (-d) must be a number!")
    # Set variables from input file
    logging.info("Reading input file")
    scan_info = pd.read_csv(args.infile, sep=",", float_precision='high', low_memory=False)
    exps = list(scan_info.Raw.unique())
    for exp in exps:
        logging.info("Experiment: " + str(exp))
        exp = str(exp).replace(".txt","").replace(".raw","").replace(".mgf","")
        sql = scan_info.loc[scan_info.Raw == exp]
        data_type = sql.type[0]
        pathdict = prepareWorkspace(exp, sql.mgfDir[0], sql.dtaDir[0], sql.outDir[0])
        mgf = os.path.join(pathdict["mgf"], exp + ".mgf")
        logging.info("\tReading mgf file")
        fr_ns = pd.read_csv(mgf, header=None)
        tquery = getTquery(fr_ns)
        tquery.to_csv(os.path.join(pathdict["out"], "tquery_"+ exp + ".csv"), index=False, sep=',', encoding='utf-8')
        for scan in list(sql.FirstScan.unique()):
            logging.info("\t\tScan: " + str(scan))
            subs = sql.loc[sql.FirstScan==scan]
            for index, sub in subs.iterrows():
                #logging.info(sub.Sequence)
                seq2 = sub.Sequence[::-1]
                doVseq(sub, tquery, fr_ns, min_dm, err, pathdict["out"], True, False, True)
                
            

if __name__ == '__main__':

    # multiprocessing.freeze_support()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='Vseq',
        epilog='''
        Example:
            python Vseq.py

        ''')
        
    defaultconfig = os.path.join(os.path.dirname(__file__), "config/Vseq.ini")
    
    parser.add_argument('-i',  '--infile', required=True, help='Input file')
    parser.add_argument('-c', '--config', default=defaultconfig, help='Path to custom config.ini file')
    parser.add_argument('-e', '--error', default=0, help='Maximum ppm error to consider')
    parser.add_argument('-d', '--deltamass', default=0, help='Minimum deltamass to consider')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()
    
    # parse config
    mass = configparser.ConfigParser(inline_comment_prefixes='#')
    mass.read(args.config)
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
    main(args)
    logging.info('end script')