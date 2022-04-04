# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:42:30 2022

@author: alaguillog
"""

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
import numpy as np
import pandas as pd
from pathlib import Path
import random
import re
import seaborn as sns
import scipy.stats
import statistics
pd.options.mode.chained_assignment = None  # default='warn'

from Vseq import doVseq
# infile= r"C:\Users\alaguillog\GitHub\eca\Sadek_fr8.mgf"
# table = 'C:\\Users\\alaguillog\\GitHub\\pyVseq\\vseqexplorer_input_data.csv'
# config = 'C:\\Users\\alaguillog\\GitHub\\CNICpyVseq\\Vseq.ini'

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
    rquery = fr_ns.loc[fr_ns[0].str.contains("RTINSECONDS=")]
    rquery = rquery[0].str.replace("RTINSECONDS=","")
    rquery.reset_index(inplace=True, drop=True)
    tquery = pd.concat([squery.rename('SCANS'),
                        mquery.rename('PEPMASS'),
                        cquery.rename('CHARGE'),
                        rquery.rename('RT')],
                       axis=1)
    tquery[['MZ','INT']] = tquery.PEPMASS.str.split(" ",expand=True,)
    tquery['CHARGE'] = tquery.CHARGE.str[:-1]
    tquery = tquery.drop("PEPMASS", axis=1)
    tquery = tquery.apply(pd.to_numeric)
    return tquery

def getTheoMZH(charge, sequence, nt, ct):
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
    for aa in sequence:
        if aa.lower() in AAs:
            total_aas += float(AAs[aa.lower()])
        if aa.lower() in MODs:
            total_aas += float(MODs[aa.lower()])
        if aa.islower():
            total_aas += float(MODs['isolab'])
    MH = total_aas - (charge-1)*m_proton
    #MZ = (total_aas + int(charge)*m_proton) / int(charge)
    if charge > 0:
        MZ = total_aas / int(charge)
        return MZ, MH
    else:
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

def theoSpectrum(seq, len_ions, dm):
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
        fragy = getTheoMZH(0,yn,nt,True) + dm
        outy[i:] = fragy
        
    ## B SERIES ##
    outb = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        bn = list(seq[::-1][i:])
        if i > 0: ct = False
        else: ct = True
        fragb = getTheoMZH(0,bn,True,ct) - 2*m_hydrogen - m_oxygen + dm
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
    ppmfinal["minv"] = ppmfinal.apply(lambda x: x.min() , axis = 1)
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

def getIons(x, tquery, mgf, index2, min_dm, min_match, ftol, outpath, standalone, massconfig, dograph):
    ions_exp = []
    b_ions = []
    y_ions = []
    escore, ppmfinal, frags = doVseq(x, tquery, mgf, index2, min_dm, min_match, ftol, outpath, standalone, massconfig, dograph)
    ppmfinal = ppmfinal.drop("minv", axis=1)
    ppmfinal.columns = frags.by
    ppmfinal[ppmfinal>ftol] = 0
    ppmfinal = ppmfinal.astype('bool').T
    ppmfinal = ppmfinal[(ppmfinal == True).any(axis=1)]
    if ppmfinal.any().any():
        b_ions = b_ions + [x for x in list(ppmfinal.index.values) if "b" in x]
        y_ions = y_ions + [x for x in list(ppmfinal.index.values) if "y" in x]
    ions_matched = len(b_ions) + len(y_ions)
    # spec, ions, spec_correction = expSpectrum(mgf, x.SCANS)
    # wdm_theo_spec = theoSpectrum(x.Sequence, len(ions), dm)
    # ions_exp = len(ions)
    # ions_matched = []
    # b_ions = []
    # y_ions = []
    # for frag in ions.MZ:
    #     terrors, terrors2, terrors3, texp = errorMatrix(ions.MZ, wdm_theo_spec)
    #     #terrors = (((frag - dm_theo_spec)/dm_theo_spec)*1000000).abs()
    #     terrors[terrors>ftol] = 0
    #     terrors = terrors.astype('bool')
    #     terrors2[terrors2>ftol] = 0
    #     terrors2 = terrors2.astype('bool')
    #     terrors3[terrors3>ftol] = 0
    #     terrors3 = terrors3.astype('bool')
    #     terrors, terrors2, terrors3 = terrors.T, terrors2.T, terrors3.T
    #     terrors.index, terrors2.index, terrors3.index = frags, frags, frags
    #     #tempbool = dm_theo_spec.between(frag-ftol, frag+ftol)
    #     if terrors.any().any():
    #         ions_matched.append(frag)
    #         b_ions = b_ions + [x for x in list(terrors[terrors==True].index.values) if "b" in x]
    #         y_ions = y_ions + [x for x in list(terrors[terrors==True].index.values) if "y" in x]
    #         b_ions = b_ions + [x+"++" for x in list(terrors2[terrors2==True].index.values) if "b" in x]
    #         y_ions = y_ions + [x+"++" for x in list(terrors2[terrors2==True].index.values) if "y" in x]
    #         b_ions = b_ions + [x+"+++" for x in list(terrors3[terrors3==True].index.values) if "b" in x]
    #         y_ions = y_ions + [x+"+++" for x in list(terrors3[terrors3==True].index.values) if "y" in x]
    return([ions_matched, ions_exp, b_ions, y_ions, escore])

def plotRT(subtquery, outpath, charge, startRT, endRT):
    outgraph = str(subtquery.Raw.loc[0]) + "_" + str(subtquery.Sequence.loc[0]) + "_M" + str(subtquery.ExpNeutralMass.loc[0]) + "_ch" + str(charge) + "_RT_plots.pdf"
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
    fig.suptitle(str(subtquery.Sequence.loc[0]) + '+' + str(round(subtquery.DeltaMass.loc[0],6)), fontsize=30)
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
    return

def main(args):
    '''
    Main function
    '''
    ## PARAMETERS ##
    ptol = float(mass._sections['Parameters']['precursor_tolerance'])
    ftol = float(mass._sections['Parameters']['fragment_tolerance'])
    bestn = int(mass._sections['Parameters']['best_n'])
    min_dm = float(mass._sections['Parameters']['min_dm'])
    min_match = int(mass._sections['Parameters']['min_ions_matched'])
    if args.outpath:
        outpath = args.outpath
    else:
        outpath = os.path.join(os.path.dirname(Path(args.infile)),"Vseq_Results")
    if not os.path.exists(Path(outpath)):
        os.mkdir(Path(outpath))
    ## INPUT ##
    logging.info("Reading input table")
    seqtable = pd.read_csv(args.table, sep=",", float_precision='high', low_memory=False)
    logging.info("Reading input file")
    mgf = pd.read_csv(args.infile, header=None)
    index2 = mgf.to_numpy() == 'END IONS'
    tquery = getTquery(mgf)
    ## COMPARE EACH SEQUENCE ##
    exploredseqs = []
    for index, query in seqtable.iterrows():
        logging.info("\tExploring sequence " + str(query.Sequence) + ", "
                     + str(query.ExpNeutralMass) + " Th, Charge "
                     + str(query.Charge))
        ## MZ and MH ##
        query['MZ'] = getTheoMZH(query.Charge, query.Sequence, True, True)[0]
        query['MH'] = getTheoMZH(query.Charge, query.Sequence, True, True)[1]
        ## DM ##
        mim = query.ExpNeutralMass + mass.getfloat('Masses', 'm_proton')
        dm = mim - query.MH
        dm_theo_spec = theoSpectrum(query.Sequence, len(query.Sequence), dm).loc[0]
        frags = ["b" + str(i) for i in list(range(1,len(query.Sequence)+1))] + ["y" + str(i) for i in list(range(1,len(query.Sequence)+1))[::-1]]
        dm_theo_spec.index = frags
        ## TOLERANCE ##
        upper = query.MZ + ptol
        lower = query.MZ - ptol
        ## OPERATIONS ##
        # subtquery = tquery[(tquery.CHARGE==query.Charge) & (tquery.MZ>=lower) & (tquery.MZ<=upper)]
        subtquery = tquery[(tquery.MZ>=lower) & (tquery.MZ<=upper)]
        logging.info("\t" + str(subtquery.shape[0]) + " scans found within ±"
                     + str(ptol) + " Th")
        if subtquery.shape[0] == 0:
            continue
        logging.info("\tComparing...")
        subtquery['Sequence'] = query.Sequence
        subtquery['ExpNeutralMass'] = query.ExpNeutralMass
        subtquery['DeltaMass'] = dm
        subtquery.rename(columns={'SCANS': 'FirstScan', 'CHARGE': 'Charge', 'RT':'RetentionTime'}, inplace=True)
        subtquery['templist'] = subtquery.apply(lambda x: getIons(x,
                                                                 tquery,
                                                                 mgf,
                                                                 index2,
                                                                 min_dm,
                                                                 min_match,
                                                                 ftol,
                                                                 Path(outpath),
                                                                 False,
                                                                 mass,
                                                                 False)
        subtquery["RawCharge"] = subtquery.Charge
        subtquery.Charge = query.Charge
                                                #if x.b_series and x.y_series else 0
                                                , axis = 1)
        subtquery['ions_matched'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 0]. tolist()
        #subtquery['ions_exp'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 1]. tolist()
        subtquery['ions_total'] = len(query.Sequence) * 2
        subtquery['b_series'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 2]. tolist()
        subtquery['y_series'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 3]. tolist()
        subtquery['Raw'] = os.path.split(Path(args.infile))[1][:-4]
        subtquery['e_score'] = pd.DataFrame(subtquery.templist.tolist()).iloc[:, 4]. tolist()
        subtquery = subtquery.drop('templist', axis = 1)
        # subtquery['e_score'] = subtquery.apply(lambda x: doVseq(x,
        #                                                         tquery,
        #                                                         mgf,
        #                                                         min_dm,
        #                                                         min_match,
        #                                                         err,
        #                                                         Path(outpath),
        #                                                         False,
        #                                                         mass,
        #                                                         False)
        #                                        #if x.b_series and x.y_series else 0
        #                                        , axis = 1)
        ## SORT BY ions_matched ##
        subtquery.sort_values(by=['INT'], inplace=True, ascending=False)
        subtquery.sort_values(by=['ions_matched'], inplace=True, ascending=False)
        subtquery.reset_index(drop=True, inplace=True)
        f_subtquery = subtquery.iloc[0:bestn]
        if f_subtquery.shape[0] > 0:
            logging.info("\tRunning Vseq on " + str(bestn) + " best candidates...")
            f_subtquery.apply(lambda x: doVseq(x,
                                               tquery,
                                               mgf,
                                               index2,
                                               min_dm,
                                               min_match,
                                               ftol,
                                               Path(outpath),
                                               False,
                                               mass,
                                               True) if x.b_series and x.y_series else logging.info("\t\tSkipping one candidate with empty fragmentation series..."), axis = 1)
        ## PLOT RT vs E-SCORE and MATCHED IONS ##
        plotRT(subtquery, outpath, query.Charge, tquery.iloc[0].RT/60, tquery.iloc[-1].RT/60)
        exploredseqs.append(subtquery)
        
    logging.info("Writing output table")
    # outfile = os.path.join(os.path.split(Path(args.table))[0],
    #                        os.path.split(Path(args.table))[1][:-4] + "_EXPLORER.csv")
    outfile = os.path.join(outpath, str(os.path.split(Path(args.infile))[1][:-4]) + "_EXPLORER.tsv")
    bigtable = pd.concat(exploredseqs, ignore_index=True, sort=False)
    bigtable = bigtable[bigtable.Charge != 0]
    bigtable.to_csv(outfile, index=False, sep='\t', encoding='utf-8')
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
        
    defaultconfig = os.path.join(os.path.dirname(__file__), "config/VseqExplorer.ini")
    
    parser.add_argument('-i',  '--infile', required=True, help='MGF file')
    parser.add_argument('-t',  '--table', required=True, help='Table of sequences to compare')
    parser.add_argument('-c', '--config', default=defaultconfig, help='Path to custom config.ini file')
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    parser.add_argument('-o', '--outpath', default=False, help='Path to save results')
    args = parser.parse_args()
    
    # parse config
    mass = configparser.ConfigParser(inline_comment_prefixes='#')
    mass.read(args.config)
    # if something is changed, write a copy of ini
    if mass.getint('Logging', 'create_ini') == 1:
        with open(os.path.dirname(args.infile) + '/Vseq.ini', 'w') as newconfig:
            mass.write(newconfig)

    # logging debug level. By default, info level
    log_file = outfile = args.infile[:-4] + 'VseqExplorer_log.txt'
    log_file_debug = outfile = args.infile[:-4] + 'VseqExplorer_log_debug.txt'
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
