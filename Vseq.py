# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:10:14 2022

@author: alaguillog
"""
# mgf = 'S:\\U_Proteomica\\UNIDAD\\DatosCrudos\\LopezOtin\\COVID\\COVID_TMT\\COVID_TMT1_F2.mgf'
# seq2 = kLETEVMq

# import modules
import os
import sys
import argparse
import configparser
import itertools
import logging
import pandas as pd
from pathlib import Path
import random
import re
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

def getTheoMH(charge, sequence, nt, ct):
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
    #♠total_aas += float(MODs['nt']) + float(MODs['ct'])
    if nt:
        total_aas += float(MODs['nt'])
    if ct:
        total_aas += float(MODs['ct'])
    for aa in sequence:
        if aa.lower() in AAs:
            total_aas += float(AAs[aa.lower()])
        if aa.lower() in MODs:
            total_aas += float(MODs[aa.lower()])
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
        fragy = getTheoMH(0,yn,nt,True) + dm
        outy[i:] = fragy
        
    ## B SERIES ##
    outb = pd.DataFrame(np.nan, index=list(range(1,len(seq)+1)), columns=list(range(1,len_ions+1)))
    for i in range(0,len(seq)):
        bn = list(seq[::-1][i:])
        if i > 0: ct = False
        else: ct = True
        fragb = getTheoMH(0,bn,True,ct) - 2*m_hydrogen - m_oxygen + dm
        outb[i:] = fragb
    
    ## FRAGMENT MATRIX ##
    yions = outy.T
    bions = outb.iloc[::-1].T
    spec = pd.concat([bions, yions], axis=1)
    spec.columns = range(spec.columns.size)
    spec.reset_index(inplace=True, drop=True)
    return(spec)

def errorMatrix(fr_ns, mz, theo_spec):
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

def assignIons(theo_spec, dm_theo_spec, frags, dm, arg_dm):
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
        for kj in list(range(0,deltamplot.shape[0])): #columns
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

def doVseq(sub, tquery, fr_ns, arg_dm):
    parental = getTheoMH(sub.Charge, sub.Sequence, True, True)
    mim = sub.ExpNeutralMass + mass.getfloat('Masses', 'm_proton')
    dm = mim - parental
    parentaldm = parental + dm
    dmdm = mim - parentaldm
    #query = tquery[(tquery["CHARGE"]==sub.Charge) & (tquery["SCANS"]==sub.FirstScan)]
    exp_spec, ions, spec_correction = expSpectrum(fr_ns, sub.FirstScan)
    theo_spec = theoSpectrum(sub.Sequence, len(ions), 0)
    terrors, terrors2, terrors3, texp = errorMatrix(fr_ns, ions.MZ, theo_spec)
    
    ## DM OPERATIONS ##
    dm_theo_spec = theoSpectrum(sub.Sequence, len(ions), dm)
    dmterrors, dmterrors2, dmterrors3, dmtexp = errorMatrix(fr_ns, ions.MZ, dm_theo_spec)
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
    assign = assignIons(theo_spec, dm_theo_spec, frags, dm, arg_dm)
    
    ## PPM ERRORS ##
    if sub.Charge == 2:
        ppmfinal = pd.DataFrame(np.array([terrors, terrors2]).min(0))
        parcial = ppmfinal
        if dm != 0: ppmfinal = pd.DataFrame(np.array([terrors, terrors2, dmterrors, dmterrors2]).min(0))
    elif sub.Charge == 3:
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
    ppmfinal[ppmfinal<=300] = 1
    ppmfinal[ppmfinal>300] = 0
    
    deltamplot, deltaplot = deltaPlot(parcialdm, parcial, ppmfinal)
    
    
    return    

def main(args):
    '''
    Main function
    '''
    try:
        arg_dm = float(args.deltamass)
    except ValueError:
        sys.exit("Minimum deltamass (-d) must be a number!")
    # Set variables from input file
    scan_info = pd.read_csv(args.infile, sep=",", float_precision='high', low_memory=False)
    exps = list(scan_info.Raw.unique())
    for exp in exps:
        exp = str(exp).replace(".txt","").replace(".raw","")
        sql = scan_info.loc[scan_info.Raw == exp]
        data_type = sql.type[0]
        pathdict = prepareWorkspace(exp, sql.mgfDir[0], sql.dtaDir[0], sql.outDir[0])
        mgf = os.path.join(pathdict["mgf"], exp + ".mgf")
        fr_ns = pd.read_csv(mgf, header=None)
        tquery = getTquery(fr_ns)
        tquery.to_csv(os.path.join(pathdict["out"], "tquery_"+ exp + ".csv"), index=False, sep=',', encoding='utf-8')
        for scan in list(sql.FirstScan.unique()):
            subs = sql.loc[sql.FirstScan==scan]
            logging.info("SCAN="+str(scan))
            for index, sub in subs.iterrows():
                logging.info(sub.Sequence)
                seq2 = sub.Sequence[::-1]
                doVseq(sub, tquery, fr_ns, arg_dm)
                
            

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
    parser.add_argument('-m', '--mass', default=defaultconfig, help='Path to custom massfile.ini file')
    parser.add_argument('-d', '--deltamass', default=0, help='Minimum deltamass to consider')

    parser.add_argument('-w',  '--n_workers', type=int, default=4, help='Number of threads/n_workers (default: %(default)s)')    
    parser.add_argument('-v', dest='verbose', action='store_true', help="Increase output verbosity")
    args = parser.parse_args()
    
    # parse config
    config = configparser.ConfigParser(inline_comment_prefixes='#')
    config.read(args.config)
    mass = configparser.ConfigParser(inline_comment_prefixes='#')
    mass.read(args.mass)
    if args.ppm is not None:
        config.set('PeakAssignator', 'ppm_max', str(args.ppm))
        config.set('Logging', 'create_ini', '1')
    # if something is changed, write a copy of ini
    if config.getint('Logging', 'create_ini') == 1:
        with open(os.path.dirname(args.infile) + '/Vseq.ini', 'w') as newconfig:
            config.write(newconfig)

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