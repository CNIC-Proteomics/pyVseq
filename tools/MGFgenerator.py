# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:55:08 2022

@author: alaguillog
"""

import argparse
import logging
import numpy as np
import pandas as pd
import sys

def getTheoMZ(AAs, charge, sequence, pos, nt, ct, mass):
    '''    
    Calculate theoretical MZ using the PSM sequence.
    '''
    m_proton = mass.getfloat('Masses', 'm_proton')
    m_hydrogen = mass.getfloat('Masses', 'm_hydrogen')
    m_oxygen = mass.getfloat('Masses', 'm_oxygen')
    total_aas = 2*m_hydrogen + m_oxygen
    total_aas += charge*m_proton
    for i, aa in enumerate(sequence):
        if aa.lower() in AAs:
            total_aas += float(AAs[aa.lower()])
    MH = total_aas - (charge-1)*m_proton
    #MZ = (total_aas + int(charge)*m_proton) / int(charge)
    if charge > 0:
        MZ = total_aas / int(charge)
        return MZ, MH
    else:
        return MH

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
    
    rsize = int(round(seq_len/4,0))
    regions = [i+1 for i in range(0,seq_len,rsize)]
    regions += [i+len(sequence) for i in range(0,seq_len,rsize)][::-1]
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
       
    return(frags)
    
def makeMGFentry(mzs, i, pepmass, charge):
    mgfentry = []
    mgfentry.append("BEGIN IONS")
    mgfentry.append("TITLE=") # TODO put case description here
    mgfentry.append("SCANS=" + str(i))
    mgfentry.append("RTINSECONDS=" + str(i))
    mgfentry.append("PEPMASS=" + str(pepmass))
    mgfentry.append("CHARGE=" + str(charge) +  "+")
    mgfentry = mgfentry + list(mzs.apply(lambda x: '\t'.join([str(x.MZ), str(x.INT)]), axis=1))
    mgfentry.append("END IONS")
    return(mgfentry)

def main(args):
    AAs = {"A":71.037114, "R":156.101111, "N":114.042927, "D":115.026943,
           "C":103.009185, "E":129.042593, "Q":128.058578, "G":57.021464,
           "H":137.058912, "I":113.084064, "L":113.084064, "K":128.094963,
           "M":131.040485, "F":147.068414, "P":97.052764, "S":87.032028,
           "T":101.047679, "U":150.953630, "W":186.079313, "Y":163.063329,
           "V":99.068414, "O":132.089878, "Z":129.042594}
    sequence = str(args.sequence)
    intensities = [1, 10, 100, 1000]
    ppmerrors = [0, 10, 40]
    # TODO: add random noise
    frags = makeFrags(len(sequence))
    frags.region = frags.region.apply(lambda x: , axis=1)
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
    parser.add_argument('-s',  '--sequence', required=True, help='Aminoacid sequence')
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