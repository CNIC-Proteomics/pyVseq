# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:55:08 2022

@author: alaguillog
"""

import argparse
import logging
import pandas as pd
import sys

def getTheoMZ(charge, sequence, mods, pos, nt, ct, mass):
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
    sequence = str(args.sequence)
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