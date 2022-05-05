# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:07:16 2022

@author: alaguillog
"""

import argparse
import sqlite3
import pandas as pd
import logging
import os
from pathlib import Path
import sys

def main(args):
    logging.info("Looking for .dta files...")
    msffiles = os.listdir(Path(args.dir))
    msffiles = [i for i in msffiles if i[-4:]=='.msf']
    logging.info(str(len(msffiles)) + " .msf files found.")
    
    for i, j in enumerate(msffiles):
        logging.info(str(i) + " out of " + str(len(msffiles)) + "...")
    con = sqlite3.connect(r"S:\LAB_JVC\RESULTADOS\AndreaLaguillo\pyVseq\EXPLORER\MGFS\IPGLLSPHPLLQLSYTATDR.msf")
    # cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # cursor.close()
    surveys_df = pd.read_sql_query("SELECT P.sequence,P.searchenginerank,PeptideScores.ScoreValue,S.FirstScan from SpectrumHeaders AS S, Peptides as P, PeptideScores WHERE S.SpectrumID=P.SpectrumID AND P.PeptideID = PeptideScores.PeptideID AND PeptideScores.ScoreID=9;", con)
    
if __name__ == '__main__':

    # multiprocessing.freeze_support()

    # parse arguments
    parser = argparse.ArgumentParser(
        description='VseqExplorer',
        epilog='''
        Example:
            python VseqExplorer.py

        ''')
    
    parser.add_argument('-d',  '--dir', required=True, help='Directory containing .MSF files')
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