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
    logging.info(str(len(msffiles)) + " .msf files found. Converting to .tsv...")
    
    allsurveys = []
    for i, j in enumerate(msffiles):
        logging.info(str(i) + " out of " + str(len(msffiles)) + "...")
        con = sqlite3.connect(os.path.join(args.dir, j))
        if int(args.sequest) == 0: # SEQUEST
            surveys_df = pd.read_sql_query("SELECT P.sequence,P.searchenginerank,PeptideScores.ScoreValue,S.FirstScan from SpectrumHeaders AS S, Peptides as P, PeptideScores WHERE S.SpectrumID=P.SpectrumID AND P.PeptideID = PeptideScores.PeptideID AND PeptideScores.ScoreID=9;", con)
        elif int(args.sequest) == 1: # SEQUEST-HT
            # surveys_df = pd.read_sql_query("SELECT P.sequence,P.searchenginerank,PeptideScores.ScoreValue,S.FirstScan from SpectrumHeaders AS S, Peptides as P, PeptideScores WHERE S.SpectrumID=P.SpectrumID AND P.PeptideID = PeptideScores.PeptideID AND PeptideScores.ScoreID=4;", con)
            surveys_df = pd.read_sql_query("SELECT P.matchedionscount,P.totalionscount,P.sequence,ProteinAnnotations.description,P.searchenginerank,PeptideScores.ScoreValue,S.FirstScan from SpectrumHeaders AS S, Peptides as P, PeptideScores, PeptidesProteins, ProteinAnnotations WHERE S.SpectrumID=P.SpectrumID AND P.PeptideID = PeptideScores.PeptideID AND P.PeptideID = PeptidesProteins.PeptideID AND PeptidesProteins.ProteinID = ProteinAnnotations.ProteinID AND PeptideScores.ScoreID=4;", con)
        outfile = os.path.join(args.dir, j[:-4] + ".tsv")
        surveys_df.to_csv(outfile, index=False, sep='\t', encoding='utf-8')
        surveys_df["FILE"] = str(j)
        allsurveys.append(surveys_df)
    allsurveys = pd.concat(allsurveys)
    outfile = os.path.join(args.dir, "ALL_MSFs.tsv")
    allsurveys.to_csv(outfile, index=False, sep='\t', encoding='utf-8')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VseqExplorer',
        epilog='''
        Example:
            python VseqExplorer.py

        ''')
    
    parser.add_argument('-d',  '--dir', required=True, help='Directory containing .MSF files')
    parser.add_argument('-s',  '--sequest', required=True, help='0 = Sequest, 1 = Sequest-HT')
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