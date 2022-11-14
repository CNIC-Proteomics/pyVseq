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
    logging.info("Looking for .msf files...")
    msffiles = os.listdir(Path(args.dir))
    msffiles = [i for i in msffiles if i[-4:]=='.msf']
    logging.info(str(len(msffiles)) + " .msf files found. Converting to .tsv...")
    
    allsurveys = []
    for i, j in enumerate(msffiles):
        logging.info(str(i) + " out of " + str(len(msffiles)) + "...")
        con = sqlite3.connect(os.path.join(args.dir, j))
        # Get tables
        # cursor = con.cursor()
        # cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        # Get columns
        # cursor.execute("SELECT * FROM Peptides_decoy")
        # print(cursor.fetchall())
        if int(args.sequest) == 0: # SEQUEST
            surveys_df = pd.read_sql_query("SELECT P.sequence,P.searchenginerank,PeptideScores.ScoreValue,S.FirstScan from SpectrumHeaders AS S, Peptides as P, PeptideScores WHERE S.SpectrumID=P.SpectrumID AND P.PeptideID = PeptideScores.PeptideID AND PeptideScores.ScoreID=9;", con)
        elif int(args.sequest) == 1: # SEQUEST-HT
            # surveys_df = pd.read_sql_query("SELECT P.sequence,P.searchenginerank,PeptideScores.ScoreValue,S.FirstScan from SpectrumHeaders AS S, Peptides as P, PeptideScores WHERE S.SpectrumID=P.SpectrumID AND P.PeptideID = PeptideScores.PeptideID AND PeptideScores.ScoreID=4;", con)
            if int(args.pd) == 0:
                surveys_df = pd.read_sql_query("SELECT P.matchedionscount,P.totalionscount,P.sequence,ProteinAnnotations.description,P.searchenginerank,PeptideScores.ScoreValue,S.FirstScan from SpectrumHeaders AS S, Peptides as P, PeptideScores, PeptidesProteins, ProteinAnnotations WHERE S.SpectrumID=P.SpectrumID AND P.PeptideID = PeptideScores.PeptideID AND P.PeptideID = PeptidesProteins.PeptideID AND PeptidesProteins.ProteinID = ProteinAnnotations.ProteinID AND PeptideScores.ScoreID=4;", con)
            elif int(args.pd) == 1:
                targets_df = pd.read_sql_query("SELECT P.matchedionscount,P.totalionscount,P.sequence,P.searchenginerank,P.xcorr,P.FirstScan from TargetPsms as P", con)
                targets_df["TYPE"] = "Target"
                con = sqlite3.connect(os.path.join(args.dir, j))
                decoys_df = pd.read_sql_query("SELECT P.matchedionscount,P.totalionscount,P.sequence,P.searchenginerank,P.xcorr,P.FirstScan from DecoyPsms as P", con)
                decoys_df["TYPE"] = "Decoy"
                surveys_df = pd.concat([targets_df, decoys_df])
            elif int(args.pd) == 2:
                surveys_df = pd.read_sql_query("SELECT P.matchedionscount,P.totalionscount,P.sequence,ProteinAnnotations.description,P.searchenginerank,PeptideScores.ScoreValue,S.FirstScan from SpectrumHeaders AS S, Peptides as P, PeptideScores, PeptidesProteins, ProteinAnnotations WHERE S.SpectrumID=P.SpectrumID AND P.PeptideID = PeptideScores.PeptideID AND P.PeptideID = PeptidesProteins.PeptideID AND PeptidesProteins.ProteinID = ProteinAnnotations.ProteinID AND PeptideScores.ScoreID=9;", con)
                surveys_df["Label"] = surveys_df.apply(lambda x: 'Decoy' if x['Description'][:6]==">DECOY" else "Target", axis = 1)
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
    parser.add_argument('-p',  '--pd', required=True, help='0 = PD 2.4 or under, 1 = PD 2.5')
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