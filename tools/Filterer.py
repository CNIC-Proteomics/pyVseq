# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:24:20 2022

@author: alaguillog
"""

import argparse
import logging
import pandas as pd
import sys

### MOUSE ###

def main(args):
    fcolumn = str(args.column)
    keep = int(args.keep)
    threshold = float(args.threshold)
    df = pd.read_csv(args.infile, sep="\t", float_precision='high', low_memory=False)
    
    if keep:
        df = df.loc[df[fcolumn] <= threshold]
    else:
        df = df.loc[df[fcolumn] >= threshold]      
    
    df.to_csv(args.infile[:-4] + '_Filtered.txt', sep="\t")
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
    parser.add_argument('-i',  '--infile', required=True, help='VseqExplorer table to be filtered')
    parser.add_argument('-c',  '--column', required=True, help='Column to filter by')
    parser.add_argument('-t',  '--threshold', required=True, help='Threshold value')
    parser.add_argument('-k',  '--keep', default=0, help='Keep values higher (0) or lower (1) than threshold')
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