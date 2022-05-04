#
# Import Modules
#
import argparse
import ast
import configparser
import itertools
import json
import logging
import multiprocessing
import numpy as np
import os
import pandas as pd
import requests
import re
import sys
import unicodedata



#
# Define Functions
#
def readFasta(seq_path):
    """
    Read Fasta
    """
    
    so = [] # Sequence Object: list of tuples (sequence, id)
    mh = []

    with open(seq_path, 'r') as f:

        h = f.readline().strip()
        s = ""
        for line in f:

            if '>' in line[0]:
                #so.append(Protein(h,s))
                #mh.append(np.cumsum([aa[i] for i in s]))
                so.append((h,s))
                h = line.strip()
                s = ""
            
            else:
                s += line.strip()
        

        #so.append(Protein(h,s))
        #mh = np.cumsum([aa[i] for i in s])
        #mh.append(np.cumsum([aa[i] for i in s]))
        so.append((h,s))
        return so#,mh


def readUPI(seq_path):
    """
    Read list of UniProt ID
    """

    with open(seq_path, 'r') as f:
        upi = []
        for line in f:
            upi.append(line.strip())

    return upi    


def requestFasta(seq_path):
    """
    Request Uniprot id
    """

    upi = readUPI(seq_path)

    bURL = [f"https://www.uniprot.org/uniprot/{i}.fasta" for i in upi]
    r = [requests.get(url) for url in bURL]
    so = [ri.text.split('\n') if ri.ok else [i, ''] for ri,i in zip(r, upi)]
    so = [(soi[0], ''.join(soi[1:])) for soi in so]
    #mh = [np.cumsum([aa[k] for k in j]) for i,j in so]

    return so#,mh


def readUnimod(unimod_path):
    '''
    Read Unimod filtered table
    '''

    full_unim = pd.read_csv(unimod_path, sep="\t")

    unim = full_unim.loc[:, ['Title', 'mono_mass', 'site']]
    unim['site'] = unim['site'].apply(ast.literal_eval)

    unim = unim.explode('site').drop_duplicates()
    unim = unim.groupby(['mono_mass', 'site']).agg(lambda x: ' // '.join(x)).reset_index()

    return full_unim, unim


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def trypticCut(so,aa, params):
    MISCLEAVAGES = int(params['miscleavages'])+max(1, int(params['modnumber'])) # Consider modifications in K that avoid tryptic cut
    MINSIZE = int(params['minsize'])
    
    # Generate list with mh of each sequence
    mh = (np.cumsum([aa[k] for k in j]).tolist() for i,j in so)

    # Tryptic digestion (PARALLELIZE AT PROTEIN LEVEL)
    so = [(i,re.sub(r'([KR])(?!P)', r'\1&', j).split('&')) for i,j in so]

    # Final position of each peptide (0-based)
    so = [(i, j, (np.cumsum([len(k) for k in j])-1).tolist()) for i,j in so]

    # mh of each peptide
    so = [(*i[:], [j[l]-j[l-len(m)] if n!=0 else j[l] for n,m,l in zip(range(len(i[1])), i[1], i[2])]) for i,j in zip(so, mh)]

    # miscleavage
    so = (
        (
            i, # Sequence ID
            #j, # Protein Sequence
            [
                (m1, m2, m3, 0) for m1,m2,m3 in zip(j,k,l) # Fully digested sequences (sequence, last position, mh, misscleavage)
            ] + [
                (''.join(j[m:m+n+1]), k[m+n], sum(l[m:m+n+1]), n) for n in np.arange(1, MISCLEAVAGES+1) for m in np.arange(0, len(j)-n) # Partially digested sequences
            ]
        ) 
        for i,j,k,l in so
    )

    # Filter by minimum number of aa
    so = ((i, *list(zip(*[list(l) for l in k if len(l[0]) > MINSIZE]))[:]) for i,k in so)
    so = [i for i in so if len(i)>1]
    so = list(zip(*[j for i in [zip(*[tuple(itertools.repeat(i[0], len(i[1]))), *i[1:]]) for i in so] for j in i]))

    return so


def modifications(p2mod, upi2p, unim, Kx):

    p2mod['site'] = p2mod['p'].apply(set)

    # Add non mod
    p2mod['site'] = [['-']+list(i) for i in p2mod['site'].tolist()]

    # Expand by aa
    p2mod = p2mod.explode('site',ignore_index=True)

    # Find sites of each aa
    p2mod['n'] = [[k.start() for k in re.finditer(i,j)] for i,j in zip(p2mod['site'].tolist(), p2mod['p'].tolist()) ]

    # Merge with unim and calculate mh
    p2mod = pd.merge(
        p2mod,
        unim,
        how='inner',
        on='site'
    )

    p2mod['mh'] = p2mod['mres'] + p2mod['mono_mass'] + 18.01056 + 1.007825
    p2mod = p2mod.explode('n', ignore_index=True)

    # Remove pdm with Kx (Acetyl or GG) in the last position
    p2mod = p2mod.loc[
        ~np.logical_and(
            np.array([len(i)-1 for i in p2mod['p'].to_list()]) == p2mod['n'].to_numpy(),
            np.array([i in Kx for i in p2mod['Title'].to_list()])
        ),
        :
    ]

    # Build pdm
    pdm = list(zip(p2mod['p'].tolist(), p2mod['n'].tolist(), p2mod['mono_mass'].tolist()))
    p2mod['pdm'] = [i if pd.isna(j) else f"{i[:j+1]}[{round(k,5)}]{i[j+1:]}" for i,j,k in pdm]

    # Add protein
    p2mod = pd.merge(
        p2mod,
        upi2p,
        how='left',
        on='p'
    )

    return p2mod



def combinations(mmod, upi2p, MODNUMBER):
    
    # Combine each modification with peptide and mres
    mmod = [(i[0], [list(itertools.product([j[:2]],list(zip(*j[2:])))) for j in i[1]]) for i in mmod]

    # Save peptide and mres
    p = [i[0] for i in mmod]

    # Combine all modifications
    mmod = [[j, list(itertools.product(*[i[1][l] for l in k]))] for n in range(2, MODNUMBER+1) for i,j in zip(mmod, p) for k in itertools.combinations(range(len(i[1])), n)]
    
    if len(mmod) == 0:
        return pd.DataFrame(columns=['p','mres', 'site', 'n', 'Title', 'mono_mass', 'mh', 'pdm', 'q'])

    # Rearrange to build dataframe
    mmod = [(i[0], [[list(zip(*k)) for k in list(zip(*j))] for j in i[1]]) for i in mmod]
    
    mmod = [(tuple(i[0]), *[l for k in j for l in k]) for i in mmod for j in i[1]]
    
    mmod = [i for i in list(zip(*mmod))]
    
    mmod = [*list(zip(*mmod[0])), *mmod[1:]]
    
    mmod = pd.DataFrame({
        'p': mmod[0],
        'mres': mmod[1],
        'site': mmod[2],
        'n': mmod[3],
        'Title': mmod[4],
        'mono_mass': mmod[5]
    })

    # Calculate mh    
    mmod['mh'] = mmod.loc[:, 'mres'] + mmod.loc[:, 'mono_mass'].apply(sum) + 18.01056 + 1.007825
    
    # Build pdm
    pdm = list(zip(mmod['p'].tolist(), mmod['n'].tolist(), mmod['mono_mass'].tolist()))
    
    mmod['pdm'] = [
        ''.join([
            f"{i[:l[1]+1]}[{round(l[2],5)}]" if l[0]==0 else 
            f"{i[j[l[0]-1]+1:l[1]+1]}[{round(l[2],5)}]" if l[0] < len(j)-1 else
            f"{i[j[l[0]-1]+1:l[1]+1]}[{round(l[2],5)}]{i[l[1]+1:]}"
            for l in zip(range(len(j)), j,k)
        ]) 

        for i,j,k in pdm
    ]
    
    mmod = pd.merge(
        mmod,
        upi2p,
        how='left',
        on='p'
    )

    return mmod


def recalcMissCleavages(p2mod, Kx):

    # p2mod['misscleavages'] = p2mod['misscleavages'] - np.array([
    #     np.sum([
    #         True if k in ['Acetyl', 'GG'] and l=='K' else False
    #         for k,l in zip(i,j)
    #     ])
    #     for i,j in zip(
    #         p2mod['Title'].to_list(),
    #         p2mod['site'].to_list()
    #     )
    # ])

    df = pd.DataFrame(
        [
            (n,k,l) 
            
            for n,i,j in zip(
                p2mod['pdm'].to_list(),
                p2mod['Title'].to_list(),
                p2mod['site'].to_list()
            ) 
            
            for k,l in zip(i,j)
        ], columns=['pdm', 'Title', 'site'])
    
    df['kx'] = np.logical_and(
        df['site'] == 'K',
        np.logical_or(*[df['Title']==i for i in Kx])
    )

    df = df.loc[:, ['pdm', 'kx']].groupby('pdm').agg(sum).reset_index()

    df = pd.merge(
        df,
        p2mod.loc[:, ['pdm', 'misscleavages']],
        how='inner',
        on='pdm'
    )

    df['mc'] = df['misscleavages'] - df['kx']

    p2mod = pd.merge(
        p2mod,
        df.loc[:, ['pdm', 'mc']],
        how='inner',
        on='pdm'
    )

    p2mod = p2mod.rename(columns={'misscleavages': 'iKR', 'mc': 'misscleavages'})

    return p2mod


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '-', value)
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def writedf(df, OUTFILE_PATH):
    r = re.search(r'GN=([^\s]+)', df[0])
    filename = r.groups()[0] if r and len(r.groups())==1 else slugify(df[0])
    df[1].to_csv(os.path.join(OUTFILE_PATH, filename+'.tsv'), sep='\t', index=False)

#
# Main
#
def main(config):
    '''
    '''
    logging.info('Reading parameters')
    params = {i[0]:i[1] for i in config.items(config.sections()[0])}
    logging.info(json.dumps(params, indent=4)[1:-2])
    
    # Get Kx as list --> Title of modifications that avoid tryptic cut
    Kx = [i.strip() for i in params['kx'].split(',') if i != '']


    # Dictionary aa --> mh of residue    
    aa = {i[0].upper():float(i[1]) for i in config.items(config.sections()[1])}

    # Read Fasta
    if params['fasta_mode'].upper() == 'TRUE':
        logging.info(f'Reading FASTA: {params["seq_path"]}')
        so = readFasta(params['seq_path'])

    else:
        logging.info('Sending requests to UniProt')
        so = requestFasta(params['seq_path'])

    logging.info(f'Protein sequences were read: {len(so)} proteins')

    # Read Unimod
    logging.info('Reading UniMod table')
    full_unim, unim = readUnimod(params['unimod_path'])

    # Filter unim
    aafilter = [i.strip() for i in params['aa'].split(',') if i != '']
    modfilter = [i.strip().upper() for i in params['mod'].split(',') if i != '']

    if len(aafilter) > 0:
        unim = unim.loc[np.isin(unim['site'], aafilter), :]

    if len(modfilter) > 0:
        unim = unim.loc[[i.upper() in modfilter for i in unim['Title'].to_list()], :]

    # Parallelize at protein level
    logging.info(f'Tryiptic digestion')
    logging.info(f'Miscleavages: {params["miscleavages"]}')
    logging.info(f'Minimum aa number: {params["minsize"]}')

    #
    # TRYPTIC CUT
    #

    #so = trypticCut(so, aa, params) #Non parallel execution 
    pool = multiprocessing.Pool(int(params['nproc']))
  
    so = list(chunks(so, 5))
    so = pool.starmap(trypticCut, zip(so, itertools.repeat(aa), itertools.repeat(params)))
    so = [[k for j in i for k in j] for i in zip(*so)]
    
    pool.close()
    pool.join()
    
    if len(so) == 0:
        logging.error('No tryptic peptide found')
        sys.exit()

    logging.info(f'Tryptic digestion finished: {len(so[1])} peptides')


    #
    # ALL 1 MODIFICATION
    #

    # Add custom modifications to unim
    unim = pd.concat([
        unim,
        pd.DataFrame({
            'Title': ['NM'],
            'mono_mass': [0],
            'site': ['-']
        })  
    ], axis=0).drop_duplicates()

    upi2p = pd.DataFrame({
        'q': so[0],
        'p': so[1]
    })

    p2mod = pd.DataFrame({
        'p': so[1],
        'mres': so[3]
    }).drop_duplicates()


    #p2mod = modifications(p2mod, upi2p, unim) # Without parallel
    logging.info('Calculating pdm with one modification')
    pool = multiprocessing.Pool(int(params['nproc']))
    
    p2mod = np.array_split(p2mod,10)
    p2mod = pool.starmap(modifications, zip(p2mod, itertools.repeat(upi2p), itertools.repeat(unim), itertools.repeat(Kx)))
    p2mod = pd.concat(p2mod)

    
    pool.close()
    pool.join()
    logging.info(f'pdm with one modification calculated: {p2mod.shape[0]} pdm')


    #
    # ALL >1 MODIFICATION
    #
    if int(params['modnumber']) > 1:
        logging.info('Calculate pdm with more than one modification')
        mmod = p2mod.loc[p2mod['site']!='-', ['p','mres','site','n','Title','mono_mass']].groupby(['p','mres', 'site', 'n']).aggregate(list).reset_index().to_dict('list')
        mmod = list(zip(*[mmod[i] for i in mmod.keys()]))
        mmod.sort(key=lambda x: x[0])
        mmod = [([*i[:2]],[*i[2:]]) for i in mmod]

        # PARALLEL HERE BY PEPTIDE
        mmod = [(i, sorted([list(*k[1:]) for k in j], key=lambda x: x[1])) for i,j in itertools.groupby(mmod, lambda x:x[0])]

        # mmod = combinations(mmod, upi2p, int(params['modnumber']))
        pool = multiprocessing.Pool(int(params['nproc']))
        mmod = chunks(mmod, 10)
        mmod = pool.starmap(combinations, zip(mmod, itertools.repeat(upi2p), itertools.repeat(int(params['modnumber']))))
        mmod = pd.concat(mmod)
        pool.close()
        pool.join()
        
        mmod['nmod'] = mmod['n'].apply(lambda x: len(x))

        logging.info(f'pdm with more than one modification calculated: {mmod.shape[0]} pdm (redundant)')

    
    # Adapt table
    p2mod['nmod'] = p2mod['site'].apply(lambda x: 0 if x=='-' else 1)
    p2mod['site'] = p2mod.loc[:, 'site'].apply(lambda x: (x, ))
    p2mod['n'] = p2mod.loc[:, 'n'].apply(lambda x: (x, ))
    p2mod['Title'] = p2mod.loc[:, 'Title'].apply(lambda x: (x, ))
    p2mod['mono_mass'] = p2mod.loc[:, 'mono_mass'].apply(lambda x: (x, ))

    if int(params['modnumber']) > 1:
        p2mod = pd.concat([
            p2mod,
            mmod
        ])


    #
    # TMT
    #
    if params['tmt'].upper() == 'TRUE':
        logging.info('Adding TMT')
        tmtmass = 229.162932

        # Add TMT to K      
        tmt = [re.sub(r'K(?!\[)', f'K[{tmtmass}]', i) for i in p2mod['pdm'].tolist()]
        
        # add tmt to N-terminal (first residue) when it is modified
        tmt = [re.sub(r'(?<=^\w)\[([\-0-9.]+)\]', lambda m: f'[{round(float(m.group(1))+tmtmass,5)}]', i) for i in tmt]
        
        # add tmt to first residue (N-terminal) when it is not modified
        tmt = [re.sub(r'(?<=^\w)(?!\[)', f'[{tmtmass}]', i) for i in tmt]

        p2mod['pdm'] = tmt
        p2mod['mh'] = p2mod['mh'] + tmtmass
        p2mod['mh'] = p2mod['mh'] + [tmtmass*(i.count('K')-j.count('K')) for i,j in zip(p2mod['p'].tolist(), p2mod['site'].tolist())]


    #
    # CALCULATE CHARGE AND MC NUMBER
    #

    logging.info('Calculating charge')

    ch = pd.DataFrame(        
        [
            (
                i,
                1+i.count('H')+i.count('R')+i.count('K'),
                len(re.findall(r'[KR](?!P)(?!$)', i))
            ) 
            for i in list(set(p2mod['p'].tolist()))
        ],
        columns=['p', 'charge', 'misscleavages']
    )

    p2mod = pd.merge(
        p2mod,
        ch,
        how='left',
        on='p'
    )


    # Remove from missing cleavages modifications in K 
    logging.info(f'Recalculating missing cleavages: {", ".join(Kx)}')
    pool = multiprocessing.Pool(int(params['nproc']))
    
    p2mod = np.array_split(p2mod,10)
    p2mod = pool.starmap(recalcMissCleavages, zip(p2mod, itertools.repeat(Kx)))
    p2mod = pd.concat(p2mod)

    pool.close()
    pool.join()

    p2mod = p2mod.loc[p2mod['misscleavages']<=int(params['miscleavages']), :]

    logging.info(f'Total pdm: {p2mod.shape[0]}')    


    #
    # WRITE OUTPUT
    #

    # combine equal pdm with different modification Title
    # p2mod = p2mod.groupby(['p', 'mres', 'site', 'n', 'mono_mass', 'mh', 'pdm', 'q', 'nmod', 'charge', 'misscleavages']).agg(list).reset_index()

    logging.info('Writing output files')
    p2mod = list(
        p2mod.rename(
            columns={
                'mh': 'MH',
                'pdm':'Sequence',
                'Title': 'DeltaMassLabel',
                'charge':'Charge',
                'misscleavages': 'MissCleavages'
            }
        ).groupby('q')
    )

    pool = multiprocessing.Pool(int(params['nproc']))
    _ = pool.starmap(writedf, zip(p2mod, itertools.repeat(params['outfile_path'])))
    pool.close()
    pool.join()


if __name__ == '__main__':

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=f'{os.path.abspath(os.path.dirname(__file__))}/VDigestor.ini' , help='Path to custom VDigestor.ini file')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='VDIGESTOR'+' - '+str(os.getpid())+' - %(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    config = configparser.ConfigParser()
    config.read(args.config)

    logging.info('Start script: '+"{0}".format(" ".join([x for x in sys.argv])))
    main(config)
    logging.info(f'End script') 
