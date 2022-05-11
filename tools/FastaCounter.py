from Bio import SeqIO
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def getTheoMZ(AAs, charge, sequence, series):
    '''    
    Calculate theoretical MZ using the PSM sequence.
    '''
    m_proton = 1.007276
    m_hydrogen = 1.007825
    m_oxygen = 15.994915
    total_aas = charge*m_proton
    if series == "y":
        total_aas += 2*m_hydrogen + m_oxygen
    for i, aa in enumerate(sequence):
        if aa.upper() in AAs:
            total_aas += float(AAs[aa.upper()])
    MH = total_aas - (charge-1)*m_proton
    #MZ = (total_aas + int(charge)*m_proton) / int(charge)
    if charge > 0:
        MZ = total_aas / int(charge)
        return MZ, MH
    else:
        return MH

def makeFrags(seq):
    '''
    Name all fragments.
    '''
    pd.options.mode.chained_assignment = None  # default='warn'
    frags = pd.DataFrame(np.nan, index=list(range(0,len(seq)*2)),
                         columns=["series", "by"])
    frags.series = ["b" for i in list(range(1,len(seq)+1))] + ["y" for i in list(range(1,len(seq)+1))[::-1]]
    frags.by = ["b" + str(i) for i in list(range(1,len(seq)+1))] + ["y" + str(i) for i in list(range(1,len(seq)+1))[::-1]]
    ## REGIONS ##
    rsize = int(round(len(seq)/4,0))
    regions = [i+1 for i in range(0,len(seq),rsize)]
    if len(seq) % 2 == 0:
        regions += [i+len(seq)+1 for i in range(0,len(seq),rsize)][::-1]
    else:
        regions += [i+len(seq) for i in range(0,len(seq),rsize)][::-1]
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
      ## SEQUENCES ##
    frags["seq"] = None
    for index, row in frags.iterrows():
        series = row.by[0]
        num = int(row.by[1:])
        if series == "b":
            frags.seq.iloc[index] = seq[0:num]
        if series == "y":
            frags.seq.iloc[index] = seq[len(seq)-num:len(seq)]
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    return(frags)

def matchSeqs(seqlist, targets, decoys):
    matches = []
    for r, s in seqlist:
        # target_matches.append(len([t for t in targets if s in t])) # ONLY COUNTS ONE OCCURENCE IN EACH PROTEIN
        # decoy_matches.append(len([d for d in decoys if s in d]))
        target_matches = len(re.findall(s, targets))
        decoy_matches = len(re.findall(s, decoys))
        matches.append((target_matches, decoy_matches))
    return(matches)

def main(args):
    ##################################
    ## PREPARE SEQUENCES FOR SEARCH ##
    ##################################
    AAs = {"A":71.037114, "R":156.101111, "N":114.042927, "D":115.026943,
           "C":103.009185, "E":129.042593, "Q":128.058578, "G":57.021464,
           "H":137.058912, "I":113.084064, "L":113.084064, "K":128.094963,
           "M":131.040485, "F":147.068414, "P":97.052764, "S":87.032028,
           "T":101.047679, "U":150.953630, "W":186.079313, "Y":163.063329,
           "V":99.068414, "O":132.089878, "Z":129.042594}
    combos = []
    for i in range(1,9):
        combo = itertools.combinations(range(1,9), i)
        for c in combo:
            combos.append(c)
    sequences = pd.read_csv(r"S:\LAB_JVC\RESULTADOS\AndreaLaguillo\pyVseq\EXPLORER\PEPTIDES\peptide_list.txt", header=None)
    sequences.columns = ["SEQUENCE"]
    sequences["LENGTH"] = sequences.SEQUENCE.str.len()
    sequences = sequences.sort_values(by=['LENGTH'])
    
    results = []
    for index, sequence in sequences.iterrows():
        print(index)
        sequence = str(sequence[0])
        frags = makeFrags(sequence)
        frags["MZ"] = frags.apply(lambda x: round(getTheoMZ(AAs, 1, x.seq, x.series)[0],6), axis=1)
        pepmass = round(getTheoMZ(AAs, 1, sequence, "y")[0],6)
        for combo in combos:
            subset = frags[frags.region.isin(combo)]
            fragseqs = []
            for i,j in subset.groupby("region"):
                fragseqs.append((i, j.iloc[np.where(j["seq"].str.len() == j["seq"].str.len().max())[0]]["seq"].iloc[0]))
            results.append([sequence, combo, subset.by.to_list(), fragseqs])
    results = pd.DataFrame(results, columns=["SEQUENCE", "REGIONS", "FRAGMENTS", "SUBSEQUENCES"])

    ##############################
    ## PREPARE FASTA FOR SEARCH ##
    ##############################
    fasta = SeqIO.parse(open(r"S:\U_Proteomica\UNIDAD\iSanXoT_DBs\202105\human_202105_uni-sw-tr.target-decoy.fasta"),'fasta')
    targets = []
    decoys = []
    for f in fasta:
        if "DECOY" in f.id:
            decoys.append(str(f.seq))
        else:
            targets.append(str(f.seq))
    jtargets = '\n'.join(targets)
    jdecoys = '\n'.join(decoys)
            
    ############
    ## SEARCH ##
    ############
    results["COUNTS"] = results.apply(lambda x: matchSeqs(x.SUBSEQUENCES, jtargets, jdecoys), axis=1)
    
    ##############
    ## PLOTTING ##
    ##############
    
    return
    
    
# Separate target and decoys, two heatmaps