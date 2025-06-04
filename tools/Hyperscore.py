import math
import numpy as np
import pandas as pd
from bisect import bisect_left
import glob
import itertools
import logging
import os
from pathlib import Path
import pyopenms
import re
import sys

def _decimal_places(x):
    s = str(x)
    if not '.' in s:
        return 0
    return len(s) - s.index('.') - 1

def getTheoMH(sequence, nt, ct, mass,
              m_proton, m_hydrogen, m_oxygen):
    '''    
    Calculate theoretical MH using the PSM sequence.
    '''
    AAs = dict(mass._sections['Aminoacids'])
    MODs = dict(mass._sections['Fixed Modifications'])
    # total_aas = 2*m_hydrogen + m_oxygen
    total_aas = m_proton
    # total_aas += charge*m_proton
    #total_aas += float(MODs['nt']) + float(MODs['ct'])
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
        # if i in pos:
        #     total_aas += float(mods[pos.index(i)]) TODO: add mod mass outside
    # MH = total_aas - m_proton
    return(total_aas)

def theoSpectrum(seq, blist, ylist, mods, pos, mass,
                 m_proton, m_hydrogen, m_oxygen, charge, dm=0):
    ## Y SERIES ##
    outy = []
    for i in ylist:
        yn = list(seq[-i:])
        if i < len(seq): nt = False
        else: nt = True
        fragy = getTheoMH(yn,nt,True,mass,
                          m_proton,m_hydrogen,m_oxygen) + 2*m_hydrogen + m_oxygen + dm
        outy += [fragy]
    ## B SERIES ##
    outb = []
    for i in blist:
        bn = list(seq[:i][::-1])
        if i > 0: ct = False
        else: ct = True
        fragb = getTheoMH(bn,True,ct,mass,
                          m_proton,m_hydrogen,m_oxygen) + dm # TODO only add +dm to fragments up until n_pos
        outb += [fragb]
    ## ADD FIXED MODS ## # TODO two modes, use mods from config file or input table
    # for i, m in enumerate(mods):
        # bpos = range(0, pos[mods.index(i)]+1)
        # ypos = range(len(seq)-pos[mods.index(i)]-1, len(seq))
        # bpos = pos[i]
        # ypos = len(seq)-pos[i]-1
        # spec[0] = spec[0][:bpos] + [b + m for b in spec[0][bpos:]]
        # spec[1] = spec[1][:ypos] + [y + m for y in spec[1][ypos:]]
    ## FRAGMENT MATRIX ##
    spec = []
    for c in range(1, charge+1):
        if c > 1:
            coutb = [(i+(c-1)*m_proton)/c for i in outb]
            couty = [(i+(c-1)*m_proton)/c for i in outy]
        else:
            coutb, couty = outb, outy
        spec += [[coutb, couty]]
    return(spec)

def addMod(spec, dm, pos, len_seq, blist, ylist):
    ## ADD MOD TO SITES ##
    bpos = [i >= pos+1 for i in blist]
    ypos = [i >= len_seq-pos for i in ylist][::-1]
    spec[0] = [spec[0][i]+dm if bpos[i]==True else spec[0][i] for i in list(range(0,len(spec[0])))]
    spec[1] = [spec[1][i]+dm if ypos[i]==True else spec[1][i] for i in list(range(0,len(spec[1])))]
    return(spec)

def makeFrags(seq, ch, full_y): # TODO: SLOW
    '''
    Name all fragments.
    '''
    bp = bh = yp = yh = []
    # if 'P' in seq[:-1]: # Cannot cut after
    #     # Disallowed b ions
    #     bp = [pos+1 for pos, char in enumerate(seq) if char == 'P']
    #     # Disallowed y ions
    #     yp = [pos for pos, char in enumerate(seq[::-1]) if char == 'P']
    # if 'H' in seq[1:]: # Cannot cut before
    #     # Disallowed b ions
    #     bh = [pos for pos, char in enumerate(seq) if char == 'H']
    #     # Disallowed y ions
    #     yh = [pos+1 for pos, char in enumerate(seq[::-1]) if char == 'H']
    seq_len = len(seq)
    blist = list(range(1,seq_len))
    blist = [i for i in blist if i not in bp + bh]
    ylist = list(range(1+int(full_y),seq_len+1))
    ylist = [i for i in ylist if i not in yp + yh]
    frags = []
    frags_m = []
    for c in range(1, ch+1):
        frags += [[["b" + str(i) + "+"*c for i in blist], ["y" + str(i) + "+"*c for i in ylist]]]
        frags_m += [[["b" + str(i) + "*" + "+"*c for i in blist], ["y" + str(i) + "*" + "+"*c for i in ylist]]]
    return(frags, frags_m, blist, ylist)

def fragCheck(plainseq, blist, ylist, dm_pos, charge):
    # ballowed = ['b'+str(i)+'*' if i >= dm_pos+1 else 'b'+str(i) for i in blist] * charge
    # yallowed = ['y'+str(i)+'*' if i >= len(plainseq)-dm_pos else 'y'+str(i) for i in ylist] * charge
    # cballowed = list(itertools.chain.from_iterable([['+'*i]*len(blist) for i in range(1,charge+1)]))
    # cyallowed = [['+'*i]*len(ylist) for i in range(1,charge+1)]
    if charge == 1:
        allowed = (['b'+str(i)+'*+' if i >= dm_pos+1 else 'b'+str(i)+'+' for i in blist] +
                   ['y'+str(i)+'*+' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'+' for i in ylist])
    elif charge == 2:
        allowed = (['b'+str(i)+'*+' if i >= dm_pos+1 else 'b'+str(i)+'+' for i in blist] +
                   ['y'+str(i)+'*+' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'+' for i in ylist] +
                   ['b'+str(i)+'*++' if i >= dm_pos+1 else 'b'+str(i)+'++' for i in blist] +
                   ['y'+str(i)+'*++' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'++' for i in ylist])
    elif charge == 3:
        allowed = (['b'+str(i)+'*+' if i >= dm_pos+1 else 'b'+str(i)+'+' for i in blist] +
                   ['y'+str(i)+'*+' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'+' for i in ylist] +
                   ['b'+str(i)+'*++' if i >= dm_pos+1 else 'b'+str(i)+'++' for i in blist] +
                   ['y'+str(i)+'*++' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'++' for i in ylist] +
                   ['b'+str(i)+'*+++' if i >= dm_pos+1 else 'b'+str(i)+'+++' for i in blist] +
                   ['y'+str(i)+'*+++' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'+++' for i in ylist])
    else:
        allowed = (['b'+str(i)+'*+' if i >= dm_pos+1 else 'b'+str(i)+'+' for i in blist] +
                   ['y'+str(i)+'*+' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'+' for i in ylist] +
                   ['b'+str(i)+'*++' if i >= dm_pos+1 else 'b'+str(i)+'++' for i in blist] +
                   ['y'+str(i)+'*++' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'++' for i in ylist] +
                   ['b'+str(i)+'*+++' if i >= dm_pos+1 else 'b'+str(i)+'+++' for i in blist] +
                   ['y'+str(i)+'*+++' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'+++' for i in ylist] +
                   ['b'+str(i)+'*++++' if i >= dm_pos+1 else 'b'+str(i)+'++++' for i in blist] +
                   ['y'+str(i)+'*++++' if i >= len(plainseq)-dm_pos else 'y'+str(i)+'++++' for i in ylist])
    return(allowed)

def findClosest(dm, dmdf, dmtol, pos):
    cand = [i for i in range(len(dmdf[1])) if dmdf[1][i] > dm-dmtol and dmdf[1][i] < dm+dmtol]
    closest = pd.DataFrame([dmdf[0][cand], dmdf[1][cand], dmdf[2][cand], dmdf[3][cand]]).T
    closest.columns = ['name', 'mass', 'site', 'site_tiebreaker']
    closest = pd.concat([closest, pd.Series({'name':'EXPERIMENTAL', 'mass':dm, 'site':[pos], 'site_tiebreaker':[pos]}).to_frame().T], ignore_index=True)
    return(closest)

def findPos(dm_set, plainseq): # TODO fix sites now that this is array instead of DF
    def _where(sites, plainseq):
        sites = sites.site
        subpos = []
        for s in sites:
            if s == 'Anywhere' or s == 'exp':
                subpos = list(range(0, len(plainseq)))
                break
            elif s == 'Non-modified':
                subpos = [-1]
                break
            elif s == 'N-term':
                subpos = [0]
            elif s == 'C-term':
                subpos = [len(plainseq) - 1]
            else:
                subpos = subpos + list(np.where(np.array(list(plainseq)) == str(s))[0])
        subpos = list(dict.fromkeys(subpos))
        subpos.sort()
        return(subpos)
    dm_set['idx'] = dm_set.apply(_where, plainseq=plainseq, axis=1)
    dm_set = dm_set[dm_set.idx.apply(lambda x: len(x)) > 0]
    return(dm_set)

def getClosestIon(spectrum, ion):
    pos = bisect_left(spectrum, ion)
    if pos == 0:
        return spectrum[0]
    if pos == len(spectrum):
        return spectrum[-1]
    before = spectrum[pos - 1]
    after = spectrum[pos]
    if after - ion < ion - before:
        return(after)
    else:
        return(before)
    
def hyperscore(exp_spec, theo_spec, frags, ftol):
    assigned_mz = [getClosestIon(exp_spec[0], mz) for mz in theo_spec]
    assigned_ppm = np.absolute(np.divide(np.subtract(assigned_mz, theo_spec), theo_spec)*1000000)
    assigned_mask = assigned_ppm <= ftol
    assigned_mz = list(itertools.compress(assigned_mz, assigned_mask))
    if len(assigned_mz) == 0: return([], [], [], 0, 0, 0, 0)
    else:
        assigned_frags = list(itertools.compress(frags, assigned_mask))
        assigned_int = [exp_spec[1][list(exp_spec[0]).index(mz)] for mz in assigned_mz]
        assigned_int_mask = [f[0]=='b' for f in assigned_frags]
        i_b = sum(list(itertools.compress(assigned_int, assigned_int_mask)))
        i_y = sum(list(itertools.compress(assigned_int, ~np.array(assigned_int_mask))))
        i_sum = i_b + i_y
        n_b = len(set([f.replace('+', '') for f in assigned_frags if f[0]=='b']))
        n_y = len(set([f.replace('+', '') for f in assigned_frags if f[0]=='y']))
        if i_b == 0: i_b = 1
        if i_y == 0: i_y = 1
        hs = math.log((i_b) * (i_y)) + math.log(math.factorial((n_b))) + math.log(math.factorial(n_y))
    return(assigned_mz, assigned_int, assigned_frags, n_b, n_y, i_sum, hs)

def scoreVseq(sub, plainseq, mods, pos, mass, ftol, dmtol, dm, m_proton, m_hydrogen, m_oxygen, ttol, tmin, score_mode, full_y):
    ## ASSIGNDB ##
    # assigndblist = []
    # assigndb = []
    ## FRAGMENT NAMES ##
    charge = sub.Charge
    if charge >= 4: charge = 4
    frags, frags_m, blist, ylist = makeFrags(plainseq, charge, full_y)
    ## DM ##
    theo_spec = theoSpectrum(plainseq, blist, ylist, mods, pos, mass,
                             m_proton, m_hydrogen, m_oxygen, charge)
    flat_theo_spec = sum(sum(theo_spec, []), [])
    flat_frags = sum(sum(frags, []), [])
    f_len = len(flat_frags)
    
    ## NON-MODIFIED ##
    NM_mz, NM_int, NM_frags, NM_n_b, NM_n_y, NM_i, NM_hs = hyperscore(sub.Spectrum, flat_theo_spec, flat_frags, ftol)
    
    ## DM OPERATIONS ##
    mod_results = [0, 0, 0, None, None]
    hyb_results = [0, 0, 0, None, None]   
    for dm_pos in range(len(plainseq)):
        ## MOD HYPERSCORE ##
        allowed_mod = fragCheck(plainseq, blist, ylist, dm_pos, charge) # TODO support charge states > 4
        theo_spec_mod = [flat_theo_spec[i]+dm if '*' in allowed_mod[i] else flat_theo_spec[i] for i in range(0, f_len)]
        MOD_mz, MOD_int, MOD_frags, MOD_n_b, MOD_n_y, MOD_i, MOD_hs = hyperscore(sub.Spectrum, theo_spec_mod, allowed_mod, ftol)
        ## HYBRID HYPERSCORE ##
        HYB_frags = [i for i in MOD_frags if i not in NM_frags]
        if len(HYB_frags) == 0:
            HYB_int, HYB_frags, HYB_n_b, HYB_n_y, HYB_i, HYB_hs = NM_int, NM_frags, NM_n_b, NM_n_y, NM_i, NM_hs
        else:
            HYB_int = [i for i in MOD_int if i not in NM_int] + NM_int
            HYB_frags += NM_frags
            HYB_n_b = len(set([f.replace('+', '').replace('*', '') for f in HYB_frags if f[0]=='b']))
            HYB_n_y = len(set([f.replace('+', '').replace('*', '') for f in HYB_frags if f[0]=='y']))
            HYB_int_mask = [f[0]=='b' for f in HYB_frags]
            HYB_i_b = sum(list(itertools.compress(HYB_int, HYB_int_mask)))
            HYB_i_y = sum(list(itertools.compress(HYB_int, ~np.array(HYB_int_mask))))
            HYB_i = HYB_i_b + HYB_i_y
            if HYB_i_b == 0: HYB_i_b = 1
            if HYB_i_y == 0: HYB_i_y = 1
            HYB_hs = math.log((HYB_i_b) * (HYB_i_y)) + math.log(math.factorial((HYB_n_b))) + math.log(math.factorial(HYB_n_y))
        ## STORE RESULTS ##
        if HYB_hs > hyb_results[2]:
            hyb_results = [HYB_n_b+HYB_n_y, HYB_i, HYB_hs, dm_pos, HYB_frags]
        if MOD_hs > mod_results[2]:
            mod_results = [MOD_n_b+MOD_n_y, MOD_i, MOD_hs, dm_pos, MOD_frags]
    # TODO calculate range of equal hyperscores for site
    if score_mode:
        return(HYB_hs, HYB_n_b+HYB_n_y, [f for f in HYB_frags if f[0]=='b'], [f for f in HYB_frags if f[0]=='y'], HYB_i, dm_pos)
    else:
        return(MOD_hs, MOD_n_b+MOD_n_y, [f for f in MOD_frags if f[0]=='b'], [f for f in MOD_frags if f[0]=='y'], MOD_i, dm_pos)