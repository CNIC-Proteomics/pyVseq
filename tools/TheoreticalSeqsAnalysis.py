import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

infile = r"S:\LAB_JVC\RESULTADOS\AndreaLaguillo\pyVseq\EXPLORER\MGFS\NOISE-100\SEQUEST-HT\ALL_MSFs.tsv"
# infile2 = r"S:\LAB_JVC\RESULTADOS\AndreaLaguillo\pyVseq\EXPLORER\MGFS\NOISE-100\SEQUEST\ALL_MSFs.tsv"
outpath = r"S:\LAB_JVC\RESULTADOS\AndreaLaguillo\pyVseq\EXPLORER\MGFS\NOISE-100\SEQUEST-HT\PLOTS\\"
combos = r"S:\LAB_JVC\RESULTADOS\AndreaLaguillo\pyVseq\EXPLORER\MGFS\Combinations.tsv"

allmsf = pd.read_csv(infile, sep='\t')
# allmsf2 = pd.read_csv(infile2, sep='\t')
# allmsf2.FILE = allmsf2.FILE.str[:-4]
# allmsf2["SEQUEST-HT"] = allmsf2.ScoreValue
# allmsf2["SEQUEST-HT_SEQ"] = allmsf2.Sequence
# allmsf2["SCAN_ID"] = allmsf2.FILE.astype(str) + "_" + allmsf2.FirstScan.astype(str) + "_" + allmsf2.SearchEngineRank.astype(str)
# allmsf2 = allmsf2[["SCAN_ID","SEQUEST-HT", "SEQUEST-HT_SEQ"]]

allmsf.FILE = allmsf.FILE.str[:-7]
allmsf["LABEL"] = allmsf.Description.str[1:6]
allmsf.loc[allmsf.LABEL!="DECOY", "LABEL"] = "TARGET"
allmsf["LENGTH"] = allmsf.FILE.str.len()
allmsf["MATCH"] = np.where(allmsf.Sequence==allmsf.FILE, True, False)
allmsf["SCAN_ID"] = allmsf.FILE.astype(str) + "_" + allmsf.FirstScan.astype(str) + "_" + allmsf.SearchEngineRank.astype(str)
combos = pd.read_csv(combos, sep='\t')
combos["FirstScan"] = combos.SCAN
combos["CATEGORY"] = combos[["REGIONS", "INTENSITY", "PPM_ERROR", "NOISE"]].values.tolist()
combos["CATEGORY"] = combos["CATEGORY"].astype(str)
allmsf = pd.merge(allmsf, combos, on ='FirstScan', how ='inner')
# firsts = allmsf[allmsf.SearchEngineRank==1]

# XCORR COMPARISON SEQUEST VS SEQUEST-HT #
# comparison = pd.merge(allmsf, allmsf2, on ='SCAN_ID', how ='outer')
# comparison = comparison[comparison.SearchEngineRank==1]
# comparison['SEQUEST-HT'] = comparison['SEQUEST-HT'].fillna(0)
# comparison['ScoreValue'] = comparison['ScoreValue'].fillna(0)
# comparison["DIFF"] = comparison.ScoreValue - comparison["SEQUEST-HT"]
# plt.xlabel("Sequest", fontsize=15)
# plt.ylabel("Sequest-HT", fontsize=15)
# plt.scatter(comparison.ScoreValue, comparison["SEQUEST-HT"], s=0.1)

# comparison = pd.merge(allmsf, allmsf2, on ='SCAN_ID', how ='inner')
# comparison["SEQMATCH"] = np.where(comparison.Sequence==comparison['SEQUEST-HT_SEQ'], True, False)
# comparison = comparison[comparison.SEQMATCH==True]
# comparison = comparison[comparison.SearchEngineRank==1]
# comparison["DIFF"] = comparison.ScoreValue - comparison["SEQUEST-HT"]
# plt.xlabel("Sequest", fontsize=15)
# plt.ylabel("Sequest-HT", fontsize=15)
# plt.scatter(comparison.ScoreValue, comparison["SEQUEST-HT"], s=0.1)

# XCORR VS EACH LENGTH #
lengroups = allmsf.groupby("LENGTH")
xys = []
for seqlen, df in lengroups:
    df = df[df.SearchEngineRank==1]
    df = df.sort_values(by=['ScoreValue'], ascending=False)
    df.reset_index(drop=True, inplace=True)
    df["Number"] = df.index
    fig = plt.figure()
    fig.set_size_inches(20, 15)
    plt.xlabel("Number", fontsize=15)
    plt.ylabel("Xcorr", fontsize=15)
    plt.title("Peptide Length " + str(seqlen), fontsize=20)
    plt.scatter(df.Number, df.ScoreValue, marker="o", linewidth=0, color="darkblue", s=5)
    plt.show()
    xys.append([df.Number, df.ScoreValue, seqlen])
    outplot = Path(outpath + "Xcorr_Len"+ str(int(seqlen)) + ".pdf")
    fig.savefig(outplot)
    fig.clear()
    plt.close(fig)

# XCORR VS ALL LENGTHS #
fig = plt.figure()
fig.set_size_inches(20, 15)
plt.xlabel("Number", fontsize=20)
plt.ylabel("Xcorr", fontsize=20)
plt.title("Xcorr by Peptide Length", fontsize=30)
color = plt.cm.rainbow(np.linspace(0, 1, len(xys)))
for i, c in zip(xys, color):
    plt.scatter(i[0], i[1], marker="o", linewidth=0, s=5, color=c, label=i[2])
plt.legend(loc="upper right", fontsize=25, markerscale=5, title="Length", title_fontsize=25)
outplot = Path(outpath + "Xcorr_vs_Len.pdf")
fig.savefig(outplot)
fig.clear()
plt.close(fig)

# CHECK 1ST CANDIDATE MATCH, ALL LENGTHS#
firsts = allmsf[allmsf.SearchEngineRank==1]
matches = firsts[firsts.MATCH==True]
lens = list(matches.LENGTH.value_counts().index)
lens.sort()
for l in lens:
    lenmatches = matches[matches.LENGTH==l]
    allresults = []
    results = []
    for i, j in lenmatches.groupby("CATEGORY"):
        results.append([i, len(j), j.REGIONS.iloc[0], j.INTENSITY.iloc[0], j.PPM_ERROR.iloc[0], j.NOISE.iloc[0], j.LENGTH.iloc[0]])
    results = pd.DataFrame(results, columns=["CATEGORY", "FREQUENCY", "REGIONS", "INTENSITY", "PPM_ERROR", "NOISE", "LENGTH"])
    for i in combos.CATEGORY.tolist():
        if i not in results.CATEGORY.tolist():
            results = pd.concat([results, pd.DataFrame([[i, 0, 0, 0, 0, 0, 0]], columns=["CATEGORY", "FREQUENCY", "REGIONS", "INTENSITY", "PPM_ERROR", "NOISE", "LENGTH"])])
    results.sort_values(by=['FREQUENCY'], inplace=True, ascending=False)
    results.reset_index(drop=True, inplace=True)
    allresults.append((results, "ALL"))
    # optional filters
    subresult = results[results.NOISE=="YES"]
    subresult.reset_index(drop=True, inplace=True)
    allresults.append((subresult, "NOISE"))
    subresult = results[results.NOISE=="NO"]
    subresult.reset_index(drop=True, inplace=True)
    allresults.append((subresult, "NO_NOISE"))
    subresult = results[results.PPM_ERROR==0]
    subresult.reset_index(drop=True, inplace=True)
    allresults.append((subresult, "0_PPM"))
    subresult = results[results.PPM_ERROR==10]
    subresult.reset_index(drop=True, inplace=True)
    allresults.append((subresult, "10_PPM"))
    subresult = results[results.PPM_ERROR==40]
    subresult.reset_index(drop=True, inplace=True)
    allresults.append((subresult, "40_PPM"))
    subresult = results[results.INTENSITY==10]
    subresult.reset_index(drop=True, inplace=True)
    allresults.append((subresult, "10_INTENSITY"))
    subresult = results[results.INTENSITY==100]
    subresult.reset_index(drop=True, inplace=True)
    allresults.append((subresult, "100_INTENSITY"))
    subresult = results[results.INTENSITY==1000]
    subresult.reset_index(drop=True, inplace=True)
    allresults.append((subresult, "1000_INTENSITY"))
    # optional filters end
    fig = plt.figure()
    fig.suptitle("PEPTIDE LENGTH " + str(l), fontsize=15, x=0.2)
    fig.set_size_inches(10, 30)
    counter = 1
    for i, j in allresults[0:3]:
        ax1 = fig.add_subplot(len(allresults),1,counter)
        plt.xlabel("Category", fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        plt.title(str(j), fontsize=20)
        plt.scatter(list(i.index), i.FREQUENCY, s=0.1)
        counter += 1
    plt.tight_layout()
    outplot = Path(outpath + "LEN" + str(l) + "_1st_candidate_is_match_NOISE.png")
    fig.savefig(outplot,bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure()
    fig.suptitle("PEPTIDE LENGTH " + str(l), fontsize=15, x=0.2)
    fig.set_size_inches(10, 30)
    counter = 1
    for i, j in allresults[3:6]:
        ax1 = fig.add_subplot(len(allresults),1,counter)
        plt.xlabel("Category", fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        plt.title(str(j), fontsize=20)
        plt.scatter(list(i.index), i.FREQUENCY, s=0.1)
        counter += 1
    plt.tight_layout()
    outplot = Path(outpath + "LEN" + str(l) + "_1st_candidate_is_match_PPM.png")
    fig.savefig(outplot,bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure()
    fig.suptitle("PEPTIDE LENGTH " + str(l), fontsize=15, x=0.2)
    fig.set_size_inches(10, 30)
    counter = 1
    for i, j in allresults[6:9]:
        ax1 = fig.add_subplot(len(allresults),1,counter)
        plt.xlabel("Category", fontsize=20)
        plt.ylabel("Frequency", fontsize=20)
        plt.title(str(j), fontsize=20)
        plt.scatter(list(i.index), i.FREQUENCY, s=0.1)
        counter += 1
    plt.tight_layout()
    outplot = Path(outpath + "LEN" + str(l) + "_1st_candidate_is_match_INT.png")
    fig.savefig(outplot,bbox_inches='tight')
    plt.close(fig)
    
    regionscount = []
    for i, j in lenmatches.groupby("REGIONS"):
        regionscount.append([i, len(j)])
    regionscount = pd.DataFrame(regionscount, columns=["REGIONS", "COUNT"])
    regionscount.sort_values(by=['COUNT'], inplace=True, ascending=False)
    regionscount.reset_index(drop=True, inplace=True)
    fig = plt.figure()
    fig.suptitle("PEPTIDE LENGTH " + str(l), fontsize=15, x=0.2)
    fig.set_size_inches(20, 15)
    ax1 = fig.add_subplot(1,1,1)
    plt.scatter(list(regionscount.index), regionscount.COUNT, s=10)
    plt.tight_layout()
    outplot = Path(outpath + "LEN" + str(l) + "_1st_candidate_is_match_REGIONS.png")
    fig.savefig(outplot,bbox_inches='tight')
    plt.close(fig)

# CHECK 1ST CANDIDATE MATCH, EACH LENGTH#


# FDR #
def FDR(group, score_column):
    group.sort_values(by=[score_column, 'LABEL'], inplace=True, ascending=False)
    group['Rank'] = group.groupby('LABEL').cumcount()+1 # This column can be deleted later
    group['Rank_T'] = np.where(group['LABEL']=='TARGET', group['Rank'], 0)
    group['Rank_T'] = group['Rank_T'].replace(to_replace=0, method='ffill')
    group['Rank_D'] = np.where(group['LABEL'] == 'DECOY', group['Rank'], 0)
    group['Rank_D'] =  group['Rank_D'].replace(to_replace=0, method='ffill')
    # calculate peak FDR
    group['FDR'] = group['Rank_D']/group['Rank_T']
    return group
allmsf = FDR(allmsf, 'ScoreValue')
allmsf001 = allmsf.loc[allmsf.FDR <= 0.01]
# xcorr cutoff 0.998334