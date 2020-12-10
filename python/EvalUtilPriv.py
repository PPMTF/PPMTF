#!/usr/bin/env python3
import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import wasserstein_distance
import math
import csv
import sys
import glob
import os

################################# Parameters ##################################
#sys.argv = ["EvalUtilPriv.py", "PF", "TK", "PPMTF"]
#sys.argv = ["EvalUtilPriv.py", "PF", "TK", "PPITF"]
#sys.argv = ["EvalUtilPriv.py", "PF", "TK", "SGD"]
#sys.argv = ["EvalUtilPriv.py", "PF", "TK", "SGLT"]
#sys.argv = ["EvalUtilPriv.py", "FS", "IS", "PPMTF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "JK", "PPMTF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "KL", "PPMTF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "NY", "PPMTF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "TK", "PPMTF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "SP", "PPMTF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "IS", "PPITF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "JK", "PPITF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "KL", "PPITF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "NY", "PPITF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "TK", "PPITF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "SP", "PPITF", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "IS", "SGD", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "JK", "SGD", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "KL", "SGD", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "NY", "SGD", 1]
#sys.argv = ["EvalUtilPriv.py", "FS", "TK", "SGD", 1]
sys.argv = ["EvalUtilPriv.py", "FS", "SP", "SGD", 1]

if len(sys.argv) < 4:
    print("Usage:",sys.argv[0],"[Dataset] [City] [SynAlg (PPMTF/PPITF/SGD/SGLT)] ([TraceNum (default:10)] [ItrNum (default:100)] [PDTest (default:1)] [Reqk (default:10)])")
    sys.exit(0)

# Dataset (PF/FS)
DataSet = sys.argv[1]
# City
City = sys.argv[2]

# Synthesizing algorithm
SynAlg = sys.argv[3]

# Number of traces per user
TraceNum = 10
if len(sys.argv) >= 5:
    TraceNum = int(sys.argv[4])

# Number of iterations in Gibbs sampling (SynAlg = "PPMTF")
ItrNum = 100
if len(sys.argv) >= 6:
    ItrNum = int(sys.argv[5])

# Perform the PD test (1: yes, 0: no)
PDTest = 1
if len(sys.argv) >= 7:
    PDTest = int(sys.argv[6])

# Required k in plausible deniability
Reqk = 10
if len(sys.argv) >= 8:
    Reqk = int(sys.argv[7])

# Data directory
DataDir = "../data/" + DataSet + "/"

# Training user index file (input)
TUserIndexFile = DataDir + "tuserindex_XX.csv"
# Testing user index file (output)
EUserIndexFile = DataDir + "euserindex_XX.csv"
# POI file (input)
POIFile = DataDir + "POI_XX.csv"
# POI index file (input)
POIIndexFile = DataDir + "POIindex_XX.csv"
# Training trace file (input)
TrainTraceFile = DataDir + "traintraces_XX.csv"
# Testing trace file (input)
TestTraceFile = DataDir + "testtraces_XX.csv"

# Type of time slots (1: 9-19h, 20min, 2: 2 hours)
if DataSet == "PF":
    TimeType = 1
elif DataSet[0:2] == "FS":
    TimeType = 2
else:
    print("Wrong Dataset")
    sys.exit(-1)

# Maximum time interval between two temporally-continuous locations (sec) (-1: none)
if DataSet == "PF":
    MaxTimInt = -1
elif DataSet[0:2] == "FS":
    MaxTimInt = 7200

# Number of columns in model parameters (A, B, C)
K = 16

# Minimum probability
DeltaProb = 0.00000001
# Top-L POIs (Time-specific)
L1 = 50
# Number of bins in the visit-fraction distribution
if DataSet == "PF":
    B = 30
elif DataSet[0:2] == "FS":
    B = 24
else:
    print("Wrong Dataset")
    sys.exit(-1)

# Minimum/maximum of y/x
if DataSet == "PF" and City == "TK":
    MIN_Y = 35.65
    MAX_Y = 35.75
    MIN_X = 139.68
    MAX_X = 139.8
elif DataSet[0:2] == "FS" and City == "IS":
    MIN_Y = 40.8
    MAX_Y = 41.2
    MIN_X = 28.5
    MAX_X = 29.5
elif DataSet[0:2] == "FS" and City == "JK":
    MIN_Y = -6.4
    MAX_Y = -6.1
    MIN_X = 106.7
    MAX_X = 107.0
elif DataSet[0:2] == "FS" and City == "KL":
    MIN_Y = 3.0
    MAX_Y = 3.3
    MIN_X = 101.6
    MAX_X = 101.8
elif DataSet[0:2] == "FS" and City == "NY":
    MIN_Y = 40.5
    MAX_Y = 41.0
    MIN_X = -74.28
    MAX_X = -73.68
elif DataSet[0:2] == "FS" and City == "TK":
    MIN_Y = 35.5
    MAX_Y = 35.9
    MIN_X = 139.5
    MAX_X = 140.0
elif DataSet[0:2] == "FS" and City == "SP":
    MIN_Y = -24.0
    MAX_Y = -23.4
    MIN_X = -46.8
    MAX_X = -46.3
else:
    print("Wrong Dataset")
    sys.exit(-1)
# Number of regions on the x-axis
NumRegX = 20
# Number of regions on the y-axis
NumRegY = 20

if SynAlg == "PPMTF":
    # Prefix of the synthesized trace directory
    SynTraceDirPre = DataDir + "PPMTF_" + City
    # Synthesized trace directory
    SynTraceDir = SynTraceDirPre + "*/"
    # Synthesized trace file (with asterisk)
    SynTraceFileAst = SynTraceDir + "syntraces_Itr" + str(ItrNum) + ".csv"
    # Result file (output)
    ResFile = DataDir + "utilpriv_PPMTF_" + City + ".csv"
elif SynAlg == "PPITF":
    # Prefix of the synthesized trace directory
    SynTraceDirPre = DataDir + "PPITF_" + City
    # Synthesized trace directory
    SynTraceDir = SynTraceDirPre + "*/"
    # Synthesized trace file (with asterisk)
    SynTraceFileAst = SynTraceDir + "syntraces_Itr" + str(ItrNum) + ".csv"
    # Result file (output)
    ResFile = DataDir + "utilpriv_PPITF_" + City + ".csv"
elif SynAlg == "SGD":
    # Prefix of the synthesized trace directory  
    SynTraceDirPre = DataDir + "SGD_" + City
    # Synthesized trace directory
    SynTraceDir = SynTraceDirPre + "/"
    # Synthesized trace file (with asterisk)
    SynTraceFileAst = SynTraceDir + "syntraces_cn*.csv"
    # Result file (output)
    ResFile = DataDir + "utilpriv_SGD_" + City + ".csv"
elif SynAlg == "SGLT":
    # Prefix of the synthesized trace directory  
    SynTraceDirPre = DataDir + "SGLT_" + City
    # Synthesized trace directory
    SynTraceDir = SynTraceDirPre + "/"
    # Synthesized trace file (with asterisk)
    SynTraceFileAst = SynTraceDir + "*_syntraces.csv"
    # Result file (output)
    ResFile = DataDir + "utilpriv_SGLT_" + City + ".csv"
else:
    print("Wrong SynAlg")
    sys.exit(-1)

# Prefix of the model parameter file (input)
ModelParameterDir = DataDir + "PPMTF_" + City + "_alp200_mnt100_mnv100/"
ModelParameterFile = ModelParameterDir + "modelparameter"

######################### Read a training trace file ##########################
# [output1]: ttrans_count ({(user_index, poi_index_from, poi_index_to): counts})
# [output2]: ttrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [output3]: tcount_sum (N x M matrix)
def ReadTrainTraceFile():
    # Initialization
    ttrans_count = {}
    ttrans_prob = {}
    tcount_sum = np.zeros((N, M))
    user_index_prev = -1
    poi_index_prev = 0

    # Read a training trace file
    f = open(TrainTraceFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        user_index = int(lst[0])
        poi_index = int(lst[1])
        # Update a transition matrix if the event and the previous event are from the same user
        if user_index == user_index_prev:
            ttrans_count[(user_index, poi_index_prev, poi_index)] = ttrans_count.get((user_index, poi_index_prev, poi_index), 0) + 1
        user_index_prev = user_index
        poi_index_prev = poi_index
    f.close()

    # Make a count sum matrix --> tcount_sum
    for (user_index, poi_index_prev, poi_index), counts in sorted(ttrans_count.items()):
        tcount_sum[user_index, poi_index_prev] += counts

    # Make a transition probability tensor --> ttrans_prob
    for (user_index, poi_index_prev, poi_index), counts in sorted(ttrans_count.items()):
        ttrans_prob[(user_index, poi_index_prev, poi_index)] = counts / tcount_sum[user_index, poi_index_prev]

    return ttrans_count, ttrans_prob, tcount_sum

######################### Read a testing trace file ##########################
# [output1]: etrans_count ({(user_index, poi_index_from, poi_index_to): counts})
# [output2]: etrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [output3]: ecount_sum (N x M matrix)
def ReadTestTraceFile():
    # Initialization
    etrans_count = {}
    etrans_prob = {}
    ecount_sum = np.zeros((N, M))
    user_index_prev = -1
    poi_index_prev = 0

    # Read a testing trace file
    f = open(TestTraceFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        user_index = int(lst[0]) - N
        poi_index = int(lst[1])
        # Update a transition matrix if the event and the previous event are from the same user
        if user_index == user_index_prev:
            etrans_count[(user_index, poi_index_prev, poi_index)] = etrans_count.get((user_index, poi_index_prev, poi_index), 0) + 1
        user_index_prev = user_index
        poi_index_prev = poi_index
    f.close()

    # Make a count sum matrix --> ecount_sum
    for (user_index, poi_index_prev, poi_index), counts in sorted(etrans_count.items()):
        ecount_sum[user_index, poi_index_prev] += counts

    # Make a transition probability tensor --> etrans_prob
    for (user_index, poi_index_prev, poi_index), counts in sorted(etrans_count.items()):
        etrans_prob[(user_index, poi_index_prev, poi_index)] = counts / ecount_sum[user_index, poi_index_prev]

    return etrans_count, etrans_prob, ecount_sum

######################## MAP re-identification attack #########################
# [input1]: ttrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [input2]: tcount_sum (N x M matrix)
# [input3]: syn_trace_file
def MAPReidentify(ttrans_prob, tcount_sum, syn_trace_file):
    # Initialization
    log_post = np.zeros(N)
    reid_res = np.zeros(N*TraceNum)
    user_index_prev = 0
    poi_index_prev = 0

    # Read a synthesized trace file
    f = open(syn_trace_file, "r")
    reader = csv.reader(f)
    next(reader)
    user_no = 0
    time_slot = 0
    for lst in reader:
        user_index = int(lst[0])

        trace_index = int(lst[1])
        user_index = user_index * TraceNum + trace_index

        poi_index = int(lst[4])

        if user_index != user_index_prev:
            time_slot = 0

        # Update the log-posterior if the event and the previous event are from the same user
        if user_index == user_index_prev and time_slot >= 1:
            # For each user
            for n in range(N):
                if tcount_sum[n,poi_index_prev] > 0:
                    if (n, poi_index_prev, poi_index) in ttrans_prob:
                        log_post[n] += math.log(ttrans_prob[n,poi_index_prev,poi_index])
                    else:
                        log_post[n] += math.log(DeltaProb)
                else:
                    log_post[n] += math.log(DeltaProb)
        # Update the re-identification result if a new user appears --> reid_res
        elif user_index != user_index_prev:
            reid_res[user_no] = np.argmax(log_post)
            log_post = np.zeros(N)
            user_no += 1

        user_index_prev = user_index
        poi_index_prev = poi_index
        time_slot += 1
    f.close()

    # Update the re-identification result for the last user
    reid_res[user_no] = np.argmax(log_post)

    return log_post, reid_res

############# Likelihood-ratio-based membership inference attack ##############
# [input1]: ttrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [input2]: etrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [input3]: syn_trace_file
# [output1]: llr_per_trace ((TraceNum x (N+N2) matrix)
# [output2]: trace_thr (1000-dim vector)
# [output3]: trace_true_pos (1000-dim vector)
# [output4]: trace_true_neg (1000-dim vector)
# [output5]: trace_max_acc
# [output6]: trace_max_adv
def LRMIA(ttrans_prob, etrans_prob, syn_trace_file):
    # Initialization
    llr_per_trace = np.full((TraceNum, N+N2), -sys.float_info.max)

    # Membership inference for each training/testing user n
    for n in range(N+N2):
        # Population transition probability matrix --> pop_trans_prob
        pop_trans_prob = np.zeros((M, M))
        # Population transition matrix except for training user n
        for (user_index, poi_index_prev, poi_index), prob in sorted(ttrans_prob.items()):
            if n < N and user_index == n:
                continue
            pop_trans_prob[poi_index_prev, poi_index] += prob
        # Population transition matrix except for testing user n-N
        for (user_index, poi_index_prev, poi_index), prob in sorted(etrans_prob.items()):
            if n >= N and user_index == n-N:
                continue
            pop_trans_prob[poi_index_prev, poi_index] += prob
        pop_trans_prob /= (N+N2-1)

        # Read a synthesized trace file
        f = open(syn_trace_file, "r")
        reader = csv.reader(f)
        next(reader)
        
        # Initialization
        user_index_prev = 0
        poi_index_prev = 0
        time_slot = 0
        llr_trace = np.zeros(TraceNum)

        for lst in reader:
            user_index = int(lst[0])
            trace_index = int(lst[1])
#            user_index = user_index * TraceNum + trace_index
    
            poi_index = int(lst[4])
    
            if user_index != user_index_prev:
                time_slot = 0
    
            # Update the log-likelihood ratio if the event and the previous event are from the same user
            if user_index == user_index_prev and time_slot >= 1:
                # Update the log-likelihood ratio for the user --> llr_trace[trace_index]
                # Membership inference for training user n
                if n < N:
                    # Add the log-likelihood using the transition matrix of training user n
                    if (n, poi_index_prev, poi_index) in ttrans_prob:
                        llr_trace[trace_index] += math.log(ttrans_prob[n,poi_index_prev,poi_index])
                    else:
                        llr_trace[trace_index] += math.log(DeltaProb)
                # Membership inference for testing user n-N
                else:
                    # Add the log-likelihood using the transition matrix of testing user n-N
                    if (n-N, poi_index_prev, poi_index) in etrans_prob:
                        llr_trace[trace_index] += math.log(etrans_prob[n-N,poi_index_prev,poi_index])
                    else:
                        llr_trace[trace_index] += math.log(DeltaProb)
                # Subtract the log-likelihood using the population matrix
                if pop_trans_prob[poi_index_prev, poi_index] > 0:
                    llr_trace[trace_index] -= math.log(pop_trans_prob[poi_index_prev, poi_index])
                else:
                    llr_trace[trace_index] -= math.log(DeltaProb)

            # Update llr_per_trace if a new user appears
            elif user_index != user_index_prev:
                for tr in range(TraceNum):
                    if llr_per_trace[tr,n] < llr_trace[tr]:
                        llr_per_trace[tr,n] = llr_trace[tr]
#                    print("Update llr_per_trace.", user_index_prev, int(lst[0]), int(lst[1]), llr_trans, llr_trace)
                llr_trace = np.zeros(TraceNum)
    
            user_index_prev = user_index
            poi_index_prev = poi_index
            time_slot += 1
        f.close()

        # Update the log-likelihood ratio for the last user
        for tr in range(TraceNum):
            if llr_per_trace[tr,n] < llr_trace[tr]:
                llr_per_trace[tr,n] = llr_trace[tr]
#        print(n, llr_per_trace[0,n], llr_per_trace[1,n], llr_per_trace[2,n])

    # Calculate #true positive/negative using llr_per_trace --> trace_true_pos, trace_true_neg
    MIA_thr = np.zeros(1000)
    max_llr_per_trace = -sys.float_info.max
    min_llr_per_trace = -sys.float_info.max
    for tr in range(TraceNum):
        if max_llr_per_trace < max(llr_per_trace[tr]):
            max_llr_per_trace = max(llr_per_trace[tr])
        if min_llr_per_trace < min(llr_per_trace[tr]):
            min_llr_per_trace = min(llr_per_trace[tr])
    MIA_true_pos = np.zeros(1000)
    MIA_true_neg = np.zeros(1000)
    # For each threshold
    for i in range(1000):
        # Threshold --> thr
        MIA_thr[i] = min_llr_per_trace + (max_llr_per_trace - min_llr_per_trace) * i / 1000
        # True positive --> true_pos
        for tr in range(TraceNum):
            for n in range(N):
                if llr_per_trace[tr,n] > MIA_thr[i]:
                    MIA_true_pos[i] += 1
            # True negative --> true_neg
            for n in range(N2):
                if llr_per_trace[tr,N+n] <= MIA_thr[i]:
                    MIA_true_neg[i] += 1
    # Calculate the maximum accuracy using llr_per_trace --> MIA_max_acc
    MIA_max_acc = 0
    for i in range(1000):
        if MIA_max_acc < MIA_true_pos[i] + MIA_true_neg[i]:
            MIA_max_acc = MIA_true_pos[i] + MIA_true_neg[i]
    MIA_max_acc /= (TraceNum*(N+N2))

    # Calculate the maximum membership advantage using llr_per_trace --> MIA_max_adv
    MIA_max_adv = -sys.float_info.max
    for i in range(1000):
        if MIA_max_adv < MIA_true_pos[i]/(TraceNum*N) - 1 + MIA_true_neg[i]/(TraceNum*N2):
            MIA_max_adv = MIA_true_pos[i]/(TraceNum*N) - 1 + MIA_true_neg[i]/(TraceNum*N2)

    return llr_per_trace, MIA_thr, MIA_true_pos, MIA_true_neg, MIA_max_acc, MIA_max_adv

############################### Read POI files ################################
# [output1]: poi_dic ({poi_index: [y, x, y_id, x_id, category]})
def ReadPOI():
    # Initialization
    poi_dic = {}
    poi_file_dic = {}

    # Calculate the boundaries of the regions (NumRegX x NumRegY) --> xb, yb
    xb = np.zeros(NumRegX)
    yb = np.zeros(NumRegY)
    for i in range(NumRegX):
        xb[i] = MIN_X + (MAX_X - MIN_X) * i / NumRegX
    for i in range(NumRegY):
        yb[i] = MIN_Y + (MAX_Y - MIN_Y) * i / NumRegY
 
    # Read a POI file --> poi_file_dic ({poi_id: [y, x, y_id, x_id]})
    f = open(POIFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        y = float(lst[1])
        x = float(lst[2])

        x_id = NumRegX-1
        for i in range(NumRegX-1):
            if xb[i] <= x < xb[i+1]:
                x_id = i
                break
        y_id = NumRegY-1
        for i in range(NumRegY-1):
            if yb[i] <= y < yb[i+1]:
                y_id = i
                break

        poi_file_dic[lst[0]] = [y, x, y_id, x_id]
    f.close()

    # Read a POI index file --> poi_dic ({poi_index: [y, x, y_id, x_id, category]})
    f = open(POIIndexFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        y = poi_file_dic[lst[0]][0]
        x = poi_file_dic[lst[0]][1]
        y_id = poi_file_dic[lst[0]][2]
        x_id = poi_file_dic[lst[0]][3]
        poi_dic[int(lst[1])] = [y, x, y_id, x_id, lst[2]]
    f.close()

    return poi_dic

########################### Read model parameters #############################
# [output1]: ParamA (N x K matrix)
# [output2]: ParamB (M x K matrix)
# [output3]: ParamC (M x K matrix)
# [output4]: ParamD (T x K matrix)
def ReadModelParameters():
    # Read model parameter A
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_A.csv"
    f = open(infile, "r")
    ParamA = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read model parameter B
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_B.csv"
    f = open(infile, "r")
    ParamB = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read model parameter C
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_C.csv"
    f = open(infile, "r")
    ParamC = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read model parameter D
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_D.csv"
    f = open(infile, "r")
    ParamD = np.loadtxt(infile, delimiter=",")
    f.close()

    return ParamA, ParamB, ParamC, ParamD

############################## Read real traces ###############################
# [input1]: st_user_index -- Start user index
# [input2]: user_num -- Number of users
# [input3]: M -- Number of POIs
# [input4]: T -- Number of time slots
# [input5]: A_bin (N x K matrix)
# [input6]: trace_file -- Trace file
# [output1]: treal_dist (T x M matrix)
# [output2]: treal_count (T-dim vector)
# [output3]: real_trans (M x M matrix)
# [output4]: vf_dist (M x B matrix)
# [output5]: kreal_dist (K x (1 x M matrix))
# [output6]: ktreal_dist (K x (T x M matrix))
def ReadRealTraces(st_user_index, user_num, M, T, A_bin, trace_file):
    # Initialization
    ureal_visit = lil_matrix((user_num, M))
#    visitor_real_rate = np.zeros(M)
    vf = np.zeros(M)
    vf_dist = np.zeros((M, B))
    ureal_count = np.zeros(user_num)
    treal_dist = np.zeros((T, M))
    treal_count = np.zeros(T)
    real_trans = np.zeros((M, M))
    user_index_prev = -1
    poi_index_prev = 0
    unixtime_prev = 0
    time_ins_prev = 0
    kreal_dist = [0] * K
    for k in range(K):
        kreal_dist[k] = np.zeros((1, M))
    ktreal_dist = [0] * K
    for k in range(K):
        ktreal_dist[k] = np.zeros((T, M))

    # Read a real trace file
    f = open(trace_file, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        user_index = int(lst[0]) - st_user_index
        poi_index = int(lst[1])
        unixtime = float(lst[3])
        ho = int(lst[5])
        if TimeType == 1:
            if int(lst[6]) >= 40:
                mi = 2
            elif int(lst[6]) >= 20:
                mi = 1
            else:
                mi = 0
            time_slot = 3 * (ho - 9) + mi
            time_ins = time_slot
        elif TimeType == 2:
            time_slot = int(ho/2)
            time_ins = ho
        else:
            print("Wrong TimeType.\n")
            sys.exit(-1)

        # New user
        if user_index != user_index_prev and user_index_prev != -1:
            if ureal_count[user_index_prev] >= 5:
                # Normalize vf
                vf /= ureal_count[user_index_prev]
                # Update vf_dist
                for i in range(M):
                    # Continue if the visit-count is zero or one
                    if vf[i] == 0 or vf[i] == 1:
                        continue
                    vf_bin = math.ceil(vf[i] * B) - 1
                    vf_dist[i,vf_bin] += 1
            # Initialization
            vf = np.zeros(M)

        # Update the user-specific visit matrix --> ureal_visit
        ureal_visit[user_index, poi_index] = 1
        # Update the visit-fraction
        vf[poi_index] += 1

        # Update visit counts for the user --> ureal_count
        ureal_count[user_index] += 1
        # Update the time-specific real distribution --> treal_dist
        treal_dist[time_slot, poi_index] += 1

        # Update the time-specific real distribution for each basis --> kreal_dist, ktreal_dist
        for k in range(K):
            if A_bin[user_index, k] == True:
                kreal_dist[k][0, poi_index] += 1
                ktreal_dist[k][time_slot, poi_index] += 1

        # Update visit counts for the time instance --> treal_count
        treal_count[time_slot] += 1
        # Update a transition matrix if the event and the previous event are from the same user
        if user_index == user_index_prev and (MaxTimInt == -1 or unixtime - unixtime_prev <= MaxTimInt) and time_ins - time_ins_prev == 1:
            real_trans[poi_index_prev, poi_index] += 1
        user_index_prev = user_index
        poi_index_prev = poi_index
        unixtime_prev = unixtime
        time_ins_prev = time_ins
    f.close()

    # Last user
    if ureal_count[user_index_prev] >= 5:
        # Normalize the visit-fraction
        vf /= ureal_count[user_index_prev]
        # Update vf_dist
        for i in range(M):
            # Continue if the visit-count is zero or one
            if vf[i] == 0 or vf[i] == 1:
                continue
            vf_bin = math.ceil(vf[i] * B) - 1
            vf_dist[i,vf_bin] += 1

    # Normalize vf_dist
    for i in range(M):
        if np.sum(vf_dist[i]) > 0:
            vf_dist[i] /= np.sum(vf_dist[i])

    # Normalize treal_dist
    for t in range(T):
        if np.sum(treal_dist[t]) > 0:
            treal_dist[t] /= np.sum(treal_dist[t])
        else:
            treal_dist[t] = np.full(M, 1.0/float(M))

    # Normalize kreal_dist
    for k in range(K):
        if np.sum(kreal_dist[k][0]) > 0:
            kreal_dist[k][0] /= np.sum(kreal_dist[k][0])
        else:
            kreal_dist[k][0] = np.full(M, 1.0/float(M))

    # Normalize ktreal_dist
    for k in range(K):
        for t in range(T):
            if np.sum(ktreal_dist[k][t]) > 0:
                ktreal_dist[k][t] /= np.sum(ktreal_dist[k][t])
            else:
                ktreal_dist[k][t] = np.full(M, 1.0/float(M))

    # Normalize real_trans
    for i in range(M):
        if np.sum(real_trans[i]) > 0:
            real_trans[i] /= np.sum(real_trans[i])
        else:
            real_trans[i,i] = 1.0

    return treal_dist, treal_count, real_trans, vf_dist, kreal_dist, ktreal_dist

########################### Read synthesized traces ###########################
# [input1]: N -- Number of users
# [input2]: M -- Number of POIs
# [input3]: T -- Number of time slots
# [input4]: A_bin (N x K matrix)
# [input5]: syn_trace_file -- Synthesized trace file
# [input6]: pdtest_file -- PD test result file
# [input7]: req_k -- Required k
# [input8]: trace_no -- Trace no.
# [output1]: tsyn_dist (T x M matrix)
# [output2]: syn_trans (M x M matrix)
# [output3]: pass_test (N-dim vector)
# [output4]: vf_dist (M x B matrix)
# [output5]: ksyn_dist (K x (1 x M matrix))
# [output6]: ktsyn_dist (K x (T x M matrix))
def ReadSynTraces(N, M, T, A_bin, syn_trace_file, pdtest_file, req_k, trace_no):
    # Initialization
    usyn_visit = lil_matrix((N, M))
    vf = np.zeros(M)
    vf_dist = np.zeros((M, B))
    usyn_count = np.zeros(N)
    tsyn_dist = np.zeros((T, M))
    syn_trans = np.zeros((M, M))
    pass_test = np.ones(N)
    user_index_prev = -1
    poi_index_prev = 0
    ksyn_dist = [0] * K
    for k in range(K):
        ksyn_dist[k] = np.zeros((1, M))
    ktsyn_dist = [0] * K
    for k in range(K):
        ktsyn_dist[k] = np.zeros((T, M))

    # Read a PD test result file --> pass_test
    if pdtest_file != "none":
        infile = pdtest_file + "_Itr" + str(ItrNum) + ".csv"
        i = 0
        f = open(infile, "r")
        reader = csv.reader(f)
        for lst in reader:
            if lst[0] == "-":
                break
            k = float(lst[0])
            if k < req_k:
                pass_test[i] = 0
            i += 1
        print("Fraction of passing the PD test:", float(np.sum(pass_test)) / float(N), "(", np.sum(pass_test), "/", N, ")")

    # Read a real trace file
    f = open(syn_trace_file, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        user_index = int(lst[0])
        trace_no_cur = int(lst[1])
        time_slot = int(lst[2])
        poi_index = int(lst[4])

        if trace_no_cur != trace_no and trace_no != -1:
            continue

        if pass_test[user_index] == 1:
            # New user
            if user_index != user_index_prev and user_index_prev != -1:
                if usyn_count[user_index_prev] >= 5:
                    # Normalize vf
                    vf /= usyn_count[user_index_prev]
                    # Update vf_dist
                    for i in range(M):
                        # Continue if the visit-count is zero or one
                        if vf[i] == 0 or vf[i] == 1:
                            continue
                        vf_bin = math.ceil(vf[i] * B) - 1
                        vf_dist[i,vf_bin] += 1
                # Initialization
                vf = np.zeros(M)

            # Update the user-specific visit matrix --> usyn_visit
            usyn_visit[user_index, poi_index] = 1
            # Update the visit-fraction
            vf[poi_index] += 1
            # Update visit counts for the user --> usyn_count
            usyn_count[user_index] += 1
            # Update the time-specific synthesized distribution --> tsyn_dist
            tsyn_dist[time_slot, poi_index] += 1

            # Update the real distribution & time-specific real distribution for each basis --> ksyn_dist, ktsyn_dist
            for k in range(K):
                if A_bin[user_index, k] == True:
                    ksyn_dist[k][0, poi_index] += 1
                    ktsyn_dist[k][time_slot, poi_index] += 1

            # Update a transition matrix if the event and the previous event are from the same user
            if user_index == user_index_prev:
                syn_trans[poi_index_prev, poi_index] += 1
            user_index_prev = user_index
            poi_index_prev = poi_index
    f.close()

    # Last user
    if usyn_count[user_index_prev] >= 5:
        # Normalize the visit-fraction
        vf /= usyn_count[user_index_prev]
        # Update vf_dist
        for i in range(M):
            # Continue if the visit-count is zero or one
            if vf[i] == 0 or vf[i] == 1:
                continue
            vf_bin = math.ceil(vf[i] * B) - 1
            vf_dist[i,vf_bin] += 1

    # Normalize vf_dist
    for i in range(M):
        if np.sum(vf_dist[i]) > 0:
            vf_dist[i] /= np.sum(vf_dist[i])
        else:
            vf_dist[i,0] = 1.0

    # Normalize tsyn_dist
    for t in range(T):
        if np.sum(tsyn_dist[t]) > 0:
            tsyn_dist[t] /= np.sum(tsyn_dist[t])
        else:
            tsyn_dist[t] = np.full(M, 1.0/float(M))

    # Normalize ksyn_dist
    for k in range(K):
        if np.sum(ksyn_dist[k][0]) > 0:
            ksyn_dist[k][0] /= np.sum(ksyn_dist[k][0])
        else:
            ksyn_dist[k][0] = np.full(M, 1.0/float(M))

    # Normalize ktsyn_dist
    for k in range(K):
        for t in range(T):
            if np.sum(ktsyn_dist[k][t]) > 0:
                ktsyn_dist[k][t] /= np.sum(ktsyn_dist[k][t])
            else:
                ktsyn_dist[k][t] = np.full(M, 1.0/float(M))

    # Normalize syn_trans
    for i in range(M):
        if np.sum(syn_trans[i]) > 0:
            syn_trans[i] /= np.sum(syn_trans[i])
        else:
            syn_trans[i,i] = 1.0

    return tsyn_dist, syn_trans, pass_test, vf_dist, ksyn_dist, ktsyn_dist

#################### Calculate the average l1 & l2 losses #####################
# [input1]: dist1 (Z x M matrix)
# [input2]: dist2 (Z x M matrix)
# [input3]: pass_test (Z-dim vector)
# [input4]: Z -- Number of distributions
# [input5]: M -- Number of POIs
# [input6]: L -- Number of top POIs in dist1
# [output1]: l1_loss
# [output2]: l2_loss
def CalcL1L2(dist1, dist2, pass_test, Z, M, L):
    l1_loss = 0.0
    l2_loss = 0.0
    z_num = 0

    # l1 & l2 losses for all POIs
    if L == M:
        for z in range(Z):
            if pass_test[z] == 1:
                # Update the l1-loss & l2-loss
                for i in range(M):
                    l1_loss += np.abs(dist1[z,i] - dist2[z,i])
                    l2_loss += (dist1[z,i] - dist2[z,i])**2
                z_num += 1
    # l1 & l2 losses for the top L POIs
    else:
        for z in range(Z):
            if pass_test[z] == 1:
                # Sort indexes in descending order of dist1
                sortindex = np.argsort(dist1[z])[::-1]

                # Update the l1-loss & l2-loss
                for i in range(L):
                    j = sortindex[i]
                    l1_loss += np.abs(dist1[z,j] - dist2[z,j])
                    l2_loss += (dist1[z,j] - dist2[z,j])**2
                z_num += 1
        
    # Normalize l1_loss and l2_loss
    l1_loss /= z_num
    l2_loss /= z_num

    return l1_loss, l2_loss

##################### Calculate the average JS-divergence #####################
# [input1]: dist1 (Z x M matrix)
# [input2]: dist2 (Z x M matrix)
# [input3]: pass_test (Z-dim vector)
# [input4]: Z -- Number of distributions
# [input5]: M -- Number of POIs
# [input6]: L -- Number of top POIs in dist1
# [output1]: js
def CalcJS(dist1, dist2, pass_test, Z, M, L):
    # Initialization
    m_dist = np.zeros((Z, M))
    js = 0.0

    for z in range(Z):
        for i in range(M):
            m_dist[z,i] = (dist1[z,i] + dist2[z,i]) / 2.0

    # JS-divergence for all POIs
    if L == M:
        for z in range(Z):
            if pass_test[z] == 1:
                for i in range(M):
                    if dist1[z,i] > 0.0:
                        js += dist1[z,i] * math.log(dist1[z,i]/m_dist[z,i]) / 2
                    if dist2[z,i] > 0.0:
                        js += dist2[z,i] * math.log(dist2[z,i]/m_dist[z,i]) / 2
    # JS-divergence for all POIs
    else:
        for z in range(Z):
            if pass_test[z] == 1:
                sortindex = np.argsort(dist1[z])[::-1]
                for i in range(L):
                    j = sortindex[i]
                    if dist1[z,j] > 0.0:
                        js += dist1[z,j] * math.log(dist1[z,j]/m_dist[z,j]) / 2
                    if dist2[z,j] > 0.0:
                        js += dist2[z,j] * math.log(dist2[z,j]/m_dist[z,j]) / 2

    # Normalize js
    js /= np.sum(pass_test)

    return js

####### Calculate the l1-loss using visit-fraction distributions ########
# [input1]: vf_dist1 (M x B matrix)
# [input2]: vf_dist2 (M x B matrix)
# [input3]: vf_exist (M-dim vector)
# [input4]: M -- Number of POIs
# [input5]: B -- Number of bins
# [output1]: l1_loss
def CalcL1VfDist(vf_dist1, vf_dist2, vf_exist, M, B):
    # Initialization
    l1_loss = 0
    x = 0

    # For each POI
    for i in range(M):
        # i-th row (visit-fraction for the i-th POI) in vf_dist1 --> p1
        p1 = vf_dist1[i,:]
        # i-th row (visit-fraction for the i-th POI) in vf_dist2 --> p2
        p2 = vf_dist2[i,:]
        # Continue if either testing vf or training vf doesn't exist
        if vf_exist[i] == 0:
            continue
        # l1-loss between p1 & p2 --> l1_loss
        for j in range(B):
            l1_loss += np.abs(p1[j] - p2[j])
        x += 1

    # Calculate the average l1-loss
    l1_loss /= x

    return l1_loss

######### Calculate the EMD on the y/x-axis using transition matrices #########
# [input1]: trans1 (M x M matrix)
# [input2]: trans2 (M x M matrix)
# [input3]: poi_dic ({poi_index: [y, x, y_id, x_id]})
# [input4]: M -- Number of POIs
# [output1]: weight_avg_emd -- Weighted average EMD
def CalcEMDTransMat(trans1, trans2, poi_dic, M):
    # Initializaion
    avg_emd_y = 0
    avg_emd_x = 0
    y_axis_ids = np.arange(NumRegY)
    x_axis_ids = np.arange(NumRegX)

    # For each POI
    for i in range(M):
        # Initialization
        p1_y = np.zeros(NumRegY)
        p1_x = np.zeros(NumRegX)
        p2_y = np.zeros(NumRegY)
        p2_x = np.zeros(NumRegX)

        # i-th row (conditional probability from the i-th POI) in trans1 --> p1
        p1 = trans1[i,:]
        # p1 on the y-axis --> p1_y
        for j in range(M):
            y_id = poi_dic[j][2]
            p1_y[y_id] += p1[j]
        # p1 on the x-axis --> p1_x
        for j in range(M):
            x_id = poi_dic[j][3]
            p1_x[x_id] += p1[j]

        # i-th row (conditional probability from the i-th POI) in trans2 --> p2
        p2 = trans2[i,:]
        # p2 on the y-axis --> p2_y
        for j in range(M):
            y_id = poi_dic[j][2]
            p2_y[y_id] += p2[j]
        # p2 on the x-axis --> p2_x
        for j in range(M):
            x_id = poi_dic[j][3]
            p2_x[x_id] += p2[j]

        # EMD between p1_y & p2_y --> avg_emd_y
        emd_y = wasserstein_distance(y_axis_ids, y_axis_ids, p1_y, p2_y)
        avg_emd_y += emd_y

        # EMD between p1_x & p2_x --> avg_emd_x
        emd_x = wasserstein_distance(x_axis_ids, x_axis_ids, p1_x, p2_x)
        avg_emd_x += emd_x

    # Calculate the average EMD
    avg_emd_y /= M
    avg_emd_x /= M

    return avg_emd_y, avg_emd_x

########### Calculate the l1 & l2 losses using transition matrices ############
# [input1]: trans1 (M x M matrix)
# [input2]: trans2 (M x M matrix)
# [input3]: M -- Number of POIs
# [output1]: l1_loss
# [output2]: l2_loss
def CalcL1L2TransMat(trans1, trans2, M):
    # Initializaion
    l1_loss = 0.0
    l2_loss = 0.0
    m_num = 0

    # For each POI
    for i in range(M):
        l1_loss_row = 0.0
        l2_loss_row = 0.0
        # i-th row (conditional probability from the i-th POI) in trans1 --> p1
        p1 = trans1[i,:]
        # i-th row (conditional probability from the i-th POI) in trans2 --> p2
        p2 = trans2[i,:]

        if np.sum(p1) > 0:
            # L1 & L2 losses for the row
            for j in range(M):
                l1_loss_row += np.abs(p1[j] - p2[j])
                l2_loss_row += (p1[j] - p2[j])**2

            # Update the average L1 & L2 losses
            l1_loss += l1_loss_row
            l2_loss += l2_loss_row
            m_num += 1

#        print(i, l1_loss_row, l2_loss_row, l1_loss, l2_loss)
    l1_loss /= m_num
    l2_loss /= m_num

    return l1_loss, l2_loss

#################################### Main #####################################
# Replace XX with City
TUserIndexFile = TUserIndexFile.replace("XX", City)
EUserIndexFile = EUserIndexFile.replace("XX", City)
POIFile = POIFile.replace("XX", City)
POIIndexFile = POIIndexFile.replace("XX", City)
TrainTraceFile = TrainTraceFile.replace("XX", City)
TestTraceFile = TestTraceFile.replace("XX", City)
ResFile = ResFile.replace("XX", City)

# Number of training users --> N
N = len(open(TUserIndexFile).readlines()) - 1
# Number of testing users --> N2
N2 = len(open(EUserIndexFile).readlines()) - 1
# Number of POIs --> M
M = len(open(POIIndexFile).readlines()) - 1
# Number of time slots --> T
if TimeType == 1:
    T = 30
elif TimeType == 2:
    T = 12
else:
    print("Wrong TimeType.\n")
    sys.exit(-1)

# Read a training/testing trace file for the MAP re-identification attack (DataSet = PF)
if DataSet == "PF":
    ttrans_count, ttrans_prob, tcount_sum = ReadTrainTraceFile()
    etrans_count, etrans_prob, ecount_sum = ReadTestTraceFile()

# Read POI files
poi_dic = ReadPOI()

# Read model parameters
ParamA_bin = np.zeros((N,K))
if os.path.exists(ModelParameterDir):
    ParamA, ParamB, ParamC, ParamD = ReadModelParameters()

    # Normalize model parameters
    for k in range(K):
        l2_norm = np.linalg.norm(ParamA[:,k])
        ParamA[:,k] /= l2_norm
        l2_norm = np.linalg.norm(ParamB[:,k])
        ParamB[:,k] /= l2_norm
        l2_norm = np.linalg.norm(ParamC[:,k])
        ParamC[:,k] /= l2_norm
        l2_norm = np.linalg.norm(ParamD[:,k])
        ParamD[:,k] /= l2_norm

    for k in range(K):
        # Set the 90th percentile of ParamA as ParamAThr
        ParamAThr = np.percentile(ParamA[:,k], 90)
        # Binarize model parameter A using ParamAThr
        ParamA_bin[:,k] = ParamA[:,k] > ParamAThr

# Read training traces
ttrain_dist, ttrain_count, atrain_trans, vf_train_dist, ktrain_dist, kttrain_dist = ReadRealTraces(0, N, M, T, ParamA_bin, TrainTraceFile)

# Read testing traces
ttest_dist, ttest_count, atest_trans, vf_test_dist, ktest_dist, kttest_dist = ReadRealTraces(N, N2, M, T, ParamA_bin, TestTraceFile)

# Set vf_exist
vf_exist = np.ones(M)
for i in range(M):
    if np.sum(vf_train_dist[i]) == 0 or np.sum(vf_test_dist[i]) == 0:
        vf_exist[i] = 0

SynTraceFileLst = glob.glob(SynTraceFileAst)

f = open(ResFile, "w")
#print("tracedir, tracefile, reid_rate, MIA_acc, MIA_adv, -, TP-TV_syn, TP-MSE_syn, TP-JS_syn, TP-TV_tra, TP-MSE_tra, TP-JS_tra, TP-TV_uni, TP-MSE_uni, TP-JS_uni, -, TP-TV-Top50_syn, TP-MSE-Top50_syn, TP-JS-Top50_syn, TP-TV-Top50_tra, TP-MSE-Top50_tra, TP-JS-Top50_tra, TP-TV-Top50_uni, TP-MSE-Top50_uni, TP-JS-Top50_uni, -, VF-TV_syn, VF-TV_tra, VF-TV_uni, -, TM-EMD-Y_syn, TM-EMD-X_syn, TM-EMD-Y_tra, TM-EMD-X_tra, TM-EMD-Y_uni, TM-EMD-X_uni, -, CP-TV_syn, CP-MSE_syn, CP-JS_syn, CP-TV_uni, CP-MSE_uni, CP-JS_uni", file=f)
print("tracedir, tracefile, reid_rate, MIA_acc, MIA_adv, -, TP-TV_syn, TP-TV_tra, TP-TV_uni, -, TP-TV-Top50_syn, TP-TV-Top50_tra, TP-TV-Top50_uni, -, VF-TV_syn, VF-TV_tra, VF-TV_uni, -, TM-EMD-Y_syn, TM-EMD-X_syn, TM-EMD-Y_tra, TM-EMD-X_tra, TM-EMD-Y_uni, TM-EMD-X_uni", file=f)

######################### Utiility of the benchmark  ##########################
################### Time-specific Geo-distribution ####################
# Uniform distribution --> uni_dist
tuni_dist = np.full((T, M), 1.0/float(M))
tones = np.ones(T)

# Calculate the l1 & l2 losses between ttest_dist & ttrain_dist
ttra_l1_loss, ttra_l2_loss = CalcL1L2(ttest_dist, ttrain_dist, tones, T, M, M)
# Calculate the JS divergence between ttest_dist & ttrain_dist
ttra_js = CalcJS(ttest_dist, ttrain_dist, tones, T, M, M)

# Calculate the l1 & l2 losses between ttest_dist & tuni_dist
tuni_l1_loss, tuni_l2_loss = CalcL1L2(ttest_dist, tuni_dist, tones, T, M, M)
# Calculate the JS divergence between ttest_dist & tuni_dist
tuni_js = CalcJS(ttest_dist, tuni_dist, tones, T, M, M)

####################### Time-specific Top-L POIs ######################
# Calculate the l1 & l2 losses between test_dist & train_dist
ttral_l1_loss, ttral_l2_loss = CalcL1L2(ttest_dist, ttrain_dist, tones, T, M, L1)
# Calculate the JS divergence between test_dist & syn_dist
ttral_js = CalcJS(ttest_dist, ttrain_dist, tones, T, M, L1)

# Calculate the l1 & l2 losses between test_dist & uni_dist
tunil_l1_loss, tunil_l2_loss = CalcL1L2(ttest_dist, tuni_dist, tones, T, M, L1)
# Calculate the JS divergence between test_dist & uni_dist
tunil_js = CalcJS(ttest_dist, tuni_dist, tones, T, M, L1)

######### Visit-fraction distribution [Ye+,KDD11][Do+,TMC13] ##########
vtra_l1 = 0
vuni_l1 = 0
if DataSet == "FS":
    # Uniform distribution --> vf_uni_dist
    #vf_uni_dist = np.full((M, B), 1.0/float(B))
    vf_uni_dist = np.zeros((M, B))
    vf_bin = math.ceil(1.0*B/M) - 1
    for i in range(M):
        vf_uni_dist[i,vf_bin] = 1
    
    # Calculate the average EMD between vf_test_dist & vf_train_dist
    vtra_l1 = CalcL1VfDist(vf_test_dist, vf_train_dist, vf_exist, M, B)
    # Calculate the average EMD between vf_test_dist & vf_uni_dist
    vuni_l1 = CalcL1VfDist(vf_test_dist, vf_uni_dist, vf_exist, M, B)

################### Basis-specific Geo-distribution ###################
atrain_dist = np.zeros((1, M))
auni_dist = np.zeros((1, M))
aones = np.ones(1)
for t in range(T):
    atrain_dist += ttrain_dist[t] / T
for t in range(T):
    auni_dist += tuni_dist[t] / T

ktra_l1_loss = np.zeros(K)
ktra_l2_loss = np.zeros(K)
for k in range(K):
    # Calculate the l1 & l2 losses between ktrain_dist & atrain_dist
    ktra_l1_loss[k], ktra_l2_loss[k] = CalcL1L2(ktrain_dist[k], atrain_dist, aones, 1, M, M)
    # Sort indices in descending order of ktra_l1_loss --> ktra_l1_loss_index
    ktra_l1_loss_index = np.argsort(-ktra_l1_loss)

kuni_l1_loss = np.zeros(K)
kuni_l2_loss = np.zeros(K)
kuni_js = np.zeros(K)
for k in range(K):
    # Calculate the l1 & l2 losses between ktrain_dist & auni_dist
    kuni_l1_loss[k], kuni_l2_loss[k] = CalcL1L2(ktrain_dist[k], auni_dist, aones, 1, M, M)
    # Calculate the JS divergence between ktrain_dist & auni_dist
    kuni_js[k] = CalcJS(ktrain_dist[k], auni_dist, aones, 1, M, M)

kuni_l1_loss_max = np.max(kuni_l1_loss)
kuni_l2_loss_max = np.max(kuni_l2_loss)
kuni_js_max = np.max(kuni_js)

########################## Mobility features ##########################
# Uniform transition matrix --> uni_trans
auni_trans = np.full((M, M), 1.0/float(M))

# Calculate the average EMD on the y/x-axis between atest_trans & atrain_trans
atra_trans_emd_y, atra_trans_emd_x = CalcEMDTransMat(atest_trans, atrain_trans, poi_dic, M)
# Calculate the average EMD on the y/x-axis between atest_trans & auni_trans
auni_trans_emd_y, auni_trans_emd_x = CalcEMDTransMat(atest_trans, auni_trans, poi_dic, M)

# For each synthesized trace file
for SynTraceFile in SynTraceFileLst:
    SynTraceFile = SynTraceFile.replace("\\", "/")
    print("Evaluating", os.path.split(SynTraceFile)[0].split("/")[-1] + "/" + os.path.split(SynTraceFile)[1])

    if DataSet == "PF":
        # MAP (Maximum a Posteriori) re-identification attack --> reid_res
        log_post, reid_res = MAPReidentify(ttrans_prob, tcount_sum, SynTraceFile)
        reid_num = 0
        for i in range(N*TraceNum):
            if reid_res[i] == int(i / TraceNum):
                reid_num += 1
        reid_rate = reid_num / (N*TraceNum)

        # Likelihood-ratio-based MIA (Membership Inference Attack) --> mia_res
        llr_per_trace, MIA_thr, MIA_true_pos, MIA_true_neg, MIA_max_acc, MIA_max_adv = LRMIA(ttrans_prob, etrans_prob, SynTraceFile)
#        # Output the detailed results of MIA
#        outfile = DataDir + "utilpriv_MIA_" + os.path.split(SynTraceFile)[0].split("/")[-1] + "_" + os.path.split(SynTraceFile)[1]
#        f2 = open(outfile, "w")
#        print("thr, #true_pos, #true_neg, accuracy, advantage", file=f2)
#        writer = csv.writer(f2, lineterminator="\n")
#        for i in range(1000):
#            s = [MIA_thr[i], MIA_true_pos[i], MIA_true_neg[i], 
#                 (MIA_true_pos[i]+MIA_true_neg[i])/(TraceNum*(N+N2)),
#                 MIA_true_pos[i]/(TraceNum*N) - 1 + MIA_true_neg[i]/(TraceNum*N2)]            
#            writer.writerow(s)
#        f2.close()
    else:
        reid_rate = 0
        MIA_max_acc = 0
        MIA_max_adv = 0

    # Initialization
    tsyn_l1_loss_avg = 0
    tsyn_l2_loss_avg = 0
    tsyn_js_avg = 0

    tsynl_l1_loss_avg = 0
    tsynl_l2_loss_avg = 0
    tsynl_js_avg = 0

    vsyn_l1_avg = 0

    ksyn_l1_loss = np.zeros(K)
    ksyn_l2_loss = np.zeros(K)
    ksyn_js = np.zeros(K)

    asyn_trans_emd_y_avg = 0
    asyn_trans_emd_x_avg = 0

    # PD test result file --> PDTestResFile
    if DataSet == "FS" and PDTest == 1 and (SynAlg == "PPMTF" or SynAlg == "PPITF"):
        PDTestResFile = DataDir + os.path.split(SynTraceFile)[0].split("/")[-1] + "/" + "pdtest_res"
    else:
        PDTestResFile = "none"

    # For each trace no.
    for trace_no in range(TraceNum):
        # Read synthesized traces
        tsyn_dist, asyn_trans, pass_test, vf_syn_dist, ksyn_dist, ktsyn_dist = ReadSynTraces(N, M, T, ParamA_bin, SynTraceFile, PDTestResFile, Reqk, trace_no)

        ################### Time-specific Geo-distribution ####################
        # Calculate the l1 & l2 losses between ttest_dist & tsyn_dist
        tsyn_l1_loss, tsyn_l2_loss = CalcL1L2(ttest_dist, tsyn_dist, tones, T, M, M)
        tsyn_l1_loss_avg += tsyn_l1_loss
        tsyn_l2_loss_avg += tsyn_l2_loss
        # Calculate the JS divergence between ttest_dist & tsyn_dist
        tsyn_js = CalcJS(ttest_dist, tsyn_dist, tones, T, M, M)
        tsyn_js_avg += tsyn_js

        ####################### Time-specific Top-L POIs ######################
        # Calculate the l1 & l2 losses between ttest_dist & tsyn_dist
        tsynl_l1_loss, tsynl_l2_loss = CalcL1L2(ttest_dist, tsyn_dist, tones, T, M, L1)
        tsynl_l1_loss_avg += tsynl_l1_loss
        tsynl_l2_loss_avg += tsynl_l2_loss
        # Calculate the JS divergence between ttest_dist & tsyn_dist
        tsynl_js = CalcJS(ttest_dist, tsyn_dist, tones, T, M, L1)
        tsynl_js_avg += tsynl_js
    
        ######### Visit-fraction distribution [Ye+,KDD11][Do+,TMC13] ##########
        if DataSet == "FS":
            # Calculate the average EMD between vf_test_dist & vf_syn_dist
            vsyn_l1 = CalcL1VfDist(vf_test_dist, vf_syn_dist, vf_exist, M, B)
            vsyn_l1_avg += vsyn_l1
    
        ################### Basis-specific Geo-distribution ###################
        for k in range(K):
            # Calculate the l1 & l2 losses between ktrain_dist & ksyn_dist
            l1_loss, l2_loss = CalcL1L2(ktrain_dist[k], ksyn_dist[k], aones, 1, M, M)
            ksyn_l1_loss[k] += l1_loss
            ksyn_l2_loss[k] += l2_loss
            # Calculate the JS divergence between ktrain_dist & ksyn_dist
            js = CalcJS(ktrain_dist[k], ksyn_dist[k], aones, 1, M, M)
            ksyn_js[k] += js

        ########################## Mobility features ##########################
        # Calculate the average EMD on the y/x-axis between atest_trans & asyn_trans
        asyn_trans_emd_y, asyn_trans_emd_x = CalcEMDTransMat(atest_trans, asyn_trans, poi_dic, M)
        asyn_trans_emd_y_avg += asyn_trans_emd_y
        asyn_trans_emd_x_avg += asyn_trans_emd_x

    # Normalization
    tsyn_l1_loss_avg /= TraceNum
    tsyn_l2_loss_avg /= TraceNum
    tsyn_js_avg /= TraceNum

    tsynl_l1_loss_avg /= TraceNum
    tsynl_l2_loss_avg /= TraceNum
    tsynl_js_avg /= TraceNum

    vsyn_l1_avg /= TraceNum

    ksyn_l1_loss /= TraceNum
    ksyn_l2_loss /= TraceNum
    ksyn_js /= TraceNum

    ksyn_l1_loss_max= np.max(ksyn_l1_loss)
    ksyn_l2_loss_max = np.max(ksyn_l2_loss)
    ksyn_js_max = np.max(ksyn_js)

    asyn_trans_emd_y_avg /= TraceNum
    asyn_trans_emd_x_avg /= TraceNum

    # Output the results
    writer = csv.writer(f, lineterminator="\n")
    if DataSet == "PF":
#        s = [os.path.split(SynTraceFile)[0].split("/")[-1], os.path.split(SynTraceFile)[1], reid_rate, MIA_max_acc, MIA_max_adv, "-", 
#             tsyn_l1_loss_avg/2.0, tsyn_l2_loss_avg, tsyn_js_avg, ttra_l1_loss/2.0, ttra_l2_loss, ttra_js, tuni_l1_loss/2.0, tuni_l2_loss, tuni_js, "-", 
#             tsynl_l1_loss_avg/2.0, tsynl_l2_loss_avg, tsynl_js_avg, ttral_l1_loss/2.0, ttral_l2_loss, ttral_js, tunil_l1_loss/2.0, tunil_l2_loss, tunil_js, "-", 
#             "-", "-", "-", "-", 
#             asyn_trans_emd_y_avg, asyn_trans_emd_x_avg, atra_trans_emd_y, atra_trans_emd_x, auni_trans_emd_y, auni_trans_emd_x, "-", 
#             ksyn_l1_loss_max/2.0, ksyn_l2_loss_max, ksyn_js_max, kuni_l1_loss_max/2.0, kuni_l2_loss_max, kuni_js_max, "-"]
        s = [os.path.split(SynTraceFile)[0].split("/")[-1], os.path.split(SynTraceFile)[1], reid_rate, MIA_max_acc, MIA_max_adv, "-", 
             tsyn_l1_loss_avg/2.0, ttra_l1_loss/2.0, tuni_l1_loss/2.0, "-", 
             tsynl_l1_loss_avg/2.0, ttral_l1_loss/2.0, tunil_l1_loss/2.0, "-", 
             "-", "-", "-", "-", 
             asyn_trans_emd_y_avg, asyn_trans_emd_x_avg, atra_trans_emd_y, atra_trans_emd_x, auni_trans_emd_y, auni_trans_emd_x]
    else:
#        s = [os.path.split(SynTraceFile)[0].split("/")[-1], os.path.split(SynTraceFile)[1], reid_rate, MIA_max_acc, MIA_max_adv, "-", 
#             tsyn_l1_loss_avg/2.0, tsyn_l2_loss_avg, tsyn_js_avg, ttra_l1_loss/2.0, ttra_l2_loss, ttra_js, tuni_l1_loss/2.0, tuni_l2_loss, tuni_js, "-", 
#             tsynl_l1_loss_avg/2.0, tsynl_l2_loss_avg, tsynl_js_avg, ttral_l1_loss/2.0, ttral_l2_loss, ttral_js, tunil_l1_loss/2.0, tunil_l2_loss, tunil_js, "-", 
#             vsyn_l1_avg/2.0, vtra_l1/2.0, vuni_l1/2.0, "-", 
#             asyn_trans_emd_y_avg, asyn_trans_emd_x_avg, atra_trans_emd_y, atra_trans_emd_x, auni_trans_emd_y, auni_trans_emd_x, "-", 
#             ksyn_l1_loss_max/2.0, ksyn_l2_loss_max, ksyn_js_max, kuni_l1_loss_max/2.0, kuni_l2_loss_max, kuni_js_max, "-"]            
        s = [os.path.split(SynTraceFile)[0].split("/")[-1], os.path.split(SynTraceFile)[1], "-", "-", "-", "-", 
             tsyn_l1_loss_avg/2.0, ttra_l1_loss/2.0, tuni_l1_loss/2.0, "-", 
             tsynl_l1_loss_avg/2.0, ttral_l1_loss/2.0, tunil_l1_loss/2.0, "-", 
             vsyn_l1_avg/2.0, vtra_l1/2.0, vuni_l1/2.0, "-", 
             asyn_trans_emd_y_avg, asyn_trans_emd_x_avg, atra_trans_emd_y, atra_trans_emd_x, auni_trans_emd_y, auni_trans_emd_x]            
    writer.writerow(s)

f.close()
