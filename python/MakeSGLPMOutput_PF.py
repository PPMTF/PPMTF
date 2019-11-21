#!/usr/bin/env python3
import csv
import sys
import glob

################################# Parameters ##################################
#sys.argv = ["MakeSglpmOutput_PF.py", 500, 10]
if len(sys.argv) < 2:
    print("Usage:",sys.argv[0],"[#User] [#Trace]" )
    sys.exit(0)

# Prefix of the sg-LPM directory (input)
SglpmDir ="../data/PF/SGLT_TK/"
# Merged synthesized trace file (output)
MSynTraceFile = "../data/PF/SGLT_TK/data_syntraces.csv"
# Number of users
N = int(sys.argv[1])
# Number of traces per user
L = int(sys.argv[2])

############################ Read synthesized traces #############################
# [input1]: N -- Number of users
# [input2]: L -- Number of traces per user
# [input3]: dirname
# [output1]: syn_trace_list ([user_index, trace_index, time_index, poi_index])
def ReadSynTrace(N, L, dirname):
    # Initialization
    syn_trace_list = []
    
    for n in range(N):
        for l in range(L):
            syn_trace_file = dirname + "/out/out/user" + str(n+1) + "/synthetic-trace" + str(l)
    
            # Read synthesized traces
            f = open(syn_trace_file, "r")
            reader = csv.reader(f)
            for event in reader:
                time_index = int(event[1])
                poi_index = int(event[2])
                syn_trace_list.append([n, l, time_index-1, poi_index-1])
            f.close()
    
    return syn_trace_list

#################################### Main #####################################
print(SglpmDir)

dirname_lst = glob.glob(SglpmDir)
for dirname in dirname_lst:
    print(dirname)
    # Read synthesized traces
    syn_trace_list = ReadSynTrace(N, L, dirname)
    
    # Output merged synthesized trace
    msyn_trace_file = dirname + "_" + MSynTraceFile
    
    f = open(msyn_trace_file, "w")
    print("user,trace_no,time_slot,time_instant,poi_index,category", file=f)
    writer = csv.writer(f, lineterminator="\n")
    for (user_index, trace_index, time_index, poi_index) in syn_trace_list:
        s = [user_index,trace_index,time_index,0,poi_index,"-"]
        writer.writerow(s)
    f.close()
