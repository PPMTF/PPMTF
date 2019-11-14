#!/usr/bin/env python3
import numpy as np
import csv
import sys

################################# Parameters ##################################
#sys.argv = ["MakeTrainTestData_FS.py", "IS"]
#sys.argv = ["MakeTrainTestData_FS.py", "NY"]
#sys.argv = ["MakeTrainTestData_FS.py", "TK"]
#sys.argv = ["MakeTrainTestData_FS.py", "JK"]
#sys.argv = ["MakeTrainTestData_FS.py", "KL"]
#sys.argv = ["MakeTrainTestData_FS.py", "SP"]

if len(sys.argv) < 2:
    print("Usage:",sys.argv[0],"[City]")
    sys.exit(0)

# City
City = sys.argv[1]

# POI file (input)
POIFile = "../data/FS/POI_XX.csv"
# Trace file (input)
TraceFile = "../data/FS/traces_XX.csv"
# Training user index file (output)
TUserIndexFile = "../data/FS/tuserindex_XX.csv"
# Testing user index file (output)
EUserIndexFile = "../data/FS/euserindex_XX.csv"
# POI index file (output)
POIIndexFile = "../data/FS/POIindex_XX.csv"
# Training trace file (output)
TrainTraceFile = "../data/FS/traintraces_XX.csv"
# Testing trace file (output)
TestTraceFile = "../data/FS/testtraces_XX.csv"
# Number of extracted POIs
ExtrPOINum = 1000
# Minimum number of locations per user
MinNumLoc = 1

# Ratio of testing users over all users
EUserRatio = 0.2

########################### Read POI & trace files ############################
# [output1]: poi_dic ({poi_id: [poi_index, y, x, category, counts]})
# [output2]: user_dic ({user_id: user_index})
# [output3]: ucount_dic ({user_id: counts})
# [output4]: trace_list ([user_id, poi_id, unixtime, dow, hour, min])
def ReadPOITrace():
    # Initialization
    poi_dic = {}
    poi_all_list = []
    ucount_dic= {}
    trace_list = []

    # Read a POI file --> poi_all_list ([poi_id, y, x, category, counts]})
    f = open(POIFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        poi_all_list.append([lst[0], lst[1], lst[2], lst[3], int(lst[4])])
    f.close()
    print("#POIs within the area of interest =", len(poi_all_list))

    # Sort poi_dic_all in descending order of counts
    poi_all_list.sort(key=lambda tup: tup[4], reverse=True)
    
    # Extract POIs --> poi_dic ({poi_id: [poi_index, y, x, category, counts]})
    for i in range(ExtrPOINum):
        lst = poi_all_list[i]
        poi_dic[lst[0]] = [len(poi_dic), lst[1], lst[2], lst[3], int(lst[4])]
    print("#Extracted POIs =", len(poi_dic))

    # Read a trace file --> ucount_dic ({user_id: counts}), 
    # trace_list ([user_id, poi_id, unixtime, dow, hour, min])
    f = open(TraceFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        if lst[1] not in poi_dic:
            continue
        trace_list.append([int(lst[0]), lst[1], float(lst[4]), lst[8], int(lst[9]), int(lst[10])])
        ucount_dic[int(lst[0])] = ucount_dic.get(int(lst[0]), 0) + 1
    f.close()

    # Sort trace_list in ascending order of (user_id, unixtime)
    trace_list.sort(key=lambda tup: (tup[0], tup[2]), reverse=False)

    # A dictionary of users whose number of locations is >= MinNumLoc --> user_dic ({user_id: user_index})
    user_dic = {}
    for user_id, counts in sorted(ucount_dic.items()):
        # Continue if the number of locations for the user is below MinNumLoc
        if counts < MinNumLoc:
            continue
        user_dic[user_id] = len(user_dic)

    print("#Users =", len(user_dic))
    print("#Checkins within the area of interest =", len(trace_list))
          
    return poi_all_list, poi_dic, user_dic, ucount_dic, trace_list

#################################### Main #####################################
# Fix a seed
np.random.seed(1)

# Replace XX with City
POIFile = POIFile.replace("XX", City)
TraceFile = TraceFile.replace("XX", City)
TUserIndexFile = TUserIndexFile.replace("XX", City)
EUserIndexFile = EUserIndexFile.replace("XX", City)
POIIndexFile = POIIndexFile.replace("XX", City)
TrainTraceFile = TrainTraceFile.replace("XX", City)
TestTraceFile = TestTraceFile.replace("XX", City)

# Read POI & trace files
poi_all_list, poi_dic, user_dic, ucount_dic, trace_list = ReadPOITrace()

# Number of testing users
EUserNum = int(len(user_dic) * EUserRatio)
# Number of training users
TUserNum = int(len(user_dic)) - EUserNum

# Randomly select training and testing users --> tuse_list, euser_list
rand_index = np.arange(len(user_dic))
tuser_list = np.zeros(len(user_dic))
euser_list = np.zeros(len(user_dic))
np.random.shuffle(rand_index)
for i in range(TUserNum):
    tuser_list[rand_index[i]] = 1
for i in range(TUserNum, TUserNum + EUserNum):
    euser_list[rand_index[i]] = 1

# Training and testing user dic --> tuser_dic, euser_dic
tuser_dic = {}
i = 0
for user_id in user_dic:
    if tuser_list[user_dic[user_id]] == 1:
        tuser_dic[user_id] = i
        i += 1
euser_dic = {}
for user_id in user_dic:
    if euser_list[user_dic[user_id]] == 1:
        euser_dic[user_id] = i
        i += 1

# Output training user index
f = open(TUserIndexFile, "w")
print("user_id,user_index", file=f)
writer = csv.writer(f, lineterminator="\n")
for user_id in user_dic:
    if tuser_list[user_dic[user_id]] == 1:
        lst = [user_id, tuser_dic[user_id]]
        writer.writerow(lst)
f.close()

# Output testing user index
f = open(EUserIndexFile, "w")
print("user_id,user_index", file=f)
writer = csv.writer(f, lineterminator="\n")
for user_id in user_dic:
    if euser_list[user_dic[user_id]] == 1:
        lst = [user_id, euser_dic[user_id]]
        writer.writerow(lst)
f.close()

# Output POI index
f = open(POIIndexFile, "w")
print("poi_id,poi_index,category,counts", file=f)
#print("poi_id,poi_index,category", file=f)
writer = csv.writer(f, lineterminator="\n")
for poi_id in poi_dic:
    lst = [poi_id, poi_dic[poi_id][0], poi_dic[poi_id][3], poi_dic[poi_id][4]]
    writer.writerow(lst)
f.close()

# Output training traces
f = open(TrainTraceFile, "w")
print("user_index,poi_index,category,unixtime,dow,hour,min", file=f)
writer = csv.writer(f, lineterminator="\n")
for event in trace_list:
    user_id = event[0]
    poi_id = event[1]
    if user_id in user_dic and tuser_list[user_dic[user_id]] == 1:
        lst = [tuser_dic[user_id],poi_dic[poi_id][0],poi_dic[poi_id][3],event[2],event[3],event[4],event[5]]
        writer.writerow(lst)
f.close()

# Output testing traces
f = open(TestTraceFile, "w")
print("user_index,poi_index,category,unixtime,dow,hour,min", file=f)
writer = csv.writer(f, lineterminator="\n")
for event in trace_list:
    user_id = event[0]
    poi_id = event[1]
    if user_id in user_dic and euser_list[user_dic[user_id]] == 1:
        lst = [euser_dic[user_id],poi_dic[poi_id][0],poi_dic[poi_id][3],event[2],event[3],event[4],event[5]]
        writer.writerow(lst)
f.close()
