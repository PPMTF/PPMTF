#!/usr/bin/env python3
import numpy as np
import csv
import sys

################################# Parameters ##################################
#sys.argv = ["MakeTrainTestData_PF.py", "PF", "TK"]
#sys.argv = ["MakeTrainTestData_PF.py", "FS", "IS"]
#sys.argv = ["MakeTrainTestData_PF.py", "FS", "NY"]
#sys.argv = ["MakeTrainTestData_PF.py", "FS", "TK"]
#sys.argv = ["MakeTrainTestData_PF.py", "FS", "JK"]
#sys.argv = ["MakeTrainTestData_PF.py", "FS", "KL"]
#sys.argv = ["MakeTrainTestData_PF.py", "FS", "SP"]

if len(sys.argv) < 3:
    print("Usage:",sys.argv[0],"[Dataset] [City] ([MaxNumTrans (default:100)] [MaxNumVisit (default:100)])")
    sys.exit(0)

# Dataset (PF/FS)
DataSet = sys.argv[1]
# City
City = sys.argv[2]

# Data directory
DataDir = "../data/" + DataSet + "/"
# Training user index file (input)
TUserIndexFile = DataDir + "tuserindex_XX.csv"
# Testing user index file (input)
EUserIndexFile = DataDir + "euserindex_XX.csv"
# POI index file (input)
POIIndexFile = DataDir + "POIindex_XX.csv"
# Training trace file (input)
TrainTraceFile = DataDir + "traintraces_XX.csv"

# Maximum number of transitions per user (-1: infinity)
MaxNumTrans = 100
if len(sys.argv) >= 4:
    MaxNumTrans = int(sys.argv[3])

# Maximum number of POI visits per user (-1: infinity)
MaxNumVisit = 100
if len(sys.argv) >= 5:
    MaxNumVisit = int(sys.argv[4])

# Training transition tensor file (output)
TrainTransTensorFile = DataDir + "traintranstensor_XX_mnt" + str(MaxNumTrans) + ".csv"

# Training visit tensor file (output)
TrainVisitTensorFile = DataDir + "trainvisittensor_XX_mnv" + str(MaxNumVisit) + ".csv"

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
else:
    print("Wrong Dataset")
    sys.exit(-1)

############################ Read training traces #############################
# [input1]: tracefile
# [output1]: train_trace_list ([user_id, poi_id, unixtime, dow, hour, min])
def ReadTrainTrace(tracefile):
    # Initialization
    train_trace_list = []

    # Read training traces
    f = open(tracefile, "r")
    reader = csv.reader(f)
    next(reader)
    for event in reader:
        user_id = int(event[0])
        poi_id = int(event[1])
        unixtime = float(event[3])
        dow = event[4]
        hour = int(event[5])
        mi = int(event[6])
        train_trace_list.append([user_id, poi_id, unixtime, dow, hour, mi])
    f.close()
    
    return train_trace_list

###################### Make a training transition tensor ######################
# [input1]: st_user_index -- Start user index
# [input2]: user_num -- Number of users
# [input3]: poi_num -- Number of POIs
# [input4]: train_trace_list ([user_id, poi_id, unixtime, dow, hour, min])
# [output1]: a ({(user_index, poi_index_from, poi_index_to): counts})
# [output2]: b ({(user_index, poi_index_from, poi_index_to): probability})
def MakeTrainTransTensor(st_user_index, user_num, poi_num, train_trace_list):
    # Initialization
    event_prev = [-1, 0, 0.0, '0', 0]
    a = {}
    b = {}
    trans_num = [0] * user_num
    count_sum = np.zeros((user_num, M))

    # Make a transition count tensor --> a
    time_ins_prev = 0
    for event in train_trace_list:
        ho = event[4]
        if TimeType == 1:
            if event[5] >= 40:
                mi = 2
            elif event[5] >= 20:
                mi = 1
            else:
                mi = 0
            time_ins = 3 * (ho - 9) + mi
        if TimeType == 2:
            time_ins = ho

        # Update a tensor if the event and the previous event are from the same user
        if event[0] == event_prev[0]:
            if event[2] - event_prev[2] < 0:
                print("Error: Unixtime is not sorted in ascending order.")
                sys.exit(1)
            # Consider only temporally-continuous locations within MaxTimInt for a transition
            if (MaxTimInt == -1 or event[2] - event_prev[2] <= MaxTimInt) and time_ins - time_ins_prev == 1:
                user_index = event[0] - st_user_index
                poi_index_to = event[1]
                poi_index_from = event_prev[1]
                a[(user_index, poi_index_from, poi_index_to)] = a.get((user_index, poi_index_from, poi_index_to), 0) + 1
        event_prev = event
        time_ins_prev = time_ins

    # Randomly delete counts for users whose number of transitions exceed MaxNumTrans --> a
    if MaxNumTrans != -1:
        # Calculate the number of transitions for each user --> trans_num
        for (user_index, poi_index_from, poi_index_to), counts in sorted(a.items()):
            trans_num[user_index] += 1
#        print("Max of trans_num:", max(trans_num))
        # Randomly delete counts for users whose number of transitions exceed MaxNumTrans
        user_index_prev = -1
        i = 0
        for (user_index, poi_index_from, poi_index_to), counts in sorted(a.items()):
            if user_index != user_index_prev:
                i = 0
                if trans_num[user_index] > MaxNumTrans:
                    rand_index = np.arange(trans_num[user_index])
                    np.random.shuffle(rand_index)
                    del_trans = np.zeros(trans_num[user_index])
                    for j in range(trans_num[user_index] - MaxNumTrans):
                        del_trans[rand_index[j]] = 1
#                    print("Deleted transitions:", user_index + st_user_index, del_trans)
            if trans_num[user_index] > MaxNumTrans and del_trans[i] == 1:
                del a[(user_index, poi_index_from, poi_index_to)]
            user_index_prev = user_index
            i += 1

    # Make a count sum matrix --> count_sum
    for (user_index, poi_index_from, poi_index_to), counts in sorted(a.items()):
        count_sum[user_index, poi_index_from] += counts

    # Make a transition probability tensor --> b
    for (user_index, poi_index_from, poi_index_to), counts in sorted(a.items()):
        b[(user_index, poi_index_from, poi_index_to)] = counts / count_sum[user_index, poi_index_from]

    return a, b

######################## Make a training visit tensor #########################
# [input1]: st_user_index -- Start user index
# [input2]: user_num -- Number of users
# [input3]: poi_num -- Number of POIs
# [input4]: train_trace_list ([user_id, poi_id, unixtime, dow, hour, min])
# [output1]: a ({(user_index, poi_index_from, time_slot): counts})
# [output2]: b ({(user_index, poi_index_from, time_slot): probability})
def MakeTrainVisitTensor(st_user_index, user_num, poi_num, train_trace_list):
    # Initialization
    a = {}
    b = {}
    visit_num = [0] * user_num
    count_sum = np.zeros((user_num, T))

    # Make a visit count tensor --> a
    for event in train_trace_list:
        user_index = event[0] - st_user_index
        poi_index_from = event[1]
        ho = event[4]
        if TimeType == 1:
            if event[5] >= 40:
                mi = 2
            elif event[5] >= 20:
                mi = 1
            else:
                mi = 0
            time_slot = 3 * (ho - 9) + mi
        elif TimeType == 2:
            time_slot = int(ho/2)

        else:
            print("Wrong TimeType.\n")
            sys.exit(-1)
        a[(user_index, poi_index_from, time_slot)] = a.get((user_index, poi_index_from, time_slot), 0) + 1

    # Randomly delete counts for users whose number of visits exceed MaxNumVisit --> a
    if MaxNumVisit != -1:
        # Calculate the number of transitions for each user --> visit_num
        for (user_index, poi_index_from, time_slot), counts in sorted(a.items()):
            visit_num[user_index] += 1
#        print("Max of visit_num:", max(visit_num))
        # Randomly delete counts for users whose number of visits exceed MaxNumVisit
        user_index_prev = -1
        i = 0
        for (user_index, poi_index_from, time_slot), counts in sorted(a.items()):
            if user_index != user_index_prev:
                i = 0
                if visit_num[user_index] > MaxNumVisit:
                    rand_index = np.arange(visit_num[user_index])
                    np.random.shuffle(rand_index)
                    del_visit = np.zeros(visit_num[user_index])
                    for j in range(visit_num[user_index] - MaxNumVisit):
                        del_visit[rand_index[j]] = 1
#                    print("Deleted visits:", user_index + st_user_index, del_visit)
            if visit_num[user_index] > MaxNumVisit and del_visit[i] == 1:
                del a[(user_index, poi_index_from, time_slot)]
            user_index_prev = user_index
            i += 1

    # Make a count sum matrix --> count_sum
    for (user_index, poi_index_from, time_slot), counts in sorted(a.items()):
#        print(user_index, poi_index_from, time_slot, counts)
        count_sum[user_index, time_slot] += counts

    # Make a transition probability tensor --> b
    for (user_index, poi_index_from, time_slot), counts in sorted(a.items()):
        b[(user_index, poi_index_from, time_slot)] = counts / count_sum[user_index, time_slot]

    return a, b

#################################### Main #####################################
# Fix a seed
np.random.seed(1)

# Replace XX with City
TUserIndexFile = TUserIndexFile.replace("XX", City)
EUserIndexFile = EUserIndexFile.replace("XX", City)
POIIndexFile = POIIndexFile.replace("XX", City)
TrainTraceFile = TrainTraceFile.replace("XX", City)
TrainTransTensorFile = TrainTransTensorFile.replace("XX", City)
TrainVisitTensorFile = TrainVisitTensorFile.replace("XX", City)

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

# Read training traces
train_trace_list = ReadTrainTrace(TrainTraceFile)

# Make a training transition tensor
a, b = MakeTrainTransTensor(0, N, M, train_trace_list)

# Output a training transition tensor
f = open(TrainTransTensorFile, "w")
print("user_index,poi_index_from,poi_index_to,count,prob", file=f)
writer = csv.writer(f, lineterminator="\n")
for (user_index, poi_index_from, poi_index_to), counts in sorted(a.items()):
    s = [user_index, poi_index_from, poi_index_to, counts, b[(user_index, poi_index_from, poi_index_to)]]
    writer.writerow(s)
f.close()

# Make a training visit tensor
a2, b2 = MakeTrainVisitTensor(0, N, M, train_trace_list)

# Output a training visit tensor
f = open(TrainVisitTensorFile, "w")
print("user_index,poi_index_from,time_slot,count,prob", file=f)
writer = csv.writer(f, lineterminator="\n")
for (user_index, poi_index_from, time_slot), counts in sorted(a2.items()):
    s = [user_index, poi_index_from, time_slot, counts, b2[(user_index, poi_index_from, time_slot)]]
    writer.writerow(s)
f.close()
