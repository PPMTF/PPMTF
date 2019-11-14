#!/usr/bin/env python3
import math
import numpy as np
import csv
import sys

################################# Parameters ##################################
#sys.argv = ["MakeTrainTestData_PF.py", "TK"]

if len(sys.argv) < 2:
    print("Usage:",sys.argv[0],"[City]")
    sys.exit(0)

# City
City = sys.argv[1]

# POI file (input)
POIFile = "../data/PF/POI_" + City + ".csv"
# Trace file (input)
TraceFile = "../data/PF/traces_" + City + ".csv"
# Training user index file (output)
TUserIndexFile = "../data/PF/tuserindex_" + City + ".csv"
# Testing user index file (output)
EUserIndexFile = "../data/PF/euserindex_" + City + ".csv"
# POI index file (output)
POIIndexFile = "../data/PF/POIindex_" + City + ".csv"
# Training trace file (output)
TrainTraceFile = "../data/PF/traintraces_" + City + ".csv"
# Testing trace file (output)
TestTraceFile = "../data/PF/testtraces_" + City + ".csv"

# Minimum number of locations per user
MinNumLoc = 30
# Maximum number of locations per user (-1: N/A)
MaxNumLoc = 30

# Type of location (0: POI, 1: region)
TypeLoc = 1
# Minimum number of users per location (TypeLoc = 0)
MinNumUser = 5
# Threshold of the Euclidean distance between a user's location and POI (km) (TypeLoc = 0)
ThrDis = 0.1
# Number of regions on the x-axis (TypeLoc = 1)
#NumRegX = 10
#NumRegX = 15
NumRegX = 20
# Number of regions on the y-axis (TypeLoc = 1)
#NumRegY = 5
#NumRegY = 10
#NumRegY = 15
NumRegY = 20
#NumRegY = 25
#NumRegY = 30
#NumRegY = 35
#NumRegY = 40
#NumRegY = 45
#NumRegY = 50

# Month & day
#MonthDay = {(7,1)}
#MonthDay = {(7,7)}
#MonthDay = {(10,7)}
#MonthDay = {(10,13)}
#MonthDay = {(12,16)}
#MonthDay = {(12,22)}
MonthDay = {(7,1), (7,7), (10,7), (10,13), (12,16), (12,22)}
# Time interval (min)
TimeInt = 20
# Start time (hour)
StartTime = 9
# End time (hour)
EndTime = 19

# Number of training users
TUserNum = 500
#Number of testing users
EUserNum = 500

########################### Read POI & trace files ############################
# [output1]: poi_dic ({poi_id: [poi_index, y, x, category, y_id, x_id, 2D_id]})
# [output2]: user_dic ({user_id: user_index})
# [output3]: ucount_dic ({user_id: counts})
# [output4]: trace_list ([user_id, poi_id (or 2D_id), unixtime, dow, hour, min, distance])
def ReadPOITrace():
    # Initialization
    poi_dic = {}
    ucount_dic= {}
    trace_list = []
    thrdis2 = ThrDis**2

    # Read a POI file --> poi_dic
    if TypeLoc == 0:
        f = open(POIFile, "r")
        reader = csv.reader(f)
        next(reader)
        for lst in reader:
            # Continue if the number of users for the location is below MinNumUser
            if int(lst[8]) < MinNumUser:
                continue
            poi_dic[lst[0]] = [len(poi_dic), float(lst[1]), float(lst[2]), lst[3], lst[4], lst[5], lst[6]]
        f.close()
        print("#POIs within the area of interest =", len(poi_dic))

    # Read a trace file --> ucount_dic, trace_list
    f = open(TraceFile, "r")
    reader = csv.reader(f)
    next(reader)
    for i, lst in enumerate(reader):
#        if i % 100000 == 0:
#            print(i)
        user_id = int(lst[0])
        y = float(lst[1])
        x = float(lst[2])
        two_dim_id = int(lst[5])
        ut = float(lst[6])
        mo = int(lst[8])
        da = int(lst[9])
        ho = int(lst[10])
        mi = int(lst[11])
        if (mo == 7 and da == 1) or (mo == 10 and da == 7) or (mo == 12 and da == 16):
            dow = "Mon"
        else:
            dow = "Sun"
            
        # Continue if time is out of scope
        if (mo,da) not in MonthDay or ho < StartTime or ho >= EndTime or mi % TimeInt != 0:
            continue

        # Continue if the number of locations reaches MaxNumLoc
        if MaxNumLoc != -1 and ucount_dic.get(user_id, 0) == MaxNumLoc:
            continue
        # If the type of location is POI
        if TypeLoc == 0:
            # Search a nearest-neighbor POI within MinDis km
            mindis2 = thrdis2
            minkey = 0
            for key in poi_dic:
                # 1 degree of latitude (resp. longitude in TK) = 111 km (resp. 91 km)
                if (111 * (y - poi_dic[key][1]))**2 + (91 * (x - poi_dic[key][2]))**2 < mindis2:
                    mindis2 = (111 * (y - poi_dic[key][1]))**2 + (91 * (x - poi_dic[key][2]))**2
                    minkey = key
            if mindis2 != thrdis2:
                if (user_id not in ucount_dic and ho == StartTime and mi == 0) or (user_id in ucount_dic and (ho - trace_list[-1][4])*60 + (mi - trace_list[-1][5]) == TimeInt):
                    ucount_dic[user_id] = ucount_dic.get(user_id, 0) + 1
                    trace_list.append([user_id, minkey, ut, dow, ho, mi, math.sqrt(mindis2)])
        # If the type of location is region
        elif TypeLoc == 1:
#            if (ho == StartTime and mi == 0) or (user_id in ucount_dic and (ho - trace_list[-1][4])*60 + (mi - trace_list[-1][5]) == TimeInt):
            if (user_id not in ucount_dic and ho == StartTime and mi == 0) or (user_id in ucount_dic and (ho - trace_list[-1][4])*60 + (mi - trace_list[-1][5]) == TimeInt):
                ucount_dic[user_id] = ucount_dic.get(user_id, 0) + 1
                trace_list.append([user_id, two_dim_id, ut, dow, ho, mi, 0])
        else:
           print("Wrong TypeLoc.")
           sys.exit(-1)
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

    return poi_dic, user_dic, ucount_dic, trace_list

#################################### Main #####################################
# Fix a seed
np.random.seed(1)

# Read POI & trace files
poi_dic, user_dic, ucount_dic, trace_list = ReadPOITrace()

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
writer = csv.writer(f, lineterminator="\n")
if TypeLoc == 0:
    for poi_id in poi_dic:
        lst = [poi_id, poi_dic[poi_id][0], poi_dic[poi_id][3], poi_dic[poi_id][4]]
        writer.writerow(lst)
elif TypeLoc == 1:
    for y_id in range(NumRegY):
        for x_id in range(NumRegX):
            two_dim_id = y_id * NumRegX + x_id
            lst = [two_dim_id, two_dim_id, "-", "-"]
            writer.writerow(lst)
f.close()

# Output training traces
f = open(TrainTraceFile, "w")
print("user_index,poi_index,category,unixtime,dow,hour,min,distance", file=f)
writer = csv.writer(f, lineterminator="\n")
for event in trace_list:
    if event[0] in user_dic and tuser_list[user_dic[event[0]]] == 1:
        if TypeLoc == 0:
            lst = [tuser_dic[event[0]],poi_dic[event[1]][0],poi_dic[event[1]][3],event[2],event[3],event[4],event[5],event[6]]
        elif TypeLoc == 1:
            lst = [tuser_dic[event[0]],event[1],"-",event[2],event[3],event[4],event[5],event[6]]
        writer.writerow(lst)
f.close()

# Output testing traces
f = open(TestTraceFile, "w")
print("user_index,poi_index,category,unixtime,dow,hour,min,distance", file=f)
writer = csv.writer(f, lineterminator="\n")
for event in trace_list:
    if event[0] in user_dic and euser_list[user_dic[event[0]]] == 1:
        if TypeLoc == 0:
            lst = [euser_dic[event[0]],poi_dic[event[1]][0],poi_dic[event[1]][3],event[2],event[3],event[4],event[5],event[6]]
        elif TypeLoc == 1:
            lst = [euser_dic[event[0]],event[1],"-",event[2],event[3],event[4],event[5],event[6]]
        writer.writerow(lst)
f.close()

