#!/usr/bin/env python3
from datetime import datetime
import csv
import sys

################################# Parameters ##################################
#sys.argv = ["Read_FS.py", "../dataset_WWW2019/", "IS"]
#sys.argv = ["Read_FS.py", "../dataset_WWW2019/", "NY"]
#sys.argv = ["Read_FS.py", "../dataset_WWW2019/", "TK"]
#sys.argv = ["Read_FS.py", "../dataset_WWW2019/", "JK"]
#sys.argv = ["Read_FS.py", "../dataset_WWW2019/", "KL"]
#sys.argv = ["Read_FS.py", "../dataset_WWW2019/", "SP"]

if len(sys.argv) < 3:
    print("Usage:",sys.argv[0],"[FSDir (in)] [City]")
    sys.exit(0)

# People flow dir (input)
PFDir = sys.argv[1]
# City
City = sys.argv[2]

# Checkin file (input)
CheckinFile = PFDir + "raw_Checkins_anonymized.txt"
# Original POI file (input)
OrgPOIFile = PFDir + "raw_POIs_fix.txt"
# POI file (output)
POIFile = "../data/FS/POI_" + City +".csv"
# Trace file (output)
TraceFile = "../data/FS/traces_" + City + ".csv"

# Minimum of y (latitude)
if City == "NY":
    MIN_Y = 40.5
elif City == "TK":
    MIN_Y = 35.5
elif City == "IS":
    MIN_Y = 40.8
elif City == "JK":
    MIN_Y = -6.4
elif City == "KL":
    MIN_Y = 3.0
elif City == "SP":
    MIN_Y = -24.0
else:
    print("Wrong City")

# Maximum of y (latitude)
if City == "NY":
    MAX_Y = 41.0
elif City == "TK":
    MAX_Y = 35.9
elif City == "IS":
    MAX_Y = 41.2
elif City == "JK":
    MAX_Y = -6.1
elif City == "KL":
    MAX_Y = 3.3
elif City == "SP":
    MAX_Y = -23.4
else:
    print("Wrong City")

# Minimum of x (longitude)
if City == "NY":
    MIN_X = -74.28
elif City == "TK":
    MIN_X = 139.5
elif City == "IS":
    MIN_X = 28.5
elif City == "JK":
    MIN_X = 106.7
elif City == "KL":
    MIN_X = 101.6
elif City == "SP":
    MIN_X = -46.8
else:
    print("Wrong City")

# Maximum of x (longitude)
if City == "NY":
    MAX_X = -73.68
elif City == "TK":
    MAX_X = 140.0
elif City == "IS":
    MAX_X = 29.5
elif City == "JK":
    MAX_X = 107.0
elif City == "KL":
    MAX_X = 101.8
elif City == "SP":
    MAX_X = -46.3
else:
    print("Wrong City")

# Country (none: no restriction for the country)
Country = "none"
#Country = "US"
#Country = "JP"
#Country = "TR"
#Country = "ID"
#Country = "MY"
#Country = "BR"

############################ Read Foursquare files ############################
# [output1]: poi_dic ({poi_id: [poi_id, y, x, category, counts]})
def ReadFS():
    month = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr":4, "May":5, "Jun":6, 
             "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
    # Initialization
    poi_dic = {}
#    cou_set = set()

    # Read an original POI file --> poi_dic ({poi_id: [poi_id, y, x, counts]})
    f = open(OrgPOIFile, "r")
    poi_num = 0
    for line in f:
        lst = line.split("\t")
        poi_id = lst[0]
        y = float(lst[1])
        x = float(lst[2])
        cat = lst[3]
        if cat == "Caf\n":
            cat = "Cafe"
        if len(lst) == 5:
            cou = lst[4].rstrip("\n")
        else:
            cou = ""
#        cou_set.add(cou)
        if (Country == "none" and MIN_Y <= y <= MAX_Y and MIN_X <= x <= MAX_X) or Country == cou:
            poi_dic[poi_id] = [poi_id, y, x, cat, 0]
            poi_num += 1
#        else:
#            print(poi_id, y, x, cat)
    f.close()
    print("#POIs within the area of interest =", poi_num)

    # Output checkin_list_smpl
    g = open(TraceFile, "w")
    print("user_id,poi_id,y,x,unixtime,year,month,day,dow,hour,min,sec", file=g)
    writer = csv.writer(g, lineterminator="\n")

    # Read a checkin file & output checkin data whose poi_id is in poi_dic
    f = open(CheckinFile, "r")
    checkin_num = 0
    for i, line in enumerate(f):
#    for line in f:
#        if i % 10000000 == 0:
#            print(i)
        lst = line.split("\t")
        user_id = int(lst[0])
#        user_id = lst[0]
        poi_id = lst[1]
        daytim = lst[2]
        if poi_id in poi_dic:
            daytim_list = daytim.split(" ")
            if(len(daytim_list) != 6):
#                print("Cannot read:", line)
                continue
            tim_list = daytim_list[3].split(":")
            if(len(tim_list) != 3):
#                print("Cannot read:", line)
                continue
            # day of week --> dow
            dow = daytim_list[0]
            # year --> ye
            ye = int(daytim_list[5])
            # month --> mo
            mo = month[daytim_list[1]]
            # day --> da
            da = int(daytim_list[2])
            # hour --> ho
            ho = int(tim_list[0])
            # min --> mi
            mi = int(tim_list[1])
            # sec --> se
            se = int(tim_list[2])
            # datetime --> dt
            dt = datetime(ye, mo, da, ho, mi, se)
            # unixtime --> ut
            ut = dt.timestamp()

            # checkin data --> lst ([user_id, poi_id, y, x, unixtime, year, month, day, dow, hour, min, sec])
            lst = [user_id, poi_id, poi_dic[poi_id][1], poi_dic[poi_id][2], ut, ye, mo, da, dow, ho, mi, se]
            # Output lst
            writer.writerow(lst)

            # Update counts
            poi_dic[poi_id][4] += 1
            checkin_num += 1
    f.close()
    g.close()

    print("#Checkins within the area of interest =", checkin_num)
    return poi_dic
#    return poi_dic, cou_set
        
#################################### Main #####################################
# Read Foursuqare files
poi_dic = ReadFS()

# Output poi_dic
f = open(POIFile, "w")
print("poi_id,y,x,category,count", file=f)
writer = csv.writer(f, lineterminator="\n")
for key in poi_dic:
    writer.writerow(poi_dic[key])
f.close()
