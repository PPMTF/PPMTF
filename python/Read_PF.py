#!/usr/bin/env python3
from datetime import datetime
import csv
import numpy as np
import sys

################################# Parameters ##################################
#sys.argv = ["Read_PF.py", "../20140808_snsbasedPeopleFlowData_nightley/", "TK"]

if len(sys.argv) < 3:
    print("Usage:",sys.argv[0],"[PFDir (in)] [City]")
    sys.exit(0)

# People flow dir (input)
PFDir = sys.argv[1]
# City
City = sys.argv[2]

# People flow files (input)
PFFiles = ["2013-07-01.csv", "2013-07-07.csv", "2013-10-07.csv", "2013-10-13.csv", "2013-12-16.csv", "2013-12-22.csv"]
# POI file (output)
POIFile = "../data/PF/POI_" + City + ".csv"
# Trace file (output)
TraceFile = "../data/PF/traces_" + City + ".csv"
# Minimum of y (latitude)
MIN_Y = 35.65
# Maximum of y (latitude)
MAX_Y = 35.75
# Minimum of x (longitude)
MIN_X = 139.68
# Maximum of x (longitude)
MAX_X = 139.8
# Type of location (0: POI, 1: region)
TypeLoc = 1
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

########################### Read People flow files ############################
# [output1]: poi_dic ({(y, x): [category, sub-category, category group, y_id, x_id, 2D_id, counts]})
# [output2]: checkin_list ([user_id, y, x, y_id, x_id, 2D_id, unixtime, year, month, day, hour, min, sec])
def ReadPF():
    # Initialization
    poi_dic = {}
    checkin_list = []
    max_x = -180
    min_x = 180
    max_y = -180
    min_y = 180

    # Calculate the boundaries of the regions (NumRegX x NumRegY) --> xb, yb
    xb = np.zeros(NumRegX)
    yb = np.zeros(NumRegY)
    for i in range(NumRegX):
        xb[i] = MIN_X + (MAX_X - MIN_X) * i / NumRegX
    for i in range(NumRegY):
        yb[i] = MIN_Y + (MAX_Y - MIN_Y) * i / NumRegY
        
    # Read a checkin file --> checkin_list
    for file in PFFiles:
        pffile = PFDir + file
        f = open(pffile, "r")
        reader = csv.reader(f)
        next(reader)
        for lst in reader:
            user_id = int(lst[0])
            y = float(lst[3])
            x = float(lst[4])
            cat = lst[5]
            subcat = lst[6]
            catgroup = lst[8]
            
            if max_x < x:
                max_x = x
            if max_y < y:
                max_y = y
            if min_x > x:
                min_x = x
            if min_y > y:
                min_y = y
            if MIN_Y <= y <= MAX_Y and MIN_X <= x <= MAX_X:
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

                # Calculate the two-dimansional ID --> two_dim_id
                two_dim_id = y_id * NumRegX + x_id

                # Update poi_dic
                if TypeLoc == 0 and cat != "":
                    if (y,x) not in poi_dic:
                        poi_dic[(y,x)] = [cat, subcat, catgroup, y_id, x_id, two_dim_id, 1]
                    else:
                        poi_dic[(y,x)][7] += 1

                # datetime --> dt
                daytim = lst[2]
                daytim_list = daytim.split(" ")
                day_list = daytim_list[0].split("-")
                tim_list = daytim_list[1].split(":")
                # year --> ye
                ye = int(day_list[0])
                # month --> mo
                mo = int(day_list[1])
                # day --> da
                da = int(day_list[2])
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
                # Update checkin_list
                checkin_list.append([user_id, y, x, y_id, x_id, two_dim_id, ut, ye, mo, da, ho, mi, se])
        f.close()
    print("max_x:", max_x, "min_x:", min_x, "max_y:", max_y, "min_y:", min_y)

    print("#POIs within the area of interest =", len(poi_dic))
    print("#Checkins within the area of interest =", len(checkin_list))

    return poi_dic, checkin_list

#################################### Main #####################################
# Read People flow files
poi_dic, checkin_list = ReadPF()

# Output poi_dic
f = open(POIFile, "w")
print("poi_id,y,x,category,y_id,x_id,2D_id,count", file=f)
writer = csv.writer(f, lineterminator="\n")
if TypeLoc == 0:
    for i, key in enumerate(poi_dic):
        lst = [i, key[0], key[1], poi_dic[key][0]+"/"+poi_dic[key][1]+"/"+poi_dic[key][2], poi_dic[key][3], poi_dic[key][4], poi_dic[key][5], poi_dic[key][6], poi_dic[key][7]]
        writer.writerow(lst)    
elif TypeLoc == 1:
    # Calculate the center of each region (NumRegX x NumRegY) --> xc, yc
    xc = np.zeros(NumRegX)
    yc = np.zeros(NumRegY)
    x_width = (MAX_X - MIN_X) / NumRegX
    y_width = (MAX_Y - MIN_Y) / NumRegY

    for i in range(NumRegX):
        xc[i] = MIN_X + x_width * i + x_width / 2
    for i in range(NumRegY):
        yc[i] = MIN_Y + y_width * i + y_width / 2
    for y_id in range(NumRegY):
        for x_id in range(NumRegX):
            two_dim_id = y_id * NumRegX + x_id
            lst = [two_dim_id, yc[y_id], xc[x_id], "-", y_id, x_id, two_dim_id, "-"]
            writer.writerow(lst)
f.close()

# Sort checkin_list in ascending order of (user_id, unixtime)
checkin_list.sort(key=lambda tup: (tup[0], tup[7]), reverse=False)

# Output checkin_list
f = open(TraceFile, "w")
print("user_id,y,x,y_id,x_id,2D_id,unixtime,year,month,day,hour,min,sec", file=f)
writer = csv.writer(f, lineterminator="\n")
for lst in checkin_list:
    writer.writerow(lst)
f.close()
