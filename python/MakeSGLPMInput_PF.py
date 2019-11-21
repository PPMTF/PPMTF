#!/usr/bin/env python3
import numpy as np
import csv
import os

################################# Parameters ##################################
# Input directory
InputDir = "../data/PF/"
# Output directory
OutputDir = "../data/PF/SGLT_TK/"

# Training trace file (input)
TrainTraceFile = InputDir + "traintraces_TK.csv"
# sg-LPM trace file (output)
SglpmTraceFile = OutputDir + "input.trace"
# sg-LPM mobility file (output)
SglpmMobFile = OutputDir + "input.mobility"
# sg-LPM location file (output)
SglpmLocFile = OutputDir + "locations"

# Minimum of y (latitude)
MIN_Y = 35.65
# Maximum of y (latitude)
MAX_Y = 35.75
# Minimum of x (longitude)
MIN_X = 139.68
# Maximum of x (longitude)
MAX_X = 139.8

# Number of regions on the x-axis
#NumRegX = 10
#NumRegX = 15
NumRegX = 20
# Number of regions on the y-axis
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


############################ Read training traces #############################
# [input1]: tracefile
# [output1]: train_trace_list ([user_index, time_index, poi_index])
def ReadTrainTrace(tracefile):
    # Initialization
    train_trace_list = []

    # Read training traces
    f = open(tracefile, "r")
    reader = csv.reader(f)
    next(reader)
    time_index = 0
    user_index_pre = -1
    for event in reader:
        user_index = int(event[0])
        poi_index = int(event[1])
        if user_index != user_index_pre:
            time_index = 0
        train_trace_list.append([user_index, time_index, poi_index])
        user_index_pre = user_index
        time_index += 1
    f.close()
    
    return train_trace_list

#################################### Main #####################################
# Make OutputDir
if not os.path.exists(OutputDir):
    os.mkdir(OutputDir)

# Number of POIs --> M
M = NumRegX * NumRegY

# Read training traces
train_trace_list = ReadTrainTrace(TrainTraceFile)

# Output sg-LPM trace data
f = open(SglpmTraceFile, "w")
writer = csv.writer(f, lineterminator="\n")
for (user_index, time_index, poi_index) in train_trace_list:
    s = [user_index+1, time_index+1,poi_index+1]
    writer.writerow(s)
f.close()

# Output sg-LPM mobility information
out_line = "1, " * (M - 1) + "1"
f = open(SglpmMobFile, "w")
for i in range(M):
    print(out_line, file=f)
f.close()

# Calculate the boundaries of the regions (NumRegX x NumRegY) [km] --> xb, yb
# 1 degree of latitude (resp. longitude in TK) = 111 km (resp. 91 km)
yb = np.zeros(NumRegY)
xb = np.zeros(NumRegX)
for i in range(NumRegY):
    yb[i] = ((MAX_Y - MIN_Y) * i / NumRegY) * 111
for i in range(NumRegX):
    xb[i] = ((MAX_X - MIN_X) * i / NumRegX) * 91

# Output sg-LPM location information
f = open(SglpmLocFile, "w")
for i in range(M):
    y_id = int(i / NumRegX)
    x_id = i % NumRegX
    out_line = '{:.4f}'.format(xb[x_id]) + ", " + '{:.4f}'.format(yb[y_id])
    print(out_line, file=f)
f.close()
