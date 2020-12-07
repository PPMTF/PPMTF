#!/bin/bash -x

cd python/
python3.6 MakeTrainTensor.py PF TK
cd ../cpp/
./PPMTF PF TK 200
./SynData_PPMTF PF TK 10 200
cd ../python/
python3.6 EvalUtilPriv.py PF TK PPMTF 10
