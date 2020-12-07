#!/bin/bash -x

cd python/
python3 MakeTrainTensor.py PF TK
cd ../cpp/
./PPMTF PF TK 200
./SynData_PPMTF PF TK 10 200
cd ../python/
python3 EvalUtilPriv.py PF TK PPMTF 10
