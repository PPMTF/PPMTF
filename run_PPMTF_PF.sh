#!/bin/bash -x

cd data/PF/
tar -zxvf traces_TK.tar.gz
cd ../../python/
python3.6 MakeTrainTestData_PF.py TK
python3.6 MakeTrainTensor.py PF TK
cd ../cpp/
./PPMTF PF TK 200
./SynData_PPMTF PF TK 10 200
cd ../python/
python3.6 EvalUtilPriv.py PF TK PPMTF 10
