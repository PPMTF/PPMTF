# PPMTF
This is a source code of PPMTF (Privacy-Preserving Multiple Tensor Factorization): https://arxiv.org/abs/1911.04226.

PPMTF is implemented with C++ (data preprocessing and evaluation are implemented with Python). This software is released under the MIT License.

# Directory Structure
- cpp/
  - cpp codes (put the required files under this directory; see README.txt).
- data/
  - Data of PF/FS (output by running codes).
- python/
  - python codes.

# Usage
**(1) Install**

Install Eigen 3.3.7, Generalized Constant Expression Math, and StatsLib (see cpp/README.txt).

Install PPMTF (C++) as follows.
```bash
cd cpp/
make
```

**(2) Synthesizing traces in PF (SNS-based people flow data) using PPMTF**

Download the [SNS-based people flow data](https://nightley.jp/archives/1954/).

Run the following commands.

```bash
cd python/
python Read_PF.py [PF directory (including the people flow data files in Tokyo)] TK
python MakeTrainTestData_PF.py TK
python MakeTrainTensor.py PF TK
cd ../cpp/
./PPMTF PF TK 200
(To change the alpha paramter from 200 to [alpha], run "./PPMTF PF TK [alpha]".)
./SynData_PPMTF PF TK 10
```

Then synthesize traces in PF by PPMTF are generated under data/PF/.

To evaluate the utility and privacy of the synthesized traces, run the following command.

```bash
./EvalUtilPriv.py PF TK PPMTF
```

Then the results are output under data/PF/.

**(3) Synthesizing traces in FS (Foursquare dataset) using PPMTF**

Download the [Foursquare dataset (Global-scale Check-in Dataset with User Social Networks)](https://sites.google.com/site/yangdingqi/home/foursquare-dataset).

Under the FS directory (including the Foursquare data files), run the following command (to fix garbled text).

```bash
cat raw_POIs.txt | sed -e "s/Caf[^.]*\t/Caf\t/" > raw_POIs_fix.txt
```

Run the following commands.

```bash
cd python/
python3 Read_FS.py [FS directory (including the Foursquare data files)] NY
python3 MakeTrainTestData_FS.py NY
python3 MakeTrainTensor.py PF NY
cd ../cpp/
./PPMTF FS NY
./SynData_PPMTF FS NY 1
./PDTest_Trace FS NY 1.0 10 32000 1
```

Then synthesize traces in NYC (New York City) by PPMTF are generated under data/FS/.

To evaluate the utility and privacy of the synthesized traces, run the following command.

```bash
./EvalUtilPriv.py FS NY PPMTF
```

Then the results are output under data/FS/.

Synthesized traces in other cities (IST/JK/KL/SP/TKY) can also be generated and evaluated by replacing NY with IS, JK, KL, SP, or TK.

**(4) Synthesizing traces in PF/FS using SGD**

Run the following commands.

```bash
./SynData_SGD PF TK 10 0
(To change the number of copied events from 0 to [CopyNum], run "./SynData_SGD PF TK 10 [CopyNum]".)
./SynData_SGD FS NY 1
```

Then synthesize traces in PF and NYC by SGD are generated under data/PF and data/FS/, respectively.

To evaluate the utility and privacy of the synthesized traces, run the following command.

```bash
./EvalUtilPriv.py PF TK SGD
./EvalUtilPriv.py FS NY SGD
```

Then the results are output under data/PF/ or data/FS/.

Synthesized traces in other cities (IST/JK/KL/SP/TKY) can also be generated and evaluated by replacing NY with IS, JK, KL, SP, or TK.

# Execution Environment
We used CentOS 7.5 with gcc 4.8.5 and python 3.6.5.

# External Libraries used by PPMTF
- [Eigen 3.3.7](http://eigen.tuxfamily.org/index.php?title=Main_Page) is distributed under the [MPL2](https://www.mozilla.org/en-US/MPL/2.0/).
- [Generalized Constant Expression Math](https://www.kthohr.com/gcem.html) is distributed under the [Apache License 2.0](https://github.com/kthohr/stats/blob/master/LICENSE).
- [StatsLib](https://www.kthohr.com/statslib.html) is distributed under the [Apache License 2.0](https://github.com/kthohr/stats/blob/master/LICENSE).
