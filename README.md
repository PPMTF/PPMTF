# PPMTF
This is a source code of PPMTF (Privacy-Preserving Multiple Tensor Factorization): https://arxiv.org/abs/1911.04226.

PPMTF is implemented with C++ (data preprocessing and evaluation are implemented with Python). This software is released under the MIT License.

# Directory Structure
- cpp/			&emsp;C++ codes (put the required files under this directory; see cpp/README.md).
- data/			&emsp;Output data (obtained by running codes).
  - PF/			&emsp;Output data in PF.
  - FS/			&emsp;Output data in FS.
- python/		&emsp;Python codes.
- results/		&emsp;Experimental results.
  - PF/			&emsp;Experimental results in PF.
  - FS/			&emsp;Experimental results in FS.
- LICENSE.txt		&emsp;MIT license.
- NOTICE.txt		&emsp;Copyright information.
- OtherSynthesizers.md	&emsp;Usage of other synthesizers.
- README.md		&emsp;This file.

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
python3 Read_PF.py [PF directory (including the people flow data files in Tokyo)] TK
python3 MakeTrainTestData_PF.py TK
python3 MakeTrainTensor.py PF TK
cd ../cpp/
./PPMTF PF TK 200
(To change the alpha paramter from 200 to [alpha], run "./PPMTF PF TK [alpha]".)
./SynData_PPMTF PF TK 10
```

Then synthesize traces (syntraces_Itr100.csv) in PF will be generated in data/PF/PPMTF_TK_alp200_mnt100_mnv100/.

To evaluate the utility and privacy of the synthetic traces, run the following command.

```bash
python3 EvalUtilPriv.py PF TK PPMTF 10
```

Then the results (utilpriv_PPMTF_TK.csv) will be stored in data/PF/.

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

Then synthesize traces (syntraces_Itr100.csv) in NYC will be generated in data/FS/PPMTF_NY_alp200_mnt100_mnv100/.

To evaluate the utility and privacy of the synthetic traces, run the following command.

```bash
python3 EvalUtilPriv.py FS NY PPMTF 1
```

Then the results (utilpriv_PPMTF_NY.csv) will be stored in data/FS/.

Synthesized traces in other cities (IST/JK/KL/SP/TKY) can also be generated and evaluated by replacing NY with IS, JK, KL, SP, or TK.

**(4) Experimental Results**

The experimental results after running (1)-(3) can be found in results/.
(To obtain the experimental results for other synthesizers, see OtherSynthesizers.md.)

# Execution Environment
We used CentOS 7.5 with gcc 4.8.5 and python 3.6.5.

# External Libraries used by PPMTF
- [Eigen 3.3.7](http://eigen.tuxfamily.org/index.php?title=Main_Page) is distributed under the [MPL2](https://www.mozilla.org/en-US/MPL/2.0/).
- [Generalized Constant Expression Math](https://www.kthohr.com/gcem.html) is distributed under the [Apache License 2.0](https://github.com/kthohr/stats/blob/master/LICENSE).
- [StatsLib](https://www.kthohr.com/statslib.html) is distributed under the [Apache License 2.0](https://github.com/kthohr/stats/blob/master/LICENSE).
