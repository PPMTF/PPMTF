# PPMTF
This is a source code of PPMTF (Privacy-Preserving Multiple Tensor Factorization) in the following paper: 

Takao Murakami, Koki Hamada, Yusuke Kawamoto, Takuma Hatano, "Privacy-Preserving Multiple Tensor Factorization for Synthesizing Large-Scale Location Traces with Cluster-Specific Features," Proceedings on Privacy Enhancing Technologies (PoPETs), Issue 2, 2021 (to appear).

Full paper: https://arxiv.org/abs/1911.04226

PPMTF is implemented with C++ (data preprocessing and evaluation are implemented with Python). This software is released under the MIT License.

# Purpose

The purpose of this source code is to reproduce experimental results of PPMTF in PF (SNS-based people flow data) and FS (Foursquare dataset). In particular, we designed our code to easily reproduce experimental results of PPMTF (alpha=200) in PF (Figure 7 "PPMTF" in our paper) using Docker files. See **Running Our Code Using Dockerfiles** for details. 

We also designed our code to reproduce experimental results of PPMTF in FS (Figure 10 "PPMTF" in our paper) by downloading the Foursquare dataset and running our code. Note that it takes a lot of time (e.g., it may take more than one day depending on the running environment) to run our code. See **Usage (4)(5)** for details.

# Directory Structure
- cpp/			&emsp;C++ codes (put the required files under this directory; see cpp/README.md).
- data/			&emsp;Output data (obtained by running codes).
  - PF/			&emsp;Output data in PF (SNS-based people flow data).
  - PF_dataset/		&emsp;Place PF (SNS-based people flow data) in this directory (currently empty).
  - FS/			&emsp;Output data in FS (Foursquare dataset).
  - FS_dataset/		&emsp;Place FS (Foursquare dataset) in this directory (currently empty).
- python/		&emsp;Python codes.
- results/		&emsp;Experimental results.
  - PF/			&emsp;Experimental results in PF (SNS-based people flow data).
  - FS/			&emsp;Experimental results in FS (Foursquare dataset).
- docker-compose.yml		&emsp;docker-compose.yml file.
- Dockerfile		&emsp;Dockerfile.
- LICENSE.txt		&emsp;MIT license.
- NOTICE.txt		&emsp;Copyright information.
- OtherSynthesizers.md	&emsp;Usage of other synthesizers.
- README.md		&emsp;This file.

# Running Our Code Using Docker Files

You can easily build and run our code using Docker files as follows.

1. Install [docker](https://docs.docker.com/get-docker/) & [docker-compose](https://docs.docker.jp/compose/install.html).

2. Clone this repository.
```
$ git clone https://github.com/PPMTF/PPMTF
```

3. Build a docker image.
```
$ cd PPMTF
$ docker-compose up -d --build
```

4. Attach to the docker container.
```
$ docker-compose exec ppmtf bash
```

5. Run our code (NOTE: it may take one hour or so depending on the running environment).
```
$ cd opt/PPMTF/
$ chmod +x run_PPMTF_PF.sh
$ ./run_PPMTF_PF.sh
```

Then experimental results of PPMTF (alpha=200) in PF will be output in "data/PF/utilpriv_PPMTF_TK.csv".

We plotted Figure 7 "PPMTF" in our paper using this file, while changing the alpha parameter from 0.5 to 1000. To see the figure, see "res/PF/utilpriv.xlsx". To change the alpha parameter, see **Usage (3)**.

# Usage

Below we describe how to build and run our code in details. Note that (1) and (3) can be done using Docker files (see **Running Our Code Using Dockerfiles** for details).

**(1) Install**

Install Eigen 3.3.7, Generalized Constant Expression Math, and StatsLib (see cpp/README.md).

Install PPMTF (C++) as follows.
```
$ cd cpp/
$ make
$ cd ../
```

**(2) Download and preprocess PF**

Download the [SNS-based people flow data](https://nightley.jp/archives/1954/) and place the dataset in data/PF_dataset/.

Run the following commands.

```
$ cd python/
$ python3 Read_PF.py data/PF_dataset TK
$ cd ../
```

Then the POI file (POI_TK.csv) and the trace file (traces_TK.csv) are output in data/PF/.

**(3) Synthesizing traces in PF using PPMTF**

Run the following commands.

```
$ cd python/
$ python3 MakeTrainTestData_PF.py TK
$ python3 MakeTrainTensor.py PF TK
$ cd ../cpp/
$ ./PPMTF PF TK 200
(To change the alpha paramter from 200 to [alpha], run "./PPMTF PF TK [alpha]".)
$ ./SynData_PPMTF PF TK 10 200
(To change the alpha paramter from 200 to [alpha], run "./SynData_PPMTF PF TK 10 [alpha]".)
$ cd ../
```

Then synthesize traces (syntraces_Itr100.csv) in PF will be generated in data/PF/PPMTF_TK_alp200_mnt100_mnv100/.

To evaluate the utility and privacy of the synthetic traces, run the following command.

```
$ cd python/
$ python3 EvalUtilPriv.py PF TK PPMTF 10
$ cd ../
```

Then experimental results of PPMTF (utilpriv_PPMTF_TK.csv) will be output in data/PF/.

We plotted Figure 7 "PPMTF" in our paper using this file, while changing the alpha parameter from 0.5 to 1000. See "res/PF/utilpriv.xlsx" for details.

**(4) Download and preprocess FS**

Download the [Foursquare dataset (Global-scale Check-in Dataset with User Social Networks)](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) and place the dataset in data/FS_dataset/.

Run the following command to fix garbled text.

```
$ cd data/FS_dataset/
$ cat raw_POIs.txt | sed -e "s/Caf[^.]*\t/Caf\t/" > raw_POIs_fix.txt
$ cd ../
```

Run the following commands.

```
$ cd python/
$ python3 Read_FS.py data/FS_dataset/ NY
$ cd ../
```

Then the POI file (POI_NY.csv) and the trace file (traces_NY.csv) are output in data/PF/.

The POI file and trace file in other cities (IST/JK/KL/SP/TKY) can also be generated by replacing NY with IS, JK, KL, SP, or TK.

**(5) Synthesizing traces in FS using PPMTF**

Run the following commands.

```
$ cd python/
$ python3 MakeTrainTestData_FS.py NY
$ python3 MakeTrainTensor.py PF NY
$ cd ../cpp/
$ ./PPMTF FS NY
$ ./SynData_PPMTF FS NY 1
$ ./PDTest_Trace FS NY 1.0 10 32000 1
```

Then synthesize traces (syntraces_Itr100.csv) in NYC will be generated in data/FS/PPMTF_NY_alp200_mnt100_mnv100/.

To evaluate the utility and privacy of the synthetic traces, run the following command.

```
$ python3 EvalUtilPriv.py FS NY PPMTF 1
```

Then experimental results of PPMTF (utilpriv_PPMTF_NY.csv) will be output in data/FS/.

Synthesized traces in other cities (IST/JK/KL/SP/TKY) can also be generated and evaluated by replacing NY with IS, JK, KL, SP, or TK.

We plotted Figure 10 "PPMTF" in our paper using these files. See "res/FS/utilpriv.xlsx" for details.

**(6) Experimental Results for Other Synthesizers**

To obtain experimental results for other synthesizers, see OtherSynthesizers.md.

# Execution Environment
We used CentOS 7.5 with gcc 4.8.5 and python 3.6.5.

# External Libraries used by PPMTF
- [Eigen 3.3.7](http://eigen.tuxfamily.org/index.php?title=Main_Page) is distributed under the [MPL2](https://www.mozilla.org/en-US/MPL/2.0/).
- [Generalized Constant Expression Math](https://www.kthohr.com/gcem.html) is distributed under the [Apache License 2.0](https://github.com/kthohr/stats/blob/master/LICENSE).
- [StatsLib](https://www.kthohr.com/statslib.html) is distributed under the [Apache License 2.0](https://github.com/kthohr/stats/blob/master/LICENSE).
