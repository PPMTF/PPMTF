# Usage

**(a) Synthesizing traces in PF/FS using SGD (Synthetic Data Generator)**

Run the following commands.

```bash
cd cpp/
./SynData_SGD PF TK 10 0
(To change the number of copied events from 0 to [CopyNum], run "./SynData_SGD PF TK 10 [CopyNum]".)
./SynData_SGD FS NY 1
```

Then synthesize traces (syntraces_cn0.csv) in PF and NYC will be generated in data/PF/SGD_TK/ and data/FS/SGD_NY/, respectively.

To evaluate the utility and privacy of the synthetic traces, run the following command.

```bash
cd python/
python3 EvalUtilPriv.py PF TK SGD 10
python3 EvalUtilPriv.py FS NY SGD 1
```

Then the results (utilpriv_PPMTF_TK.csv or utilpriv_PPMTF_NY.csv) will be stored in data/PF/ or data/FS/.

Synthesized traces in other cities (IST/JK/KL/SP/TKY) can also be generated and evaluated by replacing NY with IS, JK, KL, SP, or TK.

**(b) Synthesizing traces in PF using SGLT (Synthetic Location Traces Generator)**

Download the [SGLT (sglt-v0.1a.tgz)](https://vbinds.ch/node/70).

Run the following command to make input files for SGLT.

```bash
cd python/
python3 MakeSGLPMInput_PF.py
```

Then the input files (input.mobility, input.trace, and locations) are stored in data/PF/SGLT_TK/.

Run SGLT to synthesize traces (see README in SGLT for the usage).

Run the following command to combine the synthetic traces into one file.

```bash
cd python/
python3 MakeSGLPMOutput_PF.py 500 10
```

Then the synthetic traces (data_syntraces.csv) are stored in data/PF/SGLT_TK/.
(Change the file name based on the parameters in SGLT as follows: data_c[#cluster]\_[par_c]\_[par_m]\_[par_l]\_[par_v]_syntraces.csv.)

To evaluate the utility and privacy of the synthetic traces, run the following command.

```bash
cd python/
python3 EvalUtilPriv.py PF TK SGLT 10
```

Then the results (utilpriv_SGLT_TK.csv) will be stored in data/PF/.

**(c) Synthesizing traces in PF/FS using ITF (Independent Tensor Factorization)**

Replace PPMTF and SynData_PPMTF with PPITF and SynData_PPITF, respectively, and run (2) and (3) in README.md.

Then synthesize traces (syntraces_Itr100.csv) in PF and FS will be generated in data/PF/PPITF_TK_alp200_mnt100_mnv100/ and data/FS/PPITF_NY_alp200_mnt100_mnv100/, respectively. 

Experimental results (utilpriv_PPITF_TK.csv or utilpriv_PPITF_NY.csv) will also be stored in data/PF/ or data/FS/.
