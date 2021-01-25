# Directory Structure
- include/		&emsp;Include directory.
  - install_dependencies.sh			&emsp;Shell script to install all libraries (StatsLib/GCEM/Eigen).
- common.cpp		&emsp;Common C++ code.
- common.hpp		&emsp;Common C++ header.
- Makefile		&emsp;Makefile.
- PDTest_Trace.cpp	&emsp;Perform the PD test for synthetic traces.
- PDTest_Trace_time.cpp	&emsp;Perform the PD test for synthetic traces and measure the running time.
- PPITF.cpp		&emsp;Perform PPITF (Privacy-Preserving Independent Tensor Factorization).
- PPMTF.cpp		&emsp;Perform PPMTF (Privacy-Preserving Multiple Tensor Factorization).
- PPMTF_time.cpp	&emsp;Perform PPMTF (Privacy-Preserving Multiple Tensor Factorization) and measure the running time.
- PPXTF.cpp		&emsp;Common C++ code for PPMTF/PPITF.
- PPXTF.hpp		&emsp;Common C++ header for PPMTF/PPITF.
- ProcessingTime.hpp	&emsp;Header for measuring the running time.
- README.md		&emsp;This file.
- SynData_PPITF.cpp	&emsp;Synthesize traces using parameters in PPITF.
- SynData_PPMTF.cpp	&emsp;Synthesize traces using parameters in PPMTF.
- SynData_PPMTF.cpp	&emsp;Synthesize traces using parameters in PPMTF and measure the running time.
- SynData_SGD.cpp	&emsp;Synthesize traces using SGD (Synthetic Data Generator).
- SynData_SGD_time.cpp	&emsp;Synthesize traces using SGD and measure the running time.

# Required Libraries
* StatsLib
  * https://www.kthohr.com/statslib.htm
* Generalized Constant Expression Math
  * https://www.kthohr.com/gcem.html
* Eigen 3.3.7
  * http://eigen.tuxfamily.org/index.php?title=Main_Page

# Setup
Please put the following files/directories under 'include/'.

**Eigen 3.3.7**
- Eigen/

**Generalized Constant Expression Math**
- gcem.hpp
- gcem_incl/

**StatsLib**
- stats.hpp
- stats_incl/

You can easily put them by running include/install_dependencies.sh:
```
$ cd include
$ sh install_dependencies.sh
```
