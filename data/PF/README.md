# About the dataset
All the location data below this directory are generated based on the [SNS-based people flow data](https://nightley.jp/archives/1954/). For more details, see: https://arxiv.org/abs/1911.04226.

Attribute work to name in the SNS-based People Flow Data is as follows:

[Attribute work to name]

Nightley, Inc.

Shibasaki & Sekimoto Laboratory, the University of Tokyo

Micro Geo Data Forum

People Flow project

Center for Spatial Information Science at the University of Tokyo

# Directory Structure

Directory Structure is as follows (TK: Tokyo, alp200: alpha=200, mnt100: maximum number of transitions=100, mnv100: maximum number of visits=100).

- PPMTF_TK_alp200_mnt100_mnv100/	&emsp;Directory containing PPMTF parameters and synthetic traces.
- SGD_TK/				&emsp;Directory containing synthetic traces by SGD.
- SGLT_TK/				&emsp;Directory containing synthetic traces by SGLT.
- euserindex_TK.csv			&emsp;Testing user index file.
- POI_TK.csv				&emsp;POI file.
- POIindex_TK.csv			&emsp;POI index file.
- testtraces_TK.csv			&emsp;Testing trace file.
- traces_TK.tar.gz			&emsp;Compressed Trace file (uncompressed file is: traces_TK.csv).
- traintraces_TK.csv			&emsp;Training trace file.
- traintranstensor_TK_mnt100.csv	&emsp;Training transition-count tensor file.
- trainvisittensor_TK_mnt100.csv	&emsp;Training visit-count tensor file.
- tuserindex_TK.csv			&emsp;Training user index file.
- utilpriv_PPMTF_TK.csv			&emsp;Utility and privacy of PPMTF (experimental results).
- utilpriv_SGD_TK.csv			&emsp;Utility and privacy of SGD (experimental results).
- utilpriv_SGLT_TK.csv			&emsp;Utility and privacy of SGLT (experimental results).
