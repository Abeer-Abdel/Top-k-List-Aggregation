# Top-k-List-Aggregation
This repository contains the datasets and source code used in the journal article:
Heuristic Methods for Top-k List Aggregation under the Generalized Kendall Tau Distance - 
EvoCOP 2026

Scope and Purpose - 
The goal of this repository is to enable reproducibility of the computational experiments presented in the associated paper. It provides:
- Preference datasets used in the experiments
- Preprocessing procedures applied to the data
- Implementations of the proposed heuristic aggregation methods, local search refinement, and data reduction techniques

Data Source - 
All preference datasets are sourced from PrefLib – The Preference Library:https://www.preflib.org
The original PrefLib instances are retained, except for minimal preprocessing applied to ensure consistency across experimental instances (please see below)

Dataset Preprocessing - 
File Name	        Preprocessing Applied
00044-00000038.soi	Truncated to the minimum ranking length across all lists
00046-00000004.soi	Truncated to the minimum ranking length across all lists
00056-00001203.soc	Truncated to 50% of the ranking length (original instance is fully ranked)
00048-00000561.soi	No preprocessing applied
00045-00000029.soi	No preprocessing applied
00043-00000196.soi	Truncated to the minimum ranking length across all lists
00050-00000001.soc	Truncated to the minimum ranking length across all lists
00051-00000012.soi	Truncated to the minimum ranking length; final ranking discarded (anomaly)

Reproducibility - 
All preprocessing steps and algorithmic procedures described in the paper can be reproduced using the code provided in this repository.
Script names correspond directly to the methods they implement.

Licensing and Attribution - 
The datasets remain subject to PrefLib’s original licensing terms.
Users of this repository should cite both PrefLib and the associated journal publication.
