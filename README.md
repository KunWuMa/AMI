# AMI: cross-setting reliability of anemia prediction

Research code, aggregate results and manuscript materials for **Cross-setting reliability of anemia prediction after acute myocardial infarction: a retrospective evaluation of observation processes and model transport**.

Repository: https://github.com/KunWuMa/AMI

## Experiment index
Start with [EXPERIMENTS.md](docs/EXPERIMENTS.md), which maps scientific questions to scripts, outputs and manuscript tables. All analyses are organized by scientific purpose, not by development date or attachment number.

| Directory | Contents |
|---|---|
| code/public_development | Public extraction, harmonization, primary model development and robustness analyses |
| code/local_evaluation | Hospital cohort construction and frozen external evaluation |
| code/public_extensions | Observation/joint models, care composition, feature representations and continuous outcomes |
| code/reliability_diagnostics | Selection, observation, probability-reference and frozen-contribution diagnostics |
| code/source_qc | Source validity and linkage checks |
| code/presentation | Figure and reporting scripts |
| aggregate_results | Frozen performance, comparisons, dictionaries and supporting aggregate results |
| figure_source_data | Aggregate scientific plot coordinates and counts |
| manuscript | Current author manuscript source, existing supplement and figures |
| docs | Experiment index, methods, S19-S23 tables, provenance and access restrictions |
| tools | Public-release validation and manifest tooling |

## Scientific scope
The local cohort contains 116 eligible patients, 100 with future Hb and 46 recorded-anemia events. The public eligible cohort contains 1,934 patients, with 1,846 outcome-observed patients and 370 held-out test patients. D0-D1 features predict D2-D7 recorded anemia from the start of calendar D2, conditional on future Hb observation.

Primary models, preprocessing and public calibration remain frozen for local evaluation. Exploratory/post-hoc analyses retain that methodological status; unified packaging does not turn them into prespecified analyses. Patients without future Hb are not treated as anemia-negative.

## Reproduction
See code/README.md and code/PATH_CONFIGURATION.md. Full analysis requires authorized source data, the original allocation, frozen model objects and the documented study layout. None is distributed here. Run python prepare_aggregate_tables.py to restore supporting aggregate tables for presentation scripts. See code/reliability_diagnostics/README.md for diagnostic scripts. Run python tools/audit_public_release.py for static release validation.

## Data and manuscript status
No raw records, patient identifiers (including pseudonyms), patient-level features/predictions, linkage keys, private allocations, credentials or fitted binary objects are distributed. See docs/DATA_ACCESS.md. No DOI, journal acceptance or software license is invented. Springer Nature template notices are retained.

The manuscript source is the current author draft and still contains unfinished template/declaration material. S19-S23 and their methods are included under docs/reliability_diagnostics; no unrequested scientific rewriting of the manuscript or figures was performed.

中文导航：docs/README_zh.md。
