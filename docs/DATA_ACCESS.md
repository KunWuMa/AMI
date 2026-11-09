# Data access and disclosure boundaries

Included: scientific source code, aggregate cohort summaries, performance/uncertainty estimates, figure coordinates, dictionaries and coefficients, manuscript source and scientific tables.

Excluded: raw hospital/MIMIC extracts, patient-level features/outcomes/predictions, identifiers and pseudonyms, allocation lists, linkage maps and cryptographic keys, clinical narratives, fitted binary objects, credentials, workstation paths, private caches and logs.

MIMIC-IV requires credentialed PhysioNet access and adherence to its data-use terms. Hospital records require institutional and ethical authorization. This repository grants no clinical-data access approval. Never upload patient records in issues or pull requests.

The public code and aggregates support inspection, but full patient-level replication requires matching authorized inputs, allocation and frozen artifacts. Public model coefficients are included; private fitted binaries are not. The release validator checks schemas, prohibited formats, identifier-like literals, credentials and absolute workstation paths; new contributions require fresh review.
