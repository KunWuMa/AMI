# Reliability diagnostics

All analyses use frozen models. Execution order: 01_prepare_private.py; 02_E1_E3.py; 03_E2.py; 04_E4.py; 05_E5.py; optionally 06_build_addendum.py. E labels map to the experiment index and do not denote separate repository releases.

Set AMI_STUDY_ROOT to the authorized original study root and preserve the documented study input layout. Run preparation with pandas 3 for existing cached pickle compatibility. Run analytical scripts with requirements-analysis.txt; use requirements-preparation-and-pdf.txt for preparation/rendering. Only load trusted study pickle files.

The public root resolves from the script location. Private caches are written to private_cache (Git-ignored), aggregate output to aggregate_results/reliability_diagnostics, and documentation to docs/reliability_diagnostics. Source datasets, patient predictions and frozen objects are not included. No primary model is refitted. Bootstrap intervals are reused; unavailable intervals are not manufactured.

Existing analytical QC CSVs describe the completed scientific run. docs/public_release_audit.json describes public-package checks. Mathematical contributions and pathway comparisons are not causal effects; O=0 conditional outcomes remain unknown.
