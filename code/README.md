# Scientific code and provenance

## Analysis structure
- public_development/: extraction, harmonization, public model development and supporting analyses.
- public_extensions/: observation/joint models, representation and pathway analyses, continuous outcomes, paired uncertainty and computational validation.
- source_qc/: source extraction, linkage and validity checks.
- local_evaluation/: final hospital cohort construction, frozen evaluation and model-equation/public-prediction verification. Principal conditional outputs use Public_test and Local100.
- presentation/: current figure generation, reporting-checklist generation and PDF layout validation.
- provenance/: study chronology and image-generation disclosure.

Care-pathway and representation analyses were exploratory/post-hoc, informed by earlier local results. No local labels were used to refit primary frozen public models. Local probability updating remains a separate exploratory analysis.

## Reproduction requirements
Install requirements-analysis.txt in the stated Python environment. Access to credentialed MIMIC data, authorized hospital extracts, the saved public allocation, frozen public objects and private source intermediates is required to reproduce patient-level analyses. These protected inputs and model binaries are not distributed. Protected input roots use environment variables documented in PATH_CONFIGURATION.md. Relative-layout assumptions and scientific input filenames are preserved; adapt the authorized workspace layout accordingly. This is not a turnkey execution package, and curation was checked statically without rerunning clinical analyses.

Public model sources and source QC precede local_evaluation/01_build_confirmed_cohort.py, then 02_frozen_and_supporting_experiments.py and 09_model_equations_and_public_check.py. Existing statistical results accompany the aggregate_results directory. Figure scripts run from a manuscript root containing aggregate_results and figure_source_data. The separate journal LaTeX source archive supplies the current manuscript, supplement, class/style, figures and checklist inputs; compile before regenerating checklist page locations. Tectonic/LaTeX and Poppler are separate system dependencies.

../release_manifest.csv records public file SHA-256 hashes; ../docs/EXPERIMENTS.md maps scientific roles. Computational checks are retained, including paired uncertainty and frozen-model validation; temporary probes, console-only summaries, internal narrative reports and superseded manuscript-rewriting helpers are excluded. Original Chinese phenotype strings remain where required to interpret hospital records.

No raw or individual-level patient data, individual predictions, linkage keys or fitted binary models are included.

See PATH_CONFIGURATION.md for protected input roots, Poppler and optional font configuration.
