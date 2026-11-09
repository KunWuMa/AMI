# Portable input and tool configuration

Environment variables identify authorized inputs; none of the protected inputs is supplied. Relative defaults are resolved from the working directory. Maintain each script's documented relative study layout or adapt its root constants; these changes do not make the full analysis turnkey.

| Variable | Relative default | Purpose |
|---|---|---|
| AMI_WORKSPACE_ROOT | workspace | Workspace used by source preflight |
| AMI_STUDY_ROOT | workspace/study | Authorized study artifacts for source checks |
| AMI_SOURCE_QC_ROOT | inputs/source_qc | Hospital-source QC outputs and private intermediates |
| AMI_HOSPITAL_ZIP | inputs/hospital_export.zip | Authorized hospital source archive |
| AMI_PSEUDONYM_KEY | inputs/private/pseudonym_key.bin | Private linkage key; never publish |
| AMI_HISTORICAL_CACHE | inputs/private/historical_coronary_cache.pkl | Authorized linked historical coronary records |
| PDFTOPPM | discovered on PATH | Poppler pdftoppm executable |
| REPORT_FONT_REGULAR | fonts/arial.ttf | Optional regular report font |
| REPORT_FONT_BOLD | fonts/arialbd.ttf | Optional bold report font |

If both optional fonts are not available, reporting uses the Vera fonts bundled with ReportLab. The sibling QC script is resolved relative to its caller. Figure and reporting scripts should be placed in the manuscript root alongside aggregate_results and figure_source_data, as in the LaTeX source archive. Analysis scripts were syntax-checked after path parameterization; full patient-level analyses were not rerun.
