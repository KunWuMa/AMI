# 实验与文件索引 / Experiment index

下表统一索引现有科学分析。代码以仓库根目录为起点；CSV 文件名保留计算来源的一致性。统计方法中原有 exploratory/post-hoc 定位不变。

| 研究问题/实验 | 主要代码 | 结果位置 | 论文对应 |
|---|---|---|---|
| 数据抽取、检验统一、队列构建 | code/public_development/00_preflight.py、01_extract_common_labs.py、01b_correct_numeric_cache.py、02_build_harmonized_cohorts.py | aggregate_results/cohort_flow.csv、cohort_summary_main.csv、phenotype_and_lab_dictionary.json | 队列流程、Methods、S1-S3 |
| 本院数据质量、跨记录关联及既往病史核对 | code/source_qc/01_profile.py 至 04_independent_verification.py | 汇总流程及质量核对；患者级产物不公开 | 队列定义和质量控制 |
| 最终本地队列及固定模型评价 | code/local_evaluation/01_build_confirmed_cohort.py、02_frozen_and_supporting_experiments.py | aggregate_results/conditional_fixed_model_metrics.csv、conditional_fixed_model_intervals.csv、conditional_fixed_paired_differences.csv | 主模型比较、Figure 2 |
| 公共模型开发与校准 | code/public_development/03_develop_and_external_validate.py、supplement_common.py | aggregate_results/model_dictionary.csv、feature_dictionary.csv；supporting_results.json | Methods、主结果 |
| 观察过程稳健性、消融与统计核对 | code/public_development/04_observation_robustness.py、05_exploratory_ablations.py、06a_audit_data.py、06b_audit_models.py、07_additional_statistics.py | supporting_results.json 中 sensitivity_*、fixed_prediction_sensitivities 等 | Supplement 稳健性分析 |
| 非 CABG 重开发与冻结模型对照 | code/public_development/10_nonCABG_development.py；code/public_extensions/03_ablations_pathways.py | aggregate_results/nonCABG_redevelopment_*、public_nonCABG_frozen_* | Figure 3、care composition |
| 重复划分与开发 bootstrap | code/public_development/11_development_stability.py、12_development_bootstrap.py | supporting_results.json 中 *repeated_split*、*development_bootstrap* | 开发不确定性 |
| 本地交叉拟合概率更新 | code/public_development/13_local_crossfit_and_temporal_update.py；code/local_evaluation/02_frozen_and_supporting_experiments.py | supporting_results.json/local_crossfit_updating | 探索性更新 |
| 时间窗口和时期敏感性 | code/public_development/14_timing_and_period_validation.py | supporting_results.json 中 *strict48h*、*MIMIC_period* | 时间敏感性 |
| 插补敏感性 | code/public_development/15_imputation_sensitivity.py | supporting_results.json 中 *iterative_imputation* | 插补敏感性 |
| 结局观测、逻辑界限及加权敏感性 | code/public_development/16_future_observation_selection.py | supporting_results.json/observation_bounds、observation_selection、observation_weighted_sensitivity | 结局可观测性 |
| 来源诊断与亚组分析 | code/public_development/17_domain_and_subgroups.py | supporting_results.json/dataset_origin_diagnostic 等 | 描述性来源差异，非因果机制 |
| 汇总导出与完整性检查 | code/public_development/09_prepare_supplementary_data.py、18_audit_supplementary.py | 原始汇总和字典 | 结果可核对性 |
| CABG 诊断编码及操作时间关系 | code/public_extensions/01_procedure_audit.py | supporting_results.json/CABG_code_dictionary、public_CABG_timing_summary；aggregate_results/public_pathway_composition.csv | 操作定义与病例组成 |
| 观察 O / 联合结局 J 模型 | code/public_extensions/02_observation_two_stage.py | supporting_results.json/two_stage_*；figure_source_data/observation_joint_probabilities.csv | O/J 模型、Figure S5 |
| 检测模式表示与白蛋白相关消融 | code/public_extensions/03_ablations_pathways.py | aggregate_results/conditional_fixed_*；figure_source_data/main_representation_* | Figure 4 |
| 连续 Hb 结局 | code/public_extensions/04_continuous_Hb.py | supporting_results.json/continuous_*、risk_continuous_associations | Figure S4 |
| 配对不确定性及冻结一致性 | code/public_extensions/05_paired_uncertainty_and_consistency.py、06_validate_frozen_models.py；code/local_evaluation/09_model_equations_and_public_check.py | 配对结果；supporting_results.json/frozen_logistic_equations | 方法核对与公开参数 |
| 入组筛选与 Hb 可观测/不可观测人群比较（E1） | code/reliability_diagnostics/02_E1_E3.py | aggregate_results/reliability_diagnostics/E1_*.csv | S19-S20 |
| 概率常数参考、Brier skill、E/O 与既有校准精度（E3） | code/reliability_diagnostics/02_E1_E3.py | aggregate_results/reliability_diagnostics/E3_*.csv | S22 |
| CABG 组成与跨场景化验记录分布（E2） | code/reliability_diagnostics/03_E2.py | aggregate_results/reliability_diagnostics/E2_*.csv | S21 |
| 冻结 M3 系数、分组贡献及观测模式风险（E4） | code/reliability_diagnostics/04_E4.py | aggregate_results/reliability_diagnostics/E4_*.csv | S23 |
| 未来 Hb 检测机会和首次检测日期（E5） | code/reliability_diagnostics/05_E5.py | aggregate_results/reliability_diagnostics/E5_*.csv | S20 支持面板 |
| 诊断分析私有缓存准备和公共工具 | code/reliability_diagnostics/01_prepare_private.py、analysis_common.py | 私有缓存不公开 | 分析依赖 |
| S19-S23 表格排版 | code/reliability_diagnostics/06_build_addendum.py | docs/reliability_diagnostics/Supplement_addendum_S19-S23.pdf | S19-S23 与方法 |
| 主图和补充图绘制 | code/presentation/draw_*.py | manuscript/figures；figure_source_data | Figures 1-4、S1-S5 |
| 报告清单及 PDF 检查工具 | code/presentation/build_reporting_checklists.py、render_and_check.py | 使用者编译后的报告和页码 | 需以最终稿重新定位页码 |

## 汇总数据读取

aggregate_results/supporting_results.json 按实验名存储 columns 和 rows。根目录 prepare_aggregate_tables.py 可将其展开成绘图需要的汇总 CSV，不重算实验。文件名中的历史计算标签用于依赖追溯，不表示另一个独立验证队列。最终本地条件性评价使用统一的 Local100。

## 可复现范围

公开结果和代码可以核对定义及数值；完整重新执行需要获授权输入、原划分与冻结对象。目录重排只改变公开代码的输出定位，不重拟合模型、不改动实验数据。
