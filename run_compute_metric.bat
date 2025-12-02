@echo off
REM filepath: d:\KTH_semester_3\DD2610_DAPS\run_compute_metric.bat

python compute_metric.py ^
    --gt_folder D:\KTH_semester_3\DD2610_DAPS\dataset\test_ffhq_30 ^
    --pred_folder D:\KTH_semester_3\DD2610_DAPS\results\daps_phase_retrieval_ffhq_test_30\samples ^
    --output results/metrics_results.txt

