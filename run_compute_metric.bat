@echo off
REM filepath: d:\KTH_semester_3\DD2610_DAPS\run_compute_metric.bat

python compute_metric.py ^
    --gt_folder D:\KTH_semester_3\diffusion-posterior-sampling\data\ffhq_test^
    --pred_folder D:\KTH_semester_3\DD2610_DAPS\results\dps_phase_retrieval_ffhq ^
    --output results/metrics_results.txt

