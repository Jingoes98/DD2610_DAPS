@echo off
REM filepath: d:\KTH_semester_3\DD2610_DAPS\run_compute_metric.bat

python compute_metric.py ^
    --gt_folder D:\KTH_semester_3\DD2610_DAPS\dataset\demo-ffhq ^
    --pred_folder D:\KTH_semester_3\DD2610_DAPS\results\daps_motion_deblur_ffhq_demo\samples ^
    --output results/metrics_results.txt

