@echo off
REM filepath: d:\KTH_semester_3\DD2610_DAPS\run_compute_metric.bat

python evaluate_metrics.py ^
    --gt D:\KTH_semester_3\DD2610_DAPS\dataset\DIV2K_HR_test_3 ^
    --pred D:\KTH_semester_3\DD2610_DAPS\results\dps_SR_div2k\samples2 ^
    --image_size 256 


