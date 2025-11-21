@echo off
REM filepath: d:\KTH_semester_3\DD2610_DAPS\run_posterior_sample.bat

python posterior_sample.py ^
    --config-path=configs ^
    --config-name=run_dps_PhaseRetrieval_ffhq.yaml

