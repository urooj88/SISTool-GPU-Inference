#!/bin/bash

cd /home/urooj/eut+dats-stats-w26-GPU/v2tudembeded-model-with-SIS-final-23-july-updated-GPU/omegawatt_scripts || exit

echo "==============================="
echo " Running Model 1"
echo "==============================="
python3 run_omegawatt_log_models12.py --app-type serial --gpu-count 1
echo " Model 1 complete."

echo ""
echo " Sleeping for 20 seconds before running Model 2..."
sleep 20
echo ""

echo "==============================="
echo " Running Model 2"
echo "==============================="
python3 run_omegawatt_log_models12.py --app-type parallel --gpu-count 1
echo " Model 2 complete."

echo ""
echo " Sleeping for 20 seconds before running Model 3..."
sleep 20
echo ""

echo "==============================="
echo " Running Model 3"
echo "==============================="
python3 run_omegawatt_log_models12.py --app-type third --gpu-count 1
echo " Model 3 complete."

echo ""
echo " Sleeping for 20 seconds before running Model 4..."
sleep 20
echo ""

echo "==============================="
echo " Running Model 4"
echo "==============================="
python3 run_omegawatt_log_models12.py --app-type fourth --gpu-count 1
echo " Model 4 complete."

echo ""
echo "==============================="
echo " All model runs finished."
echo " Check model_metrices.csv for the updated results."
echo "==============================="
