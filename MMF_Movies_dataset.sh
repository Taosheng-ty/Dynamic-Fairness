#!/bin/bash
python main.py with 'EXPERIMENT=1_8' 'PLOT_PREFIX="plots/Asaved/batch_run_Movies_dataset/original/"' 'trials=2' 'iterations=6000' 'add_args={}'&
sleep 20
# python main.py with 'EXPERIMENT=2_8' 'PLOT_PREFIX="plots/Asaved/batch_run_Movies_dataset/top_down_0.0/"' 'trials=2' 'iterations=6000' 'add_args={"lambda_fair":0.0}' &
# sleep 20
# python main.py with 'EXPERIMENT=2_8' 'PLOT_PREFIX="plots/Asaved/batch_run_Movies_dataset/top_down_0.2/"' 'trials=2' 'iterations=6000' 'add_args={"lambda_fair":0.2}'&
# sleep 20
# python main.py with 'EXPERIMENT=2_8' 'PLOT_PREFIX="plots/Asaved/batch_run_Movies_dataset/top_down_0.4/"' 'trials=2' 'iterations=6000' 'add_args={"lambda_fair":0.4}'&
# sleep 20
# python main.py with 'EXPERIMENT=2_8' 'PLOT_PREFIX="plots/Asaved/batch_run_Movies_dataset/top_down_0.6/"' 'trials=2' 'iterations=6000' 'add_args={"lambda_fair":0.6}'&
# sleep 20
# python main.py with 'EXPERIMENT=2_8' 'PLOT_PREFIX="plots/Asaved/batch_run_Movies_dataset/top_down_0.8/"' 'trials=2' 'iterations=6000' 'add_args={"lambda_fair":0.8}'&
# sleep 20
# python main.py with 'EXPERIMENT=2_8' 'PLOT_PREFIX="plots/Asaved/batch_run_Movies_dataset/top_down_1.0/"' 'trials=2' 'iterations=6000' 'add_args={"lambda_fair":1.0}'&