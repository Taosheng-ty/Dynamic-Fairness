#!/bin/bash
python main.py with 'EXPERIMENT=1_1' 'PLOT_PREFIX="plots/Asaved/batch_run_News_dataset/original/"' 'trials=20' 'iterations=6000' 'add_args={}'&
sleep 10
python main.py with 'EXPERIMENT=4' 'PLOT_PREFIX="plots/Asaved/batch_run_News_dataset/lp_E/"' 'trials=5' 'iterations=6000' 'add_args={}' 'item_file="plots/Asaved/batch_run_News_dataset/original/"'&
sleep 10
python main.py with 'EXPERIMENT=2_1' 'PLOT_PREFIX="plots/Asaved/batch_run_News_dataset/top_down_0.0/"' 'trials=20' 'iterations=6000' 'add_args={"lambda_fair":0.0}' 'item_file=plots/Asaved/batch_run_News_dataset/original/'&
sleep 10
python main.py with 'EXPERIMENT=2_1' 'PLOT_PREFIX="plots/Asaved/batch_run_News_dataset/top_down_0.2/"' 'trials=20' 'iterations=6000' 'add_args={"lambda_fair":0.2}' 'item_file=plots/Asaved/batch_run_News_dataset/original/'&
sleep 10
python main.py with 'EXPERIMENT=2_1' 'PLOT_PREFIX="plots/Asaved/batch_run_News_dataset/top_down_0.4/"' 'trials=20' 'iterations=6000' 'add_args={"lambda_fair":0.4}' 'item_file=plots/Asaved/batch_run_News_dataset/original/'&
sleep 10
python main.py with 'EXPERIMENT=2_1' 'PLOT_PREFIX="plots/Asaved/batch_run_News_dataset/top_down_0.6/"' 'trials=20' 'iterations=6000' 'add_args={"lambda_fair":0.6}' 'item_file=plots/Asaved/batch_run_News_dataset/original/'&
sleep 10
python main.py with 'EXPERIMENT=2_1' 'PLOT_PREFIX="plots/Asaved/batch_run_News_dataset/top_down_0.8/"' 'trials=20' 'iterations=6000' 'add_args={"lambda_fair":0.8}' 'item_file=plots/Asaved/batch_run_News_dataset/original/'&
sleep 10
python main.py with 'EXPERIMENT=2_1' 'PLOT_PREFIX="plots/Asaved/batch_run_News_dataset/top_down_1.0/"' 'trials=20' 'iterations=6000' 'add_args={"lambda_fair":1.0}' 'item_file=plots/Asaved/batch_run_News_dataset/original/'&