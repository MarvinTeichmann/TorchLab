bliationBench.sh 
#!/bin/bash

source /data/scratch-no-backup/ibu6429/shared_file_system/segmentation_experiments/exp_012_torch_experiments/localconf

pv2 train /data/scratch-no-backup/ibu6429/shared_file_system/segmentation_experiments/exp_012_torch_experiments/RUNS/localseg/AbliationBench/HeavyAug_final_2018_11_14_18.35 --gpus 0
pv2 train /data/scratch-no-backup/ibu6429/shared_file_system/segmentation_experiments/exp_012_torch_experiments/RUNS/localseg/AbliationBench/NoAug_final_2018_11_14_18.35 --gpus 0
pv2 train /data/scratch-no-backup/ibu6429/shared_file_system/segmentation_experiments/exp_012_torch_experiments/RUNS/localseg/AbliationBench/NoCross_final_2018_11_14_18.35 --gpus 1
pv2 train /data/scratch-no-backup/ibu6429/shared_file_system/segmentation_experiments/exp_012_torch_experiments/RUNS/localseg/AbliationBench/Sphere_final_2018_11_14_18.35 --gpus 2

pv2 train /data/scratch-no-backup/ibu6429/shared_file_system/segmentation_experiments/exp_012_torch_experiments/RUNS/localseg/AbliationBench/SphereNoInstance_final_2018_11_14_18.35 --gpus 3
pv2 train /data/scratch-no-backup/ibu6429/shared_file_system/segmentation_experiments/exp_012_torch_experiments/RUNS/localseg/AbliationBench/Subsample10_final_2018_11_14_18.35 --gpus 4
pv2 train /data/scratch-no-backup/ibu6429/shared_file_system/segmentation_experiments/exp_012_torch_experiments/RUNS/localseg/AbliationBench/Subsample30_final_2018_11_14_18.35 --gpus 5
