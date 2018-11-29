# pv2 eval --gpus 5 --eval=$GEOPROD --level=minor --sys_packages /home/mifs/mttt2/cvfs/RUNS/localseg/FinalP5Bench/xentropy_res50_camvid_geo_2018_11_14_19.57
# pv2 eval --gpus 5 --eval=$GEOPROD --level=minor --sys_packages /home/mifs/mttt2/cvfs/RUNS/localseg/FinalP5Bench/NoXentropy_res50_camvid_geo_2018_11_14_19.57
pv2 eval --gpus 5 --eval=$GEOPROD --level=minor --sys_packages /home/mifs/mttt2/cvfs/RUNS/localseg/FinalP3Bench/xentropy_final_2018_11_14_20.02
pv2 eval --gpus 5 --eval=$GEOPROD --level=minor --sys_packages /home/mifs/mttt2/cvfs/RUNS/localseg/FinalP3Bench/NoXentropy_final_2018_11_14_20.02
pv2 eval --gpus 5 --eval=$GEOPROD --level=minor --sys_packages /home/mifs/mttt2/cvfs/RUNS/localseg/FinalP4Bench/xentropy_res50_camvid_geo_2018_11_13_23.32
pv2 eval --gpus 5 --eval=$GEOPROD --level=minor --sys_packages /home/mifs/mttt2/cvfs/RUNS/localseg/FinalP4Bench/NoXentropy_res50_camvid_geo_2018_11_13_23.32 
pv2 eval --gpus 6 --eval=$GEOPROD --level=minor --sys_packages /home/mifs/mttt2/cvfs/RUNS/localseg/FinalP3Bench/NewLoss_final_2018_11_14_20.02


pv2 eval --gpus 2 --eval=$GEOPROD --level=minor --sys_packages /home/mifs/mttt2/cvfs/RUNS/localseg/FinalP6Bench/xentropy_res50_camvid_geo_2018_11_14_20.19 && pv2 eval --gpus 2 --eval=$GEOPROD --level=minor --sys_packages /home/mifs/mttt2/cvfs/RUNS/localseg/FinalP6Bench/NoXentropy_res50_camvid_geo_2018_11_14_20.19