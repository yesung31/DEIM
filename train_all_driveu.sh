CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deim_dfine/deim_hgnetv2_n_driveu.yml --use-amp --seed=0
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deim_dfine/deim_hgnetv2_s_driveu.yml --use-amp --seed=0
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deim_dfine/deim_hgnetv2_m_driveu.yml --use-amp --seed=0
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deim_dfine/deim_hgnetv2_l_driveu.yml --use-amp --seed=0
CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deim_dfine/deim_hgnetv2_x_driveu.yml --use-amp --seed=0
