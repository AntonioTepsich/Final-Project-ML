#!/bin/bash -ex       

# python main.py --model "configs/model/UNet_32_v1.py" \
#     	       --data "configs/data/500.py" \
# 	       --features "configs/features/none.py"


# python main.py --model "configs/model/CAE_32_v1.py" \
#     	       --data "configs/data/500.py" \
# 	       --features "configs/features/none.py"


python main.py --model "configs/model/CAE_32_v1.py" \
    	       --data "configs/data/CelebA.py" \
	       --features "configs/features/celeba.py"
