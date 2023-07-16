#!/bin/bash

cd LPAP

python train.py --dataset=cifar_fs \
                       --device=cpu \
                       --device_index=0 \
                       --num_ways=5 \
                       --num_shots=1 \
                       --num_querys=15 \
                       --algorithm=MAML \
                       --model=4conv \
                       --meta_lr=1e-3 \
                       --hidden_size=64 \
                       --batch_size=4 \
                       --outer_iter=300 \
                       --train_batches=100 \
                       --valid_batches=25 \
                       --test_batches=2500 \
                       --note=Base
echo "finished"




