#!/bin/bash

cd LPAP

for d in {miniimagenet,FC100,cifar_fs}
do
for j in {MAML,ANIL,BOIL}
do
for i in {1,5}
do
python valid.py --dataset=$d \
                       --device=cuda \
                       --device_index=0 \
                       --num_ways=5 \
                       --num_shots=$i \
                       --num_querys=15 \
                       --algorithm=$j \
                       --model=4conv \
                       --meta_lr=1e-3 \
                       --hidden_size=64 \
                       --batch_size=4 \
                       --outer_iter=300 \
                       --train_batches=100 \
                       --valid_batches=25 \
                       --test_batches=2500 \
                       --note=V1
done
done
done
echo "finished"




