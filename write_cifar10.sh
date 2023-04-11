#!/bin/bash

write_dataset () {
    write_path=/home/andong.hua/datasets/cifar10/${1}.beton
    data_dir=/home/andong.hua/datasets/raw/cifar102
    dataset=cifar102
    split=test
    echo "Writing cifar10 ${1} dataset to ${write_path}"
    python write_imagenet.py \
        --cfg.dataset=${dataset} \
        --cfg.split=${split} \
        --cfg.data_dir=${data_dir} \
        --cfg.write_path=$write_path \
        --cfg.max_resolution=-1 \
        --cfg.write_mode=raw \
        --cfg.compress_probability=0.5 \
        --cfg.jpeg_quality=90
}

# write_dataset train
write_dataset test102
