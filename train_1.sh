#!/bin/bash
CUDA_VISIBLE_DEVICES="0,1" python main.py -b 6 -n e -td datasets/kitti360_seq03.hdf5 -tdc datasets/kitti360_penet_train.json -bs 16 -vd datasets/kitti360_seq03.hdf5 -vdc datasets/kitti360_penet_val.json -he 256 -w 512 --workers 0
