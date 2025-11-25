#!/bin/bash
docker run --rm -v `pwd`:/code causality:latest python experiment.py \
    --experiment_type CEP --algorithm FOM --dataset ./datasets/CEP