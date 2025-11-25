#!/bin/bash
docker run --rm -v `pwd`:/code causality:latest python ./data/synthetic_data.py