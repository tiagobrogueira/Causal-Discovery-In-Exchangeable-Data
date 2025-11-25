# FOM

## Prepare environment

use dockerfile to build experiment environment

```
#!/bin/bash
docker build -t causality:latest .
```

## Synthetic dataset[Optional]

run the script will generate dataset into `./datasets/`

```
#!/bin/bash
docker run --rm -v `pwd`:/code causality:latest python ./data/synthetic_data.py
```

## Run experiment

run experiment for different algorithms, e.g. ANM, IGCI, RECI, FOM(our method)

```
#!/bin/bash
docker run --rm -v `pwd`:/code causality:latest python experiment.py \
    --experiment_type CEP --algorithm IGCI --dataset ./datasets/CEP
```