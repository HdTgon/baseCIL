# On the Power of Statistics in Class-Incremental Learning with Pretrained Models

The code repository for "On the Power of Statistics in Class-Incremental Learning with Pretrained Models" (ICML 2026) in PyTorch.

## Requirements
### Building environment

```
conda env create -f baseCIL.yaml
```

### Datasets
The datasets can be downloaded from the sources provided in [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT "PILOT").

You need to modify the path of the datasets in `./utils/data.py` according to your own path.

### How to Run
```
python main.py --config exps/[config_name].json
```

## Acknowledgement
This repo is based on [PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT "PILOT").
