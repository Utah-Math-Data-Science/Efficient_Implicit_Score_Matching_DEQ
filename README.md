# Efficient Score Matching with Deep Equilibrium Layers

This code is based on the code of the following paper: Sliced Score Matching: A Scalable Approach to Density and Score Estimation](https://arxiv.org/abs/1905.07088), UAI 2019. 


## Dependencies

The following are packages needed for running this repo.

- PyTorch==1.0.1
- TensorFlow==1.12.0
- tqdm
- tensorboardX
- Scipy
- PyYAML



## Running the experiments
```bash
python main.py --runner [runner name] --config [config file]
```

Here `runner name` is one of the following:

- `DKEFRunner`. This corresponds to experiments on deep kernel exponential families.
- `NICERunner`. This corresponds to the sanity check experiment of training a NICE model.
- `VAERunner`. Experiments on VAEs.
- `WAERunner`. Experiments on Wasserstein Auto-Encoders (WAEs).

and `config file` is the directory of some YAML file in `configs/`.



For example, if you want to train an implicit VAE of latent size 8 on MNIST with Sliced Score Matching, just run

```bash
python main.py --runner VAERunner --config vae/mnist_ssm_8.yml
```

For the NCSN model, use
`main.py` is the file that you should run for both training and sampling. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] --config CONFIG [--seed SEED] [--exp EXP] --doc DOC
               [--comment COMMENT] [--verbose VERBOSE] [--test] [--sample]
               [--fast_fid] [--resume_training] [-i IMAGE_FOLDER] [--ni]

The DEQ models will run by default. If you wanna run other models, just open the config file and set deq (or fix_nn) as False.

