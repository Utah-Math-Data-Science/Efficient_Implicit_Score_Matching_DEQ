import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
from runners import *
import wandb


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    ## runner ##
    parser.add_argument('--runner', type=str, default='VAERunner', required=False, help='The runner to execute')

    ## config file ##
    parser.add_argument('--config', type=str, default='/root/workspace/sliced_score_matching/configs/vae/celeba_ssm.yml', required=False, help='Path to the config file')

    ## saving file folder
    parser.add_argument('--doc', type=str, 
                        default='Cifera10_vae_gen512_256_2', 
                        help='A string for documentation purpose')

    parser.add_argument('--scalability_dim', type=int, default=200, help='Dimension for scalability testing')
    parser.add_argument('--fixed_net', type=bool, default=False, help='Sigma for DSM tuning')
    parser.add_argument('--test', default=False, action='store_true', help='Whether to test the model')

    # debug
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--run', type=str, default='run', help='Path for saving running related data.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('--dsm_sigma', type=float, default=0.16, help='Sigma for DSM tuning')
    parser.add_argument('--sigma_f', type=float, default=0.5, help='Sigma for image prepare')
    parser.add_argument('--mu_f', type=float, default=0.5, help='mu for image prepare')
    parser.add_argument('--load_path', type=str, default='', help='Path to state dict for resuming training')

    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])
    args.log = os.path.join(args.run, 'logs', args.doc)

    kwargs = {
                'entity': 'utah-math-data-science', 
                'project': 'NCSN-CIFAR10-DEQ',
                'mode': 'disabled',
                'name': 'cifer10_resdeq_test05', 
                'config': args,
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True
             }
    wandb.init(**kwargs)
    wandb.save('*.txt')


    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f)
    new_config = dict2namespace(config)

    if not args.test:
        if not args.resume_training:
            if os.path.exists(args.log):
                shutil.rmtree(args.log)
            os.makedirs(args.log)

        with open(os.path.join(args.log, 'config.yml'), 'w') as f:
            yaml.dump(new_config, f, default_flow_style=False)

        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    print(config)
    print("<" * 80)


    try:
        runner = eval(args.runner)(args, config)
        if not args.test:
            runner.train()
        else:
            # runner.test_fid()
            runner.fid_test_fast()
    except:
        logging.error(traceback.format_exc())

    return 0

if __name__ == '__main__':
    sys.exit(main())
