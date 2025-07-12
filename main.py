import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from sklearn.metrics import mean_squared_error
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config
from Utils.metrics2 import get_metric



def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', default=True, 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default=None,
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)
    net = config['model']['params']['net']
    args.save_dir = os.path.join(args.output, f'{args.name}+{net}+{args.mode}')

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
        trainer = Trainer(config=config, args=args, model=model, logger=logger)
    else:
        dataloader_info = build_dataloader(config, args)
        trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)
    
    if args.train:
        trainer.train(args.mode) 
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        print('Testing ', args.mode)
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        if args.mode in 'infill':
            samples, reals, shifts, masks, inputs = trainer.inference(dataloader, [dataset.seq_len, dataset.var_num], args.mode, coef, stepsize, sampling_steps)
        else:
            print('predicting ...')
            samples, reals, shifts, masks, inputs = trainer.predict(dataloader, [dataset.seq_len, dataset.var_num], args.mode, coef, stepsize, sampling_steps)
            # samples, reals, shifts, masks = trainer.predict(dataloader, [dataset.seq_len, dataset.var_num], args.mode, coef, stepsize, sampling_steps)
        
        print(samples.shape, reals.shape, shifts.shape, masks.shape, 'data info ...')
        
        # print('unnormalize ...')
        # samples = dataset.unnormalize(samples)
        # reals = dataset.unnormalize(reals)
        # shifts = dataset.unnormalize(shifts)
        # inputs = dataset.unnormalize(inputs)
        
        # np.save(os.path.join(args.save_dir, f'samples_{args.mode}_{args.name}.npy'), samples)
        # np.save(os.path.join(args.save_dir, f'reals_{args.mode}_{args.name}.npy'), reals)
        # np.save(os.path.join(args.save_dir, f'inputs_{args.mode}_{args.name}.npy'), reals)
        print('results saved to {}'.format(args.save_dir))
            
        
        samples, reals, shifts = samples.flatten(), reals.flatten(), shifts.flatten()

        print(samples.shape, reals.shape, shifts.shape, 'masked data info ...')
        mse = mean_squared_error(samples, reals)
        print(mse)
        
        samples = samples.reshape(samples.shape[0],-1)
        reals = reals.reshape(reals.shape[0],-1)
        shifts = shifts.reshape(shifts.shape[0],-1)
        print(samples.dtype, reals.dtype, shifts.shape)
        
        mae, mse, rmse, rmdspe, mape, _smape, _mase, q25, q75, crps = get_metric(reals, samples, shifts)
        print('mse:{:.3f}, mae:{:.3f}, rmse:{:.3f}, rmdspe:{:.3f}, mape:{:.3f}, smape:{:.3f}, mase:{:.3f}, Q25:{:.3f}, Q75:{:.3f}, crps:{:.3f}'.format(mse, mae, rmse, rmdspe, mape, _smape, _mase, q25, q75, crps))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(args.save_dir)
        f.write('mse:{:.3f}, mae:{:.3f}, rmse:{:.3f}, rmdspe:{:.3f}, mape:{:.3f}, smape:{:.3f}, mase:{:.3f}, Q25:{:.3f}, Q75:{:.3f}, crps:{:.3f}, '.format(mse, mae, rmse, rmdspe, mape, _smape, _mase, q25, q75, crps))
        f.write('\n')
        f.write('\n')
        f.close()
        
    else: ### generation
        print("Generating ...")
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.seq_len, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()
