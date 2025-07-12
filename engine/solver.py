import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader=None, logger=None):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_epochs = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader']) if dataloader else None
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(self.model.parameters(), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train_bak(self, mode=None):
        if self.dl is None:
            raise TypeError('Please provides dataloader')
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_epochs) as pbar:
            while step < self.train_epochs:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data, target = next(self.dl)
                    loss = self.model(data.to(device), target=target.to(device) if mode in 'predict' else None)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()

                pbar.set_description(f'loss: {total_loss:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)

        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))


    def train(self, mode=None):
        if self.dl is None:
            raise TypeError('Please provides dataloader')
        print('training mode => ', 'predict' if mode in 'predict' else 'infill')
        device = self.device
        
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)
        time_now = time.time()
        for epoch in range(self.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, mask) in enumerate(self.dl):
                iter_count += 1
                self.opt.zero_grad()
                x = batch_x.float().to(self.device)
                y = batch_y.float().to(self.device) if self.args.mode in 'predict' else batch_x.float().to(self.device)
                if self.args.mode in 'infill':
                    x = x * mask.to(self.device)
            
                loss, outputs = self.model(x.to(device), target=y.to(device) if mode in 'predict' else None)
                loss.backward()
                train_loss.append(loss.item())

                self.opt.step()
                self.ema.update()

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.train_epochs - epoch) * iter_count - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        self.logger.add_scalar(tag='train/loss', scalar_value=np.mean(train_loss), global_step=self.step)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            #vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Test Loss: {3:.7f}".format(
            #     epoch + 1, train_steps, train_loss, test_loss))
            # #early_stopping(vali_loss, self.model, path)
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break

            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            with torch.no_grad():
                if self.step != 0 and self.step % self.save_cycle == 0:
                    self.milestone += 1
                    self.save(self.milestone)
                if self.logger is not None and self.step % self.log_frequency == 0:
                    self.logger.add_scalar(tag='train/loss', scalar_value=np.mean(train_loss), global_step=self.step)


        print('training complete')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples
                       
    def inference(self, raw_dataloader, shape=None, mode=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = []
        reals = []
        masks = []
        shifts = []
        inputs = []

        for idx, (x, y, t_m) in enumerate(raw_dataloader):
            x, y, t_m = x.to(self.device), y.to(self.device), t_m.to(self.device)
            if sampling_steps == self.model.num_timesteps:
                sample = self.ema.ema_model.sample_infill(x, target=x*t_m, partial_mask=t_m, mode = mode, 
                                                          model_kwargs=model_kwargs)
            else:
                sample = self.ema.ema_model.fast_sample_infill(x, target=x*t_m, sampling_timesteps=sampling_steps,  partial_mask=t_m, mode = mode,  model_kwargs=model_kwargs)

            sample = sample.detach().cpu().numpy()
            real = y.detach().cpu().numpy()
            samples.append(sample)
            reals.append(real if mode in 'predict' else x.detach().cpu().numpy())
            masks.append(t_m.detach().cpu().numpy())
            shifts.append(t_m[:,-1:,:].repeat(1, t_m.shape[1], 1).detach().cpu().numpy())
            inputs.append(x.detach().cpu().numpy())
            
            if idx % 20 == 0:
                input = x.detach().cpu().numpy()
                gt = np.concatenate((input[0, :, -1], real[0, :, -1]), axis=0)
                pd = np.concatenate((input[0, :, -1], sample[0, :, -1]), axis=0)
                self.visual(gt, pd, os.path.join(self.results_folder, str(idx) + '.pdf'))
        
        samples = np.vstack(samples)
        reals = np.vstack(reals)
        masks = np.vstack(masks)
        shifts = np.vstack(shifts)
        inputs = np.vstack(inputs)
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, shifts, masks.astype(np.bool8), inputs
    
    def predict(self, raw_dataloader, shape=None, mode=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = []
        reals = []
        masks = []
        shifts = []
        inputs = []

        for idx, (x, y, t_m) in enumerate(raw_dataloader):
            x, y, t_m = x.to(self.device), y.to(self.device), t_m.to(self.device)

            sample = self.ema.ema_model.predict(x*t_m)

            samples.append(sample.detach().cpu().numpy())
            reals.append(y.detach().cpu().numpy() if mode in 'predict' else x.detach().cpu().numpy())
            masks.append(t_m.detach().cpu().numpy())
            shifts.append(t_m[:,-1:,:].repeat(1, t_m.shape[1], 1).detach().cpu().numpy())
            inputs.append(x.detach().cpu().numpy())
        
        samples = np.vstack(samples)
        reals = np.vstack(reals)
        masks = np.vstack(masks)
        shifts = np.vstack(shifts)
        inputs = np.vstack(inputs)
        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, shifts, masks.astype(np.bool8), inputs
    
    

    def visual(self, true, preds=None, name='./pic/test.pdf'):
        """
        Results visualization
        """
        plt.figure()
        plt.plot(true, label='GroundTruth', linewidth=2)
        if preds is not None:
            plt.plot(preds, label='Prediction', linewidth=2)
        plt.legend()
        plt.savefig(name, bbox_inches='tight')