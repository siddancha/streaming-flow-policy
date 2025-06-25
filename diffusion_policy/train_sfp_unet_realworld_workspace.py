if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import torch.distributed as dist #aded for two gpus
from torch.utils.data import DataLoader, DistributedSampler # added for two gpus
import shutil
from diffusion_policy.base_workspace import BaseWorkspace
from diffusion_policy.policy.sfp_unet_keypoint_policy import SFPUnetKeypointPolicy
from diffusion_policy.policy.sfp_unet_hybrid_image_policy import SFPUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def debug_device_assignment():
    rank = dist.get_rank() if dist.is_initialized() else 0
    # If you’re using torchrun or launch, LOCAL_RANK *must* be set:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    print(f"[rank {rank}] LOCAL_RANK={local_rank}  →  torch.cuda.current_device()={torch.cuda.current_device()}")

OmegaConf.register_new_resolver("eval", eval, replace=True)
class TrainSFPRealworldWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.ngpus = torch.cuda.device_count()
        print(f"Found {self.ngpus} GPUs")

        # added for sfp
        self.x_t_sigma = cfg.training.x_t_sigma
        self.multi_gpu = cfg.policy.multi_gpu
        
        # configure model
        self.model: SFPUnetKeypointPolicy = hydra.utils.instantiate(cfg.policy)

        print('gripper_normalize', self.model.gripper_normalize)

        self.ema_model: SFPUnetKeypointPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        debug_device_assignment()
        cfg = copy.deepcopy(self.cfg)
        print('---------------------Running SFP training---------------------')

        if self.multi_gpu:
        #     # 1) Initialize the default process group
        #     print("--- Initializing distributed process group ---")
        #     dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            if local_rank != 0: os.environ["WANDB_MODE"] = "disabled"
        #     torch.cuda.set_device(local_rank)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        
        
        if self.multi_gpu: 
            train_sampler = DistributedSampler(dataset, shuffle=True)
            train_dataloader = DataLoader(dataset, sampler=train_sampler, **cfg.dataloader)
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        if self.multi_gpu:
            # get your rank
            rank = dist.get_rank()
            # how many batches this process will see each epoch
            num_batches = len(train_dataloader)
            print(f"[rank {rank}] will process {num_batches} batches per epoch")

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        if self.multi_gpu: 
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **cfg.val_dataloader)
        else:
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     k_p_scale=1, # use k_p_scale=5 for SSFP
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)
        print('wandb', cfg.logging.name)
        # configure logging
        if not self.multi_gpu or is_main_process():
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True # added for resuming
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        if self.multi_gpu: 
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                if self.epoch>cfg.training.max_epochs: # only run max_epochs
                    break
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                if self.multi_gpu: train_sampler.set_epoch(self.epoch)
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch, self.x_t_sigma)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if not self.multi_gpu or is_main_process():
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)

                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy) #use k_p_scale=5 for SSFP
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    if self.multi_gpu: val_sampler.set_epoch(self.epoch)
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch, self.x_t_sigma)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                # if (self.epoch % cfg.training.sample_every) == 0:
                #     with torch.no_grad():
                #         # sample trajectory from training set, and evaluate difference
                #         batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                #         obs_dict = batch['obs']
                #         gt_action = batch['action']
                        
                #         result = policy.predict_action(obs_dict)
                #         pred_action = result['action_pred']
                #         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                #         step_log['train_action_mse_error'] = mse.item()
                #         del batch
                #         del obs_dict
                #         del gt_action
                #         del result
                #         del pred_action
                #         del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None and (not self.multi_gpu or is_main_process()):
                        self.save_checkpoint(path=topk_ckpt_path)
                    # save last 10 epochs
                    if cfg.training.num_epochs-self.epoch < 10*cfg.training.checkpoint_every:
                        self.save_checkpoint(tag = f'last-epoch={self.epoch:03d}.ckpt')
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if dist.get_rank() == 0: wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

# class TrainSFPImageRealworldWorkspace(BaseWorkspace):
#     include_keys = ['global_step', 'epoch']

#     def __init__(self, cfg: OmegaConf, output_dir=None):
#         super().__init__(cfg, output_dir=output_dir)

#         # set seed
#         seed = cfg.training.seed
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         random.seed(seed)

#         # added for sfp
#         self.x_t_sigma = cfg.training.x_t_sigma
        
#         # configure model
#         self.model: SFPUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

#         print('gripper_normalize', self.model.gripper_normalize)

#         self.ema_model: SFPUnetHybridImagePolicy = None
#         if cfg.training.use_ema:
#             self.ema_model = copy.deepcopy(self.model)

#         # configure training state
#         self.optimizer = hydra.utils.instantiate(
#             cfg.optimizer, params=self.model.parameters())

#         # configure training state
#         self.global_step = 0
#         self.epoch = 0

#     def run(self):
#         cfg = copy.deepcopy(self.cfg)
#         if MULTI_GPU:
#             # 1) Initialize the default process group
#             dist.init_process_group(backend="nccl")
#             local_rank = int(os.environ["LOCAL_RANK"])
#             torch.cuda.set_device(local_rank)
#         # resume training
#         if cfg.training.resume:
#             lastest_ckpt_path = self.get_checkpoint_path()
#             if lastest_ckpt_path.is_file():
#                 print(f"Resuming from checkpoint {lastest_ckpt_path}")
#                 self.load_checkpoint(path=lastest_ckpt_path)

#         # configure dataset
#         dataset: BaseImageDataset
#         dataset = hydra.utils.instantiate(cfg.task.dataset)
#         train_sampler = DistributedSampler(dataset)
#         assert isinstance(dataset, BaseImageDataset)
#         train_dataloader = DataLoader(dataset, sampler=train_sampler, **cfg.dataloader)
#         normalizer = dataset.get_normalizer()

#         # configure validation dataset
#         val_dataset = dataset.get_validation_dataset()
#         val_sampler = DistributedSampler(val_dataset)
#         val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **cfg.val_dataloader)

#         self.model.set_normalizer(normalizer)
#         if cfg.training.use_ema:
#             self.ema_model.set_normalizer(normalizer)

#         # configure lr scheduler
#         lr_scheduler = get_scheduler(
#             cfg.training.lr_scheduler,
#             optimizer=self.optimizer,
#             num_warmup_steps=cfg.training.lr_warmup_steps,
#             num_training_steps=(
#                 len(train_dataloader) * cfg.training.num_epochs) \
#                     // cfg.training.gradient_accumulate_every,
#             # pytorch assumes stepping LRScheduler every epoch
#             # however huggingface diffusers steps it every batch
#             last_epoch=self.global_step-1
#         )

#         # configure ema
#         ema: EMAModel = None
#         if cfg.training.use_ema:
#             ema = hydra.utils.instantiate(
#                 cfg.ema,
#                 model=self.ema_model)

#         # configure env
#         # env_runner: BaseImageRunner
#         # env_runner = hydra.utils.instantiate(
#         #     cfg.task.env_runner,
#         #     k_p_scale=1, # use k_p_scale=5 for SSFP
#         #     output_dir=self.output_dir)
#         # assert isinstance(env_runner, BaseImageRunner)
#         print('wandb', cfg.logging.name)
#         # configure logging
#         wandb_run = wandb.init(
#             dir=str(self.output_dir),
#             config=OmegaConf.to_container(cfg, resolve=True),
#             **cfg.logging
#         )
#         wandb.config.update(
#             {
#                 "output_dir": self.output_dir,
#             },
#             allow_val_change=True # added for resuming
#         )

#         # configure checkpoint
#         topk_manager = TopKCheckpointManager(
#             save_dir=os.path.join(self.output_dir, 'checkpoints'),
#             **cfg.checkpoint.topk
#         )

#         # device transfer
#         device = torch.device(cfg.training.device)
#         self.model.to(device)
#         if self.ema_model is not None:
#             self.ema_model.to(device)
#         optimizer_to(self.optimizer, device)

#         # save batch for sampling
#         train_sampling_batch = None

#         if cfg.training.debug:
#             cfg.training.num_epochs = 2
#             cfg.training.max_train_steps = 3
#             cfg.training.max_val_steps = 3
#             cfg.training.rollout_every = 1
#             cfg.training.checkpoint_every = 1
#             cfg.training.val_every = 1
#             cfg.training.sample_every = 1

#         # training loop
#         log_path = os.path.join(self.output_dir, 'logs.json.txt')
#         with JsonLogger(log_path) as json_logger:
#             for local_epoch_idx in range(cfg.training.num_epochs):
#                 if self.epoch>cfg.training.max_epochs: # only run max_epochs
#                     break
#                 step_log = dict()
#                 # ========= train for this epoch ==========
#                 train_losses = list()
#                 train_sampler.set_epoch.set_epoch(self.epoch)
#                 with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
#                         leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
#                     for batch_idx, batch in enumerate(tepoch):
#                         # device transfer
#                         batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
#                         if train_sampling_batch is None:
#                             train_sampling_batch = batch

#                         # compute loss
#                         raw_loss = self.model.compute_loss(batch, self.x_t_sigma)
#                         loss = raw_loss / cfg.training.gradient_accumulate_every
#                         loss.backward()

#                         # step optimizer
#                         if self.global_step % cfg.training.gradient_accumulate_every == 0:
#                             self.optimizer.step()
#                             self.optimizer.zero_grad()
#                             lr_scheduler.step()
                        
#                         # update ema
#                         if cfg.training.use_ema:
#                             ema.step(self.model)

#                         # logging
#                         raw_loss_cpu = raw_loss.item()
#                         tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
#                         train_losses.append(raw_loss_cpu)
#                         step_log = {
#                             'train_loss': raw_loss_cpu,
#                             'global_step': self.global_step,
#                             'epoch': self.epoch,
#                             'lr': lr_scheduler.get_last_lr()[0]
#                         }

#                         is_last_batch = (batch_idx == (len(train_dataloader)-1))
#                         if not is_last_batch:
#                             # log of last step is combined with validation and rollout
#                             wandb_run.log(step_log, step=self.global_step)
#                             json_logger.log(step_log)
#                             self.global_step += 1

#                         if (cfg.training.max_train_steps is not None) \
#                             and batch_idx >= (cfg.training.max_train_steps-1):
#                             break

#                 # at the end of each epoch
#                 # replace train_loss with epoch average
#                 train_loss = np.mean(train_losses)
#                 step_log['train_loss'] = train_loss

#                 # ========= eval for this epoch ==========
#                 policy = self.model
#                 if cfg.training.use_ema:
#                     policy = self.ema_model
#                 policy.eval()

#                 # run rollout
#                 # if (self.epoch % cfg.training.rollout_every) == 0:
#                 #     runner_log = env_runner.run(policy) #use k_p_scale=5 for SSFP
#                 #     # log all
#                 #     step_log.update(runner_log)

#                 # run validation
#                 if (self.epoch % cfg.training.val_every) == 0:
#                     val_sampler.set_epoch(self.epoch)
#                     with torch.no_grad():
#                         val_losses = list()
#                         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
#                                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
#                             for batch_idx, batch in enumerate(tepoch):
#                                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
#                                 loss = self.model.compute_loss(batch, self.x_t_sigma)
#                                 val_losses.append(loss)
#                                 if (cfg.training.max_val_steps is not None) \
#                                     and batch_idx >= (cfg.training.max_val_steps-1):
#                                     break
#                         if len(val_losses) > 0:
#                             val_loss = torch.mean(torch.tensor(val_losses)).item()
#                             # log epoch average validation loss
#                             step_log['val_loss'] = val_loss

#                 # run diffusion sampling on a training batch
#                 # if (self.epoch % cfg.training.sample_every) == 0:
#                 #     with torch.no_grad():
#                 #         # sample trajectory from training set, and evaluate difference
#                 #         batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
#                 #         obs_dict = batch['obs']
#                 #         gt_action = batch['action']
                        
#                 #         result = policy.predict_action(obs_dict)
#                 #         pred_action = result['action_pred']
#                 #         mse = torch.nn.functional.mse_loss(pred_action, gt_action)
#                 #         step_log['train_action_mse_error'] = mse.item()
#                 #         del batch
#                 #         del obs_dict
#                 #         del gt_action
#                 #         del result
#                 #         del pred_action
#                 #         del mse
                
#                 # checkpoint
#                 if (self.epoch % cfg.training.checkpoint_every) == 0:
#                     # checkpointing
#                     if cfg.checkpoint.save_last_ckpt:
#                         self.save_checkpoint()
#                     if cfg.checkpoint.save_last_snapshot:
#                         self.save_snapshot()

#                     # sanitize metric names
#                     metric_dict = dict()
#                     for key, value in step_log.items():
#                         new_key = key.replace('/', '_')
#                         metric_dict[new_key] = value
                    
#                     # We can't copy the last checkpoint here
#                     # since save_checkpoint uses threads.
#                     # therefore at this point the file might have been empty!
#                     topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

#                     if topk_ckpt_path is not None:
#                         self.save_checkpoint(path=topk_ckpt_path)
#                     # save last 10 epochs
#                     if cfg.training.num_epochs-self.epoch < 10*cfg.training.checkpoint_every:
#                         self.save_checkpoint(tag = f'last-epoch={self.epoch:03d}.ckpt')
#                 # ========= eval end for this epoch ==========
#                 policy.train()

#                 # end of epoch
#                 # log of last step is combined with validation and rollout
#                 wandb_run.log(step_log, step=self.global_step)
#                 json_logger.log(step_log)
#                 self.global_step += 1
#                 self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()