from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

# added
import numpy as np
from streaming_flow_policy.trainning_data_utils import get_total_xt_ut_ot

class SFPUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            unit_int_steps = 1, #added
            k = 0, #added for stablizing flow
            use_action_traj=False, #added
            robomimic=False, #added
            biased_gripper = False, #added
            gripper_velocity = False, #added
            gripper_no_noise = False, #added
            gripper_normalize = 1, #added
            biased_prob = 0.9, #added
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim #+ obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            sin_embedding_scale = 100, #added sfp
            linear_updownsample = True, #added sfp
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        # SFP added
        self.k = k #stablizing flow parameter
        # intergration steps and timesteps
        self.unit_int_steps = unit_int_steps
        self.num_inference_steps = num_inference_steps
        max_time = n_action_steps/horizon #action horizon/ prediction horizon
        total_int_steps = n_action_steps * self.unit_int_steps + 1
        self.t_span = torch.linspace(0, max_time, total_int_steps)
        if use_action_traj:
            self.select_action_indices = np.arange(0, total_int_steps-1, unit_int_steps) #[0, 1, ..., 7]
        else:
            self.select_action_indices = np.arange(unit_int_steps, total_int_steps, unit_int_steps) #[1, 2, ..., 8]

        # added
        self.use_action_traj = use_action_traj

        print('---------- Model Original -------------------')
        print('self.horizon', self.horizon)
        print('use_action_traj', self.use_action_traj)
        print('t span', self.t_span)
        print('np.arange', 0, total_int_steps-1, self.unit_int_steps)
        print('select_action_indices', self.select_action_indices)
        # added
        self.robomimic = robomimic
        self.gripper_velocity = gripper_velocity
        self.biased_gripper = biased_gripper
        self.gripper_no_noise = gripper_no_noise
        self.gripper_normalize = gripper_normalize
        self.biased_prob = biased_prob

    def set_n_action_int_steps(self, set_action_steps, set_int_steps):
        '''
        Set the number of action steps and update the time span and select action indices accordingly.
        '''
        self.unit_int_steps = int(set_int_steps)
        self.n_action_steps = int(set_action_steps)
        max_time = self.n_action_steps/self.horizon #action horizon/ prediction horizon
        total_int_steps = self.n_action_steps * self.unit_int_steps +1 # fixed, previously did not multiply by unit_int_steps
        self.t_span = torch.linspace(0, max_time, total_int_steps)
        
        if self.use_action_traj:
            print(0, total_int_steps-1, self.unit_int_steps)
            self.select_action_indices = np.arange(0, total_int_steps-1, self.unit_int_steps) #[0, 1, ..., 7]
        else:
            self.select_action_indices = np.arange(self.unit_int_steps, total_int_steps, self.unit_int_steps) #[1, 2, ..., 8]
        print('---------- Currently Set -------------------')
        print('self.horizon', self.horizon)
        print('t span', self.t_span)
        print('select_action_indices', self.select_action_indices)
    
    # ========= inference  ============
    def get_traj(self,
                 global_cond, #dictionary
                 nobs, 
                 obs,
                 prev_action = None,
                **kwargs
            ):
            '''
            nobs: {
                'image': torch.Size([56, 2, 3, 96, 96]),
                'agent_pos': torch.Size([56, 2, 2])
            }
            '''
            noise_pred_net = self.model
            if not self.use_action_traj:
                x_test = nobs['agent_pos'][:,-1:,:] # agent position nobs (56, 2, 2) -> x_test (56, 1, 2)
            elif prev_action is None:
                if self.robomimic:
                    pass ### TODO: add robomimic obs_to_action
                    # unnorm_x_test = self.robomimic_obs_to_action(obs[:,-1:,:]) #28, 1, 10
                else:
                    unnorm_x_test = obs['agent_pos'][:,-1:,:] 
                x_test = self.normalizer['action'].normalize(unnorm_x_test) #normalize with action normalizer
            else:
                x_test = prev_action 

            # manual integration
            traj_list = []
            prev_x = x_test
            with torch.no_grad():
                for t in self.t_span:
                    traj_list.append(prev_x)
                    pred_v =noise_pred_net(sample=prev_x,timestep=t.repeat(x_test.shape[0]).to("cuda"), global_cond=global_cond) #[56, 1, 2] # previously had global_cond.flatten(start_dim=1) -> not needed
                    prev_x = prev_x + pred_v * 1/(self.horizon * self.unit_int_steps) # [56, 1, 2]
                    prev_x[:, :, -1] = prev_x[:, :, -1].clamp(0, 0.08)
                    ## TODO: check robomimic code
                    # if self.robomimic: # clamp for robomimic
                    #     prev_x[:, :, -1] = prev_x[:, :, -1].clamp(-1.0, 1.0) # clamp the gripper value prev_x[0, :, -1] -> dim 2, -1 idx
            traj = torch.stack(traj_list, dim=0)  

            return traj #[9, 56, 1, 2]
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], prev_action = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # {
        # 'image': torch.Size([56, 2, 3, 96, 96]),
        # 'agent_pos': torch.Size([56, 2, 2])
        # }
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        To = self.n_obs_steps


        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) #torch.Size([112, 3, 96, 96]) torch.Size([112, 2])
        nobs_features = self.obs_encoder(this_nobs) # [112, 66]
        # reshape back to B, Do
        global_cond = nobs_features.reshape(B, -1) #[56, 132]
        
        naction_pred = self.get_traj(global_cond, nobs, obs_dict, prev_action) #[8, 56, 1, 2]
        # print('naction_pred', naction_pred[:,0,...])
        # use the last two dim of obs normalizer s.t. it's consistent with training normalization
        if self.use_action_traj:
            action_pred = self.normalizer['action'].unnormalize(naction_pred) # [9, 56, 1, 2]
        else:
            action_pred = self.normalizer['agent_pos'].unnormalize(naction_pred, # agent_pos [8, 56, 1, 2]
                        obs_for_action=True, action_idx_start = -self.action_dim) # Push T: action_idx_start = -2
        prev_action = naction_pred[-1] # save a_8, last action as memory var [9, 56, 1, 2] -> #[56, 1, 2]
        action = action_pred[self.select_action_indices] # [9, 56, 1, 2] -> [8, 56, 1, 2] select action indices - for use_action_traj case, select first 8
        action  = action.permute(1, 0, 2, 3).squeeze(2) #[56, 8, 2] 
        # print('action', naction_pred[self.select_action_indices][:, 0, 0,...])
        # print('prev_action', prev_action[0,...])
        result = {
            'action': action,
            'action_pred': action_pred,
            'prev_action': prev_action, 
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch, x_t_sigma):
        # normalize input
        assert 'valid_mask' not in batch
        nactions = self.normalizer['action'].normalize(batch['action'])
        nobs = self.normalizer.normalize(batch['obs'])
        # [64, 16, 3, 96, 96]) torch.Size([64, 16, 2]
        batch_size = nactions.shape[0]
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs) #128, 66]
        # reshape back to B, Do
        global_cond = nobs_features.reshape(batch_size, -1) #[64, 132]
        if self.use_action_traj:
            action_traj = nactions
        else:
            action_traj = nobs['agent_pos'][:, 1:, :] #[64, 18, 2] -> [64, 17, 2] #action_traj = obs[:,1:,-2:] #[256, 17, 2]

        # sample a random time step in Uniform(0, 1)
        t = torch.rand(action_traj.shape[0]).float().to(self.device)
        # sample q(t) ~ N(q̃(t), σ₀ exp(-kt))
        # calculate velosity at time t: u = -k * (qt - q̃t) + ṽt 
        # print('action_traj', action_traj.shape, 't', t.shape, 'T', self.horizon)
        xt, ut = get_total_xt_ut_ot(action_traj, t = t, T = self.horizon, k = self.k,
                                    sigma = x_t_sigma, device=self.device,
                                    #### TODO: robomimic check
                                    # gripper_no_noise = self.gripper_no_noise, gripper_normalize = self.gripper_normalize,
                                    # biased_prob = self.biased_prob, 
                                    # biased_gripper = self.biased_gripper, gripper_velocity = self.gripper_velocity
                                        ) # xt shape: [256, 2], (batch_size, action_dim)
        xt, ut = xt.unsqueeze(1), ut.unsqueeze(1) #xt: (batch_size, pred_horizon, action_dim)[256, 1, 2]; ut: same as xt
        # print('xt', xt.shape, 'ut', ut.shape)
        # predict the vector field from the model
        noise_pred_net = self.model
        vt = noise_pred_net(sample=xt,timestep=t, global_cond = global_cond)

        # L2 loss
        loss = nn.functional.mse_loss(vt, ut)
        return loss