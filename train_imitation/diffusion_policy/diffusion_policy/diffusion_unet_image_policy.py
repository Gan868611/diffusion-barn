from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
import einops

torch.manual_seed(3)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CNNModel(nn.Module):
    def __init__(self, num_lidar_features, num_non_lidar_features, output_dim=32, nframes=1):
        super(CNNModel, self).__init__()
        self.output_dim = output_dim
        self.act_fea_cv1 = nn.Conv1d(
            in_channels=nframes, out_channels=32, kernel_size=5, stride=2, padding=2, padding_mode='circular'
        )
        self.act_fea_cv2 = nn.Conv1d(
            in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, padding_mode='circular'
        )

        # conv_output_size = (num_lidar_features - 5 + 2*6) // 2 + 1  # Output size after self.act_fea_cv1
        # conv_output_size = (conv_output_size - 3 + 2*1) // 2 + 1  # Output size after self.act_fea_cv2
        # conv_output_size *= 32  # Multiply by the number of output channels
        with torch.no_grad():
            sample_input = torch.randn(1, nframes, num_lidar_features)
            sample_output = self.act_fea_cv1(sample_input)
            sample_output = self.act_fea_cv2(sample_output)
            conv_output_size = sample_output.view(1, -1).shape[1]

        # Calculate the output size of the CNN
        self.fc1 = nn.Linear(conv_output_size + num_non_lidar_features*nframes, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.norm = nn.LayerNorm(output_dim)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, obs):
        obs = obs['obs']
        lidar = obs[:, :360]
        lidar = lidar.unsqueeze(1)
        non_lidar = obs[:, 360:]
        non_lidar = non_lidar.unsqueeze(1)
        # print("check2:",non_lidar[0,0])
        lidar_batch_size = lidar.shape[:-2]
        non_lidar_batch_size = non_lidar.shape[:-2]
        if len(lidar.shape) > 3:
            if len(lidar.shape) == 4:
                # print('reach here')
                lidar = einops.rearrange(lidar, 'b n c l -> (b n) c l')
        # lidar = lidar.unsqueeze(1)  # Add channel dimension
        
        feat = F.relu(self.act_fea_cv1(lidar))
        feat = F.relu(self.act_fea_cv2(feat))
        
        feat = feat.view(feat.shape[0], -1)
        # print("feat shape: ", feat.shape)
        # print("non_lidar shape: ", non_lidar.shape)
        # print("non_lidar shape: ",  non_lidar.view(-1, non_lidar.shape[-1]*non_lidar.shape[-2]).shape)
        feat = torch.cat((feat, non_lidar.view(-1, non_lidar.shape[-1]*non_lidar.shape[-2])), dim=-1)
        feat = F.relu(self.fc1(feat))
        feat = self.fc2(feat)
        # feat = self.norm(feat)
        # print(feat.shape)
        feat = einops.rearrange(feat, '(b n) d -> b n d', b=lidar_batch_size[0])
        # print(feat.shape)
        return feat


class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            noise_scheduler: DDPMScheduler,
            obs_encoder: CNNModel,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            obs_dim, 
            action_dim, 
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            dims_multiplier=128,
            down_dims=(2,4,8),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # get feature dim
        obs_feature_dim = obs_dim

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            dims_multiplier=dims_multiplier,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        #TODO
        total_param = sum(p.numel() for p in obs_encoder.parameters())
        print(f"CNN parameters: {total_param}")
        total_params = sum(p.numel() for p in model.parameters()) + total_param
        print(f"Total parameters: {total_params}") # current param : 43,183,458 for dims_mul = 128

        self.obs_encoder = obs_encoder
        self.model = model
        self.cnn_model = obs_encoder
        self.noise_scheduler = noise_scheduler
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
        self.prev_trajectory = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        if self.prev_trajectory != None:
            noise = torch.randn(trajectory.shape, device=trajectory.device)
            trajectory = self.prev_trajectory.clone()
            # print("traj shape: ",trajectory.shape)
            trajectory[:,:-1,:] = self.prev_trajectory[:,1:,:].clone() # [1,2,3,4] -> [2,3,4,4]
            # print("traj shape: ",trajectory.shape)
            # print(self.num_inference_steps, torch.IntTensor(self.num_inference_steps).long())

            trajectory = self.noise_scheduler.add_noise(
                trajectory, noise, torch.tensor(self.num_inference_steps).long())

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]     
        #prev_traj
        # self.prev_trajectory = trajectory.clone()       

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet

        # normalize input
        nlidar_data = self.normalizer['lidar_data'].normalize(obs_dict['lidar_data'])
        nnonlidar_data = self.normalizer['non_lidar_data'].normalize(obs_dict['non_lidar_data'])
        # batch, horizon, 360+4
        nobs = {'obs': torch.cat([nlidar_data, nnonlidar_data], dim=-1)}
        # nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            # print(this_nobs['obs'].shape)
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        # batch , horizon, action_dim
        naction_pred = nsample[...,:Da]
        # print("naction_pred: ", naction_pred.shape)
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        #nobs_step - 1
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        # print("start", start, 'end',end,action.shape  )
        
        result = {
            'action': action, #slice of action_pred
            'action_pred': action_pred
        }
        # print("action:", action.shape, action_pred.shape)
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        # print(self.normalizer)

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nlidar_data = self.normalizer['lidar_data'].normalize(batch['lidar_data'])
        nnonlidar_data = self.normalizer['non_lidar_data'].normalize(batch['non_lidar_data'])
        # batch, horizon, 360+4
        nobs = {'obs': torch.cat([nlidar_data, nnonlidar_data], dim=-1)}
        # print(nobs['obs'][0][0][-4])
        # print("nobs:", nobs['obs'].shape)

        nactions = self.normalizer['action'].normalize(batch['action'])
        # print("nactions:", nactions.shape)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            # print(this_nobs['obs'].shape)
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        # print("noisy_traj", noisy_trajectory.shape)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        # print("pred_typpe:", pred_type)
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
    
        # print("pred:", pred.shape, " target:", target.shape)

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
