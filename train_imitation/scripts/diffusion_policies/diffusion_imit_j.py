import warnings
import pandas as pd

warnings.filterwarnings('ignore')

import torch
# from torch.utils.data import Dataset
import numpy as np
# from sklearn.preprocessing import MinMaxScaler
from diffusers.optimization import get_cosine_schedule_with_warmup

import random
# set random seed
random.seed(42)

# import os
# import hydra
import torch
from omegaconf import OmegaConf
# import pathlib
from torch.utils.data import DataLoader
# import copy
import numpy as np
import random
# import wandb
import tqdm
# import shutil
# from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
# from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import TrainDiffusionUnetLowdimWorkspace
# import os


from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from typing import Dict
import torch
import numpy as np
import copy
# from diffusion_policy.common.pytorch_util import dict_apply
# from diffusion_policy.common.replay_buffer import ReplayBuffer
# from diffusion_policy.common.sampler import (
    # SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset



NO_WORLDS = 300
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
NFRAMES = 4

df = pd.read_csv('/jackal_ws/src/mlda-barn-2024/kul_data_10Hz_done.csv')
print(df.head())

world_ids = [i for i in range(NO_WORLDS)]
test_ids = [id for id in range(0, NO_WORLDS, 5)]
train_evals = [id for id in world_ids if id not in test_ids]
train_ids = random.sample(train_evals, int(NO_WORLDS * TRAIN_RATIO))
val_ids = [id for id in train_evals if id not in train_ids]

train_df = df[df['world_idx'].isin(train_ids)]
val_df = df[df['world_idx'].isin(val_ids)]

print(len(train_ids))
print(len(val_ids))
print(len(test_ids))

train_dataset = KULBarnDiffusionDataset(train_df, NFRAMES)
val_dataset = KULBarnDiffusionDataset(val_df, NFRAMES)
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
normalizer = train_dataset.get_normalizer()
print(len(train_dataloader))

for batch in train_dataloader:
    print(batch['lidar_data'].shape)
    print(batch['non_lidar_data'].shape)
    print(batch['action'].shape)
    break

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.diffusion_unet_lowdim_policy_with_cnn1d_jd import DiffusionUnetLowdimPolicyWithCNN1D, CNNModel

lidar_dim = batch['lidar_data'].shape[-1]
non_lidar_dim = batch['non_lidar_data'].shape[-1]
cnn_model = CNNModel(num_lidar_features=lidar_dim, num_non_lidar_features=non_lidar_dim, nframes=NFRAMES)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
cnn_model.to(device)
obs_dim = cnn_model.output_dim
action_dim = batch['action'].shape[-1]
input_dim = obs_dim + action_dim
model = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim)
noise_scheduler = DDPMScheduler(num_train_timesteps=20, beta_schedule='linear')
policy = DiffusionUnetLowdimPolicyWithCNN1D(
    cnn_model=cnn_model,
    model=model, 
    noise_scheduler=noise_scheduler, 
    horizon=NFRAMES, 
    obs_dim=obs_dim, 
    action_dim=action_dim, 
    n_obs_steps=1,
    n_action_steps=4,
    obs_as_global_cond=True,
    oa_step_convention=True,
)

policy.set_normalizer(normalizer)
policy.to(device)

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

NUM_EPOCHS = 20
losses = []
mse_losses = []
save_loss_every = 10
total_loss = 0
count = 0

# optimizer = optim.Adam(policy.model.parameters(), lr=5e-5)
optimizer = optim.Adam(list(policy.model.parameters()) + list(policy.cnn_model.parameters()), lr=5e-5)
# optimizer = optim.Adam(list(policy.model.parameters()) + list(policy.cnn_model.parameters()), lr=1e-4)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=NUM_EPOCHS * len(train_dataloader))
policy.model.train()
for epoch in range(NUM_EPOCHS):
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        # batch['obs'] = cnn_model(batch['lidar_data'], batch['non_lidar_data'])
        loss = policy.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        # print(policy.cnn_model.act_fea_cv1.weight.grad)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        count += 1
        if count >= save_loss_every:
            curr_loss = total_loss / save_loss_every
            print("Loss:", curr_loss)
            losses.append(curr_loss)
            total_loss = 0
            count = 0

    with torch.no_grad():
        total_mse_losses = 0
        for batch in tqdm(val_dataloader):
            obs_dict = {'lidar_data': batch['lidar_data'].to(device), 'non_lidar_data': batch['non_lidar_data'].to(device)}
            pred = policy.predict_action(obs_dict)['action_pred']
            mse_loss = F.mse_loss(pred, batch['action'].to(device))
            total_mse_losses += mse_loss.item()  
        mse_loss = total_mse_losses / len(val_dataloader)
        mse_losses.append(mse_loss)
        print("Val MSE Loss:", mse_loss)
    print("Epoch: ", epoch, "/",NUM_EPOCHS)

# save losses
import matplotlib.pyplot as plt

suffix = "diffuser_policy_10Hz_backbone_diffusion_steps_20"
# save losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
save_path = f'diffuser_losses_{suffix}.png'
plt.savefig(save_path)
plt.clf()  # Clear the current figure

plt.plot(mse_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Val MSE Loss')
save_path = f'val_mse_losses_{suffix}.png'
plt.savefig(save_path)

# save the policy
save_path = f'/jackal_ws/src/mlda-barn-2024/train_imitation/diffusion_policy/{suffix}.pth'
torch.save({
    'cnn_model': policy.cnn_model.state_dict(),
    'model': policy.model.state_dict(),
    'normalizer': policy.normalizer.state_dict()
}, save_path)

print(f"Policy saved to {save_path}")