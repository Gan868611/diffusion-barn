import sys
import os
sys.path.append('/jackal_ws/src/mlda-barn-2024/train_imitation')
sys.path.append('/jackal_ws/src/mlda-barn-2024/train_imitation/diffusion_policy/diffusion_policy')
import warnings
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import random
import shutil
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dataset.KULBarnDiffusionBackboneDataset import KULBarnDiffusionDataset
from model.diffusion_policy_model_backbone import DiffusionModel
from diffusers.optimization import get_cosine_schedule_with_warmup

# Load hyperparameters from config.yaml
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = OmegaConf.load(config_path)
np.random.seed(3)


no_worlds = config.no_worlds
train_ratio = config.train_ratio
horizon = config.horizon

print("Reading dataset........")
df = pd.read_csv(config.inspection_data)
print(df.head())

world_ids = [i for i in range(no_worlds)]
test_ids = [id for id in range(0, no_worlds, 5)]
non_test_ids = np.setdiff1d(world_ids, test_ids)
train_evals = [id for id in world_ids if id not in test_ids]
train_ids = np.random.choice(non_test_ids, int(train_ratio * len(non_test_ids)), replace=False)
val_ids = np.setdiff1d(non_test_ids, train_ids)

train_df = df[df['world_idx'].isin(train_ids)]
val_df = df[df['world_idx'].isin(val_ids)]

print("val id: ", val_ids)
print(len(train_ids))
print(len(val_ids))
print(len(test_ids))

train_dataset = KULBarnDiffusionDataset(train_df, horizon) #NFRAMES
val_dataset = KULBarnDiffusionDataset(val_df, horizon)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
normalizer = train_dataset.get_normalizer()
print(len(train_dataloader))

for batch in train_dataloader:
    print(batch['lidar_data'].shape)
    print(batch['non_lidar_data'].shape)
    print(batch['action'].shape)
    break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


model = DiffusionModel(config=config)
policy = model.policy

policy.set_normalizer(normalizer)
policy.to(device)

num_epochs = config.num_epochs
losses = []
mse_losses = []
save_loss_every = config.save_loss_every
validate_every = config.validate_every
total_loss = 0
count = 0

from datetime import datetime, timedelta
import pytz

singapore_tz = pytz.timezone('Asia/Singapore')
current_time = datetime.now(singapore_tz)
adjusted_time = current_time - timedelta(minutes=9)
timestamp = adjusted_time.strftime('%y%m%d_%H%M%S')

base_path = '/jackal_ws/src/mlda-barn-2024/outputs/diffusion_policies_backbone/'
dir_path = os.path.join(base_path, timestamp)
os.makedirs(dir_path, exist_ok=True)

config_dst = os.path.join(dir_path, 'config.yaml')
shutil.copyfile(config_path, config_dst)

optimizer = optim.Adam(list(policy.model.parameters()) + list(policy.cnn_model.parameters()), lr=config.learning_rate)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=num_epochs * len(train_dataloader))
policy.model.train()
for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = policy.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        count += 1
        if count >= save_loss_every:
            curr_loss = total_loss / save_loss_every
            # print("Loss:", curr_loss)
            losses.append(curr_loss)
            total_loss = 0
            count = 0

    if epoch % validate_every == 0:
        with torch.no_grad():
            total_mse_losses = 0
            for batch in tqdm(val_dataloader):
                obs_dict = {'lidar_data': batch['lidar_data'].to(device), 'non_lidar_data': batch['non_lidar_data'].to(device)}
                pred = policy.predict_action(obs_dict)['action_pred']  #[batch, horizon, action_dim]
                # print(pred.shape)
                mse_loss = F.mse_loss(pred[:,config.n_obs_steps - 1,:], batch['action'][:,config.n_obs_steps - 1,:].to(device))
                total_mse_losses += mse_loss.item()  
            mse_loss = total_mse_losses / len(val_dataloader)
            mse_losses.append(mse_loss)
            print("Val MSE Loss:", mse_loss)
    print("Epoch: ", epoch, "/",num_epochs)




# save losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.scatter(len(losses)-1, losses[-1], color='red', label=f"Last: {losses[-1]:.4f}")  # Highlight last point
plt.legend()
plt.savefig(dir_path + f'/diffuser_losses.png')
plt.clf()  # Clear the current figure

plt.plot(mse_losses)
plt.scatter(len(mse_losses)-1, mse_losses[-1], color='red', label=f"Last: {mse_losses[-1]:.4f}")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Val MSE Loss')
plt.legend()
plt.savefig(dir_path + f'/val_mse_losses.png')

# save the policy
torch.save({
    'cnn_model': policy.cnn_model.state_dict(),
    'model': policy.model.state_dict(),
    'normalizer': policy.normalizer.state_dict()
}, dir_path + '/diffusion_policies_model.pth')

print(f"Policy saved to {dir_path + '/diffusion_policies_backbone_model.pth'}")

