import sys
import os
sys.path.append('/jackal_ws/src/mlda-barn-2024/train_imitation')
import warnings
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
import random
# set random seed
random.seed(42)

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import random
import shutil
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dataset.KULBarnDiffusionDataset import KULBarnDiffusionDataset
from model.diffusion_policy_model import DiffusionModel

# Load hyperparameters from config.yaml
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = OmegaConf.load(config_path)


no_worlds = config.no_worlds
train_ratio = config.train_ratio
val_ratio = config.val_ratio
test_ratio = config.test_ratio
horizon = config.horizon

print("Reading dataset........")
df = pd.read_csv(config.inspection_data)
print(df.head())

world_ids = [i for i in range(no_worlds)]
test_ids = [id for id in range(0, no_worlds, 5)]
train_evals = [id for id in world_ids if id not in test_ids]
train_ids = random.sample(train_evals, int(no_worlds * train_ratio))
val_ids = [id for id in train_evals if id not in train_ids]

train_df = df[df['world_idx'].isin(train_ids)]
val_df = df[df['world_idx'].isin(val_ids)]

print(len(train_ids))
print(len(val_ids))
print(len(test_ids))

train_dataset = KULBarnDiffusionDataset(train_df, horizon)
val_dataset = KULBarnDiffusionDataset(val_df, horizon)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
normalizer = train_dataset.get_normalizer()
print(len(train_dataloader))

for batch in train_dataloader:
    print(batch['obs'].shape)
    print(batch['action'].shape)
    break

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
obs_dim = batch['obs'].shape[-1]
action_dim = batch['action'].shape[-1]
input_dim = obs_dim + action_dim

model = DiffusionModel(config=config, obs_dim=obs_dim)
policy = model.policy


# model = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim)
# noise_scheduler = DDPMScheduler(num_train_timesteps=20, beta_schedule='linear')
# policy = DiffusionUnetLowdimPolicy(
#     model=model, 
#     noise_scheduler=noise_scheduler, 
#     horizon=horizon, 
#     obs_dim=obs_dim, 
#     action_dim=action_dim, 
#     n_obs_steps=1,
#     n_action_steps=4,
#     obs_as_global_cond=True,
# )

policy.set_normalizer(normalizer)
policy.to(device)

num_epochs = config.num_epochs
losses = []
mse_losses = []
save_loss_every = config.save_loss_every
total_loss = 0
count = 0

optimizer = optim.Adam(list(policy.model.parameters()), lr=config.learning_rate)
policy.model.train()
for epoch in range(num_epochs):
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = policy.compute_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        count += 1
        if count >= save_loss_every:
            curr_loss = total_loss / save_loss_every
            # print("Loss:", curr_loss)
            losses.append(curr_loss)
            total_loss = 0
            count = 0

    with torch.no_grad():
        total_mse_losses = 0
        for batch in tqdm(val_dataloader):
            obs_dict = {'obs': batch['obs'].to(device)}
            pred = policy.predict_action(obs_dict)['action_pred']
            mse_loss = F.mse_loss(pred, batch['action'].to(device))
            total_mse_losses += mse_loss.item()  
        mse_loss = total_mse_losses / len(val_dataloader)
        mse_losses.append(mse_loss)
        print("Val MSE Loss:", mse_loss)
    print("Epoch: ", epoch, "/",num_epochs)



from datetime import datetime, timedelta
import pytz

singapore_tz = pytz.timezone('Asia/Singapore')
current_time = datetime.now(singapore_tz)
adjusted_time = current_time - timedelta(minutes=9)
timestamp = adjusted_time.strftime('%y%m%d_%H%M%S')

base_path = '/jackal_ws/src/mlda-barn-2024/outputs/diffusion_policies/'
dir_path = os.path.join(base_path, timestamp)
os.makedirs(dir_path, exist_ok=True)




suffix = "diffuser_policy_10Hz_diffusion_steps_20"
# save losses
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig(dir_path + f'/diffuser_losses.png')
plt.clf()  # Clear the current figure

plt.plot(mse_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Val MSE Loss')
plt.savefig(dir_path + f'/val_mse_losses.png')

# save the policy
torch.save({
    'model': policy.model.state_dict(),
    'normalizer': policy.normalizer.state_dict()
}, dir_path + '/diffusion_policies_model.pth')

print(f"Policy saved to {dir_path + '/diffusion_policies_model.pth'}")

config_dst = os.path.join(dir_path, 'config.yaml')
shutil.copyfile(config_path, config_dst)