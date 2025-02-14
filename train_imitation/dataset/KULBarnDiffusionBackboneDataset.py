
import numpy as np
import pandas as pd
from typing import Dict
import torch

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer


class KULBarnDiffusionDataset(BaseLowdimDataset):
    def get_normalized_goal(self):
        x = self.data['pos_x']
        y = self.data['pos_y']
        goal_x = self.data['goal_x']
        goal_y = self.data['goal_y']
        theta = self.data['pose_heading']
        self.data['goal_x'] = np.cos(theta) * (goal_x - x) + np.sin(theta) * (goal_y - y)
        self.data['goal_y'] = -np.sin(theta) * (goal_x - x) + np.cos(theta) * (goal_y - y)

    def __init__(self, df, horizon):
        super().__init__()
        
        self.data = df
        # only take successful episodes
        self.data = self.data[self.data['success'] == True].reset_index(drop=True)
        self.get_normalized_goal()  

        self.data = pd.DataFrame(self.data, columns=self.data.columns)
        self.horizon = horizon

        # Process data columns
        self.lidar_cols = [col for col in self.data.columns if 'lidar' in col]
        self.actions_cols = ['cmd_vel_linear', 'cmd_vel_angular']
        self.non_lidar_cols = ['local_goal_x', 'local_goal_y', 'goal_x', 'goal_y']
        self.obs_cols = self.lidar_cols + self.non_lidar_cols

        self.lidar_data = self.data[self.lidar_cols].values
        self.non_lidar_data = self.data[self.non_lidar_cols].values
        self.actions_data = self.data[self.actions_cols].values
        self.obs_data = self.data[self.obs_cols].values

        print("Lidar Columns:", self.lidar_cols)
        print("Non Lidar Columns:", self.non_lidar_cols)
        print("Action Columns:", self.actions_cols)     

        self.data['episode_id'] = (self.data['timestep'] == 0).cumsum()
        self.grouped_data = self.data.groupby(['episode_id'])
        self.horizon = horizon
        self.indices = self.make_indices(horizon)
        # print("indices: ", self.indices)

    def make_indices(self, horizon): #n_obs
        indices = []
        for name, group in self.grouped_data:
            original_indices = group.index.values
            path_length = len(group)
            max_start = path_length - horizon
            for start in range(max_start + 1):  # Include the last possible starting point
                end = start + horizon
                indices.append(original_indices[start:end])
        print("indices",indices[0])
        return indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        curr_indices = self.indices[idx]
        data = {
            'lidar_data': self.lidar_data[curr_indices],
            'non_lidar_data': self.non_lidar_data[curr_indices],
            'action': self.actions_data[curr_indices],
        }
        torch_data = data
        return torch_data

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        data_dict = {
            'lidar_data': self.lidar_data,
            'non_lidar_data': self.non_lidar_data,
            'action': self.actions_data
        }
        normalizer.fit(data=data_dict, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.actions_data)