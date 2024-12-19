

# torch Dataset
from torch.utils.data import Dataset
import numpy as np

class KULBarnDataset(Dataset):
    def get_normalized_goal(self):
        x = self.data['pos_x']
        y = self.data['pos_y']
        goal_x = self.data['goal_x']
        goal_y = self.data['goal_y']
        theta = self.data['pose_heading']
        self.data['goal_x'] = np.cos(theta) * (goal_x - x) + np.sin(theta) * (goal_y - y)
        self.data['goal_y'] = -np.sin(theta) * (goal_x - x) + np.cos(theta) * (goal_y - y)

    def __init__(self, df, mode="train"):
        super().__init__()

        self.data = df
        self.get_normalized_goal()  
        
        # get all the column values that contain the word lidar
        self.lidar_cols = ["lidar_" + str(i) for i in range(0, 360, 1)]
        # get actions columns
        self.actions_cols = ['cmd_vel_linear', 'cmd_vel_angular']
        # get other columns
        self.non_lidar_cols = ['local_goal_x', 'local_goal_y', 'goal_x', 'goal_y']

        # if mode == "train":
        #     # Manually compute the min and max values for each column
        #     self.min = self.data.min()
        #     self.max = self.data.max()
        #     # Save the mean and std to a JSON file
        #     scaler_params = {
        #         'min': self.min.to_dict(),
        #         'max': self.max.to_dict()
        #     }
        #     with open('scaler_params.json', 'w') as f:
        #         json.dump(scaler_params, f)
        # else:
        #     # Load the mean and std from the JSON file
        #     with open('scaler_params.json', 'r') as f:
        #         scaler_params = json.load(f)
        #     self.min = pd.Series(scaler_params['min'])
        #     self.max = pd.Series(scaler_params['max'])
        
        # dont normalizer local_x and local_y
        # self.normalized_data = (self.data - self.min) / (self.max - self.min)
        self.normalized_data = self.data
         
        self.lidar_data = self.normalized_data[self.lidar_cols].values
        self.non_lidar_data = self.normalized_data[self.non_lidar_cols].values
        self.actions_data = self.normalized_data[self.actions_cols].values

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        lidar = self.lidar_data[idx]
        non_lidar = self.non_lidar_data[idx]
        actions = self.actions_data[idx]
        return lidar, non_lidar, actions
