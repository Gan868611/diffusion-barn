#!/root/miniconda3/envs/robodiff/bin/python
import torch
import time 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import sys

# Add your path
sys.path.append('/jackal_ws/src/mlda-barn-2024/train_imitation/diffusion_policy')
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.common.normalizer import LinearNormalizer

# Define your model and normalizer
lidar_dim = 360 # Replace with the actual lidar dimension
non_lidar_dim = 4  # Replace with the actual non-lidar dimension
NFRAMES = 4  # Replace with the actual number of frames





class DiffusionModel():
    def __init__(self, config,obs_dim, filepath=None,):
        action_dim = 2
        input_dim = obs_dim + action_dim
        print("Python version:", sys.version)
        print("PyTorch version:", torch.__version__)
        model = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim)
        noise_scheduler = DDPMScheduler(num_train_timesteps=config.diffusion_steps, beta_schedule='linear')
        policy = DiffusionUnetLowdimPolicy(
            model=model, 
            noise_scheduler=noise_scheduler, 
            horizon=config.horizon, 
            obs_dim=obs_dim, 
            action_dim=action_dim, 
            n_obs_steps=1,
            n_action_steps=config.horizon,
            obs_as_global_cond=True,
            oa_step_convention=True,
        )
        
        normalizer = LinearNormalizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the state dictionaries
        if filepath:
            print("Diffusion model:", self.device)
            checkpoint = torch.load(filepath, map_location=self.device)

            policy.cnn_model.load_state_dict(checkpoint['cnn_model'])
            policy.model.load_state_dict(checkpoint['model'])
      
            policy.cnn_model.eval()
            policy.model.eval()
            for param in policy.model.parameters():
                param.requires_grad = False
            for param in policy.cnn_model.parameters():
                param.requires_grad = False

            normalizer.load_state_dict(checkpoint['normalizer'])

        # Set the normalizer in the policy
        policy.set_normalizer(normalizer)

        # Move the policy to the appropriate device
        
        policy.to(self.device)

        self.policy = policy
        print("Model and normalizer loaded successfully.")

    def __call__(self, lidar, non_lidar):
        obs_dict = {'lidar_data': lidar.to(self.device), 'non_lidar_data': non_lidar.to(self.device)}
        return self.policy.predict_action(obs_dict)['action_pred']

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    exit()

    model = DiffusionModel()
    print(model.policy)

    start_time = time.time()
    lidar = torch.rand(1, 4, 360)
    non_lidar = torch.rand(1, 4, 4)
    action = model(lidar, non_lidar)
    print(action)
    
    print(f"Time taken: {time.time() - start_time} seconds")