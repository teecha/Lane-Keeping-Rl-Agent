import torch
import torch.nn as nn
from torch import autograd
from gym.spaces import Box
from torch.distributions import Normal, MultivariateNormal

class gaussianPolicy(nn.Module):
    """
    This class defines a pyTorch based Diagonal Gaussian 
    policy(actor) and a critic.

    The Gaussian policy is parameterized by the mean and
    standard deviation. The critic is a simple feed-forward
    fully connected neural-network.

    Attributes
    ----------
    * obs_space: Gym compatible observation space.
    * action_space: Gym compatible action space.
    * std: Initial policy standard deviation.
    """

    def __init__(self, obs_space, action_space, std=0.0):
        super(gaussianPolicy, self).__init__()

        # Save environment observation and action space.
        self.obs_space = obs_space
        self.action_space = action_space
        self.std = std

        # Deduce observation space 
        self.occ_obs_space = self.obs_space[0]
        self.ego_speed_obs_space = self.obs_space[1]

        if not isinstance(self.occ_obs_space, Box):
            raise TypeError("occupancy observation space must be an instance of Box.")
        if not isinstance(self.ego_speed_obs_space, Box):
            raise TypeError("ego speed observation space must be an instance of Box.")
        if not isinstance(action_space, Box):
            raise TypeError("action space must be an instance of Box.")
        

        self.num_input_channels = self.obs_space[0].shape[0]
        padding_shape = (0,
                         self.get_padding_size(self.occ_obs_space.shape[1], 8, 4),
                         0,
                         self.get_padding_size(self.occ_obs_space.shape[2], 8, 4))

        # CNN based feature extractor for occupancy grids.
        self.occ_features  = nn.Sequential(
            nn.ZeroPad2d(padding_shape),
            nn.Conv2d(in_channels=self.num_input_channels, 
                      out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, 
                      out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, 
                      out_channels=64, kernel_size=3, stride=1),
            nn.ReLU())

        # Fully-connected feature extractor for ego-vehicle speeds.
        self.speed_features = nn.Sequential(
            nn.Linear(in_features=self.ego_speed_obs_space.shape[0], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),)

        self.merged_size = self.feature_size(self.occ_features, self.occ_obs_space.shape) + self.feature_size(self.speed_features, self.ego_speed_obs_space.shape)

        """
         Actor-network
         Mean """
        self.actor = nn.Sequential(
            nn.Linear(in_features=self.merged_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.action_space.shape[0]))
        """ Std-deviation """
        self.log_std = nn.Parameter(torch.ones(1, *self.action_space.shape) * std)
        

        # Critic-network
        self.critic = nn.Sequential(
            nn.Linear(in_features=self.merged_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1))
        
    
    def feature_size(self, seq_model, input_shape):
        '''Returns the flattened output_shape of a pyTorch sequential model.

        Attributes
        ----------

        * seq_model: The pytorch sequential model.
        '''
        return seq_model(autograd.Variable(torch.zeros(1, *input_shape))).view(1, -1).size(1)

        
    def get_padding_size(self, input_size, kernel_size, stride):
        '''Function to return the valid padding size. We only pad
            right and bottom of the occupancy grid.
        '''
        if(self.check_for_valid_size(input_size, 
                                     kernel_size,
                                     stride,
                                     0)):
            return 0
        else:
            for pad in range(0, stride):
                if(self.check_for_valid_size(input_size,
                                             kernel_size,
                                             stride,
                                             pad)):
                    return pad
            raise ValueError("Could not able to find valid padding size.")
    
    def check_for_valid_size(self, input_size, kernel_size, stride, padding):
        '''Check whether the given input image is valid'''
        out_shape = input_size - kernel_size + padding
        if out_shape % stride == 0:
            return True
        else:
            return False
        
    def get_weights(self, ):
        '''This function returnt the weights of the policy.
        '''
        return self.state_dict()
    
    def set_weights(self, state_dict):
        '''Updates the weight of the policy specified by
            state dict.

        Attributes
        ----------

        * state_dict: The state dict containing weights of the policy.
        '''
        self.load_state_dict(state_dict)
        print("Loaded Policy weights form state-dict")
    
    def forward(self, obs):
        occ_obs, speed_obs = obs
        assert occ_obs.size(0) == speed_obs.size(0)
        bsize = occ_obs.size(0)

        # Occ features
        self.grid_features = self.occ_features(occ_obs).view(bsize, -1)
        
        # Ego-vehicle speed features
        self.ego_vehicle_speed_features = self.speed_features(speed_obs)
        
        # Concatenated features vector
        self.merged_features = torch.cat((self.grid_features, self.ego_vehicle_speed_features,),
                                         dim=1)
        
        assert self.merged_features.size(1) == self.merged_size

        # Policy
        mean = self.actor(self.merged_features)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)

        # value-function
        value = self.critic(self.merged_features)

        return dist, value