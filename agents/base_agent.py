import torch

class BaseAgent(object):
    """
    Class for the basic agent objects.
    To define your own agent, subclass this class and implement the functions below.
    """

    def __init__(self, 
                 env, 
                 policy,
                 logger,
                 storage,
                 device):
        """
        env: (gym.Env) environment following the openAI Gym API
        """
        self.env = env
        self.policy = policy
        self.logger = logger
        self.storage = storage
        self.device = device
        
        self.t = 0
        # By default, pytorch utilizes multi-threaded cpu
        # However, under the small model or using GPU, synchronization over threads often harm the performance
        torch.set_num_threads(1)

        
    def predict(self, obs):
        """
        Predict the action with the given input 
        """
        pass
        
    def update_policy(self):
        """
        Train the neural network model
        """
        pass
        
    def train(self, num_timesteps):
        """
        Train the agent with collecting the trajectories
        """
        pass    
