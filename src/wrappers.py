import gymnasium as gym
import numpy as np

class TimePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.01):
        super().__init__(env)
        self.penalty = penalty
    def step(self, action):
        obs,reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.penalty, terminated, truncated, info   
class HoverRewardWrapper(gym.Wrapper):
    def __init__(self,env, bonus=0.1):
        super().__init__(env)
        self.bonus = bonus 

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        #obs:[x,y,vy, angle, angular_velocity, leg1, leg2]
        upright = abs(obs[4]) < 0.1  # Check if the lander is upright (angle close to 0)
        slow = abs(obs[2]) < 0.1 and abs(obs[3]) < 0.1  # Check if the lander is moving slowly (vertical velocity close to 0)

        if upright and slow:
            reward += self.bonus  # Add bonus reward for hovering
        return obs, reward, terminated, truncated, info

class PotentialShapingWrapper(gym.Wrapper):
    def __init__(self, env, gamma=0.999, scale=1/0):
        super().__init__(env)
        self.gamma = gamma
        self.scale = scale
        self.prev_potential = None
    def _potential(self, obs):
        # Negative distance to the origin aka landing pad
        x,y = obs[0], obs[1]
        return -np.sqrt(x**2 +y*y)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_potential = self._potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_potential = self._potential(obs)
        shaped = reward +self.scale * (self.gamma * new_potential -self.prev_potential)
        self.prev_potential = new_potential
        return obs, shaped, terminated, truncated, info
    
