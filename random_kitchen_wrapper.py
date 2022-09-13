import collections
import gym
import time, datetime
import numpy as np
import re 
import mj_envs
import PIL.Image
import os
from mujoco_py import cymj
from glob import glob 
import random
import mujoco_py
from copy import deepcopy
from matplotlib import pyplot as plt

class RandomKitchenEnv(gym.Env):
    def __init__(
        self, 
        task_name, 
        layout_seed_range=[0,11], 
        seed=0,
        camera_names=['new_left','new_right'],
        frame_size=(224,224),
        render_device_id=0,
        env_augment_kwargs=dict(augment_types=['light', 'texture', 'joint']) # NOTE(Mandi): this by default samples a different lighting & joint for non-task objects
        ):

        self.layout_seed_range = list(layout_seed_range)
        self.env_augment_kwargs = env_augment_kwargs
        self.np_random, seed = gym.utils.seeding.np_random(seed) 
        self.camera_names = camera_names
        self.frame_size = frame_size
        self.render_device_id = render_device_id

        self._env = None 
        self.action_space = None # to be set after sample_one_env()
        self.sample_one_env()
        

    def sample_one_env(self):
        if self._env is not None:
            self._env.close()
        del self._env
        layout_seed = self.np_random.randint(self.layout_seed_range[0], self.layout_seed_range[1])
        self._env = gym.make(
            task_name, sample_appliance=True, sample_layout=True, seed=layout_seed)
        self._env.set_augment_kwargs(
            self.env_augment_kwargs
            )
        self.action_space = self._env.action_space
        self.task_name = task_name
        obj_goal = self._env.obj_goal.copy()
        task_layout_vector = [obj_goal]
        for obj in ['hingecabinet', 'microwave', 'kettle',  'slidecabinet']:
            bid = self._env.sim.model.body_name2id(obj)
            bpos = self._env.sim.model.body_pos[bid] 
            task_layout_vector.append(bpos)
            bquat = self._env.sim.model.body_quat[bid]
            task_layout_vector.append(bquat)
 
        task_layout_vector = np.concatenate(task_layout_vector, axis=0)
        self.task_layout_vector = task_layout_vector # should be shape (43,)

    def reset(self):
        # deletes previous env sample a new one with same task but different layout seed!
        self.sample_one_env()
        state_obs = self._env.reset()
        obs = self.render_env(self._env)
        return {"state_obs": state_obs, "image_obs": obs, "task_layout_vector": self.task_layout_vector}
    
    def render_env(self, env):
        obs = env.render_camera_offscreen(
            sim=env.sim,
            cameras=self.camera_names,
            width=self.frame_size[0],
            height=self.frame_size[1],
            device_id=self.render_device_id
        ) 
        # NOTE(Mandi): this should be shape (2, 224, 224, 3) images, since we use two cameras
        return obs 

    def step(self, act):
        next_state_obs, rwd, done, next_env_info = self._env.step(act)
        next_obs_dict = {'state_obs': next_state_obs}
        next_obs_dict['task_layout_vector'] = self.task_layout_vector
        next_obs_dict['image_obs'] = self.render_env(self._env)

        return next_obs_dict, rwd, done, next_env_info

if __name__ == '__main__':
    task_name = 'kitchen_knob1_on-v3'
    frame_size = (224,224)
    env = RandomKitchenEnv(task_name, frame_size=frame_size)
    breakpoint()
    env.reset()
