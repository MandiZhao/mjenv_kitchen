""" =================================================
Copyright (C) 2018 Vikash Kumar
Author  :: Vikash Kumar (vikashplus@gmail.com)
Source  :: https://github.com/vikashplus/mj_envs
License :: Under Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
================================================= """

import collections
import gym
import time, datetime
import numpy as np
import re 
from mj_envs.envs.multi_task.multi_task_base_v1 import KitchenBase
from mj_envs.envs.env_base import get_sim
from mj_envs.utils.obj_vec_dict import ObsVecDict
from mj_envs.utils.quat_math import euler2quat
from mujoco_py.modder import TextureModder, LightModder
import PIL.Image
import os
from mujoco_py import cymj
from glob import glob 
import random
import mujoco_py
from copy import deepcopy
from mj_envs.envs.multi_task.common.constants import \
    DEMO_RESET_QPOS, DEMO_RESET_QVEL, \
    OBJ_INTERACTION_SITES, OBJ_JNT_NAMES, ROBOT_JNT_NAMES, \
    TEXTURE_ID_TO_INFOS, OBJ_JNT_RANGE, DEFAULT_BODY_RANGE

class KitchenFrankaFixed(KitchenBase):
    OBJ_INTERACTION_SITES = OBJ_INTERACTION_SITES
    OBJ_JNT_NAMES = OBJ_JNT_NAMES
    ROBOT_JNT_NAMES = ROBOT_JNT_NAMES

    def _setup(
        self,
        robot_jnt_names=ROBOT_JNT_NAMES,
        obj_jnt_names=OBJ_JNT_NAMES,
        obj_interaction_site=OBJ_INTERACTION_SITES,
        **kwargs,
    ):
        super()._setup(
            robot_jnt_names=robot_jnt_names,
            obj_jnt_names=obj_jnt_names,
            obj_interaction_site=obj_interaction_site,
            **kwargs,
        )


class KitchenFrankaDemo(KitchenFrankaFixed):

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        super()._setup(**kwargs)

    def reset(self, reset_qpos=None, reset_qvel=None):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qvel = self.init_qvel.copy()
            reset_qpos[self.robot_dofs] = DEMO_RESET_QPOS[self.robot_dofs]
            reset_qvel[self.robot_dofs] = DEMO_RESET_QVEL[self.robot_dofs]
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)


class KitchenFrankaRandom(KitchenFrankaFixed):

    def __init__(self, model_path, obsd_model_path=None, seed=None, *kwargs):
        # EzPickle.__init__(**locals()) is capturing the input dictionary of the init method of this class.
        # In order to successfully capture all arguments we need to call gym.utils.EzPickle.__init__(**locals())
        # at the leaf level, when we do inheritance like we do here.
        # kwargs is needed at the top level to account for injection of __class__ keyword.
        # Also see: https://github.com/openai/gym/pull/1497
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)

        # This two step construction is required for pickling to work correctly. All arguments to all __init__
        # calls must be pickle friendly. Things like sim / sim_obsd are NOT pickle friendly. Therefore we
        # first construct the inheritance chain, which is just __init__ calls all the way down, with env_base
        # creating the sim / sim_obsd instances. Next we run through "setup"  which relies on sim / sim_obsd
        # created in __init__ to complete the setup.
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed)

        super()._setup(**kwargs)

    def reset(self, reset_qpos=None, reset_qvel=None):
        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy()
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )
        return super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)

TEX_DIR = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "kitchen") 
DEFAULT_TEXTURE_KWARGS = {
    'tex_ids': [1, 5, 6, 7, 10, 11, 12, 13, 14, 16],
    'tex_names': {
        'surface': [
            'wood',
            'stone',
            ],
        'handle': [
            'metal',
            ],
        'floor': [
            'tile'
            ],
        },
    'tex_path':  TEX_DIR + "/textures/*/*.png",
    }

OBJ_LIST = list(OBJ_JNT_RANGE.keys())


def randomize_appliances(model_path, np_random, seed, write_local=True):

    model_file = open(model_path, "r")
    model_xml = model_file.read()
    opt = np_random.randint(low=0, high=4)
    model_xml = re.sub('microwave_body\d.xml','microwave_body{}.xml'.format(opt), model_xml)
    opt = np_random.randint(low=0, high=8) 
    model_xml = re.sub('kettle_body\d.xml','kettle_body{}.xml'.format(opt), model_xml)

    processed_model_path = None
    if write_local:
        # Save new random xml
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        processed_model_path = model_path[:-4]+f"_random_seed{seed}.xml"
        # os check if file exists:
        if not os.path.isfile(processed_model_path):
            print('writing new model:', processed_model_path)
            with open(processed_model_path, 'w') as file:
                file.write(model_xml)

    return processed_model_path
    
class KitchenFrankaAugment(KitchenFrankaFixed):

    def __init__(
            self, model_path, obsd_model_path=None, seed=None, 
            sample_appliance=False, sample_layout=False, augment_types=[],
            focused_goal=False,
            **kwargs):
        """ Overwrites the init function in env_base.MujocoEnv """
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        self.seed(seed)
        self.focused_goal = focused_goal # if True, only consider goal error from input_obj_goal
        if sample_appliance:
            model_path = randomize_appliances(model_path, self.np_random, seed=seed, write_local=True) 
            assert obsd_model_path is None
        self.sim = get_sim(model_path=model_path)
        self.sim_obsd = get_sim(obsd_model_path) if obsd_model_path else get_sim(model_path=model_path)
        self.sim.forward()
        self.sim_obsd.forward() 
        ObsVecDict.__init__(self)
        self._model_path = model_path
        
        super()._setup(**kwargs)


        # set default kwargs for randomization
        self.augment_types = augment_types
        self.body_rand_kwargs = DEFAULT_BODY_RANGE
        self.texture_modder = TextureModder(self.sim)
        self.texture_rand_kwargs = DEFAULT_TEXTURE_KWARGS 
        self.original_obj_goals = deepcopy(self.obj_goal) # save original obj_goal!
        self.joints_rand_kwargs = {
            'num_objects': 10,
            'non_target_objects': [obj for obj in OBJ_LIST if obj not in self.input_obj_goal.keys()]
        }
        
        self.light_rand_kwargs = {
            'ambient': {
                'low': -0.1,
                'high': 0.2,
                'center': deepcopy(self.sim.model.light_ambient), 
            },
            'diffuse': {
                'low': -0.3,
                'high': 0.3,
                'center': deepcopy(self.sim.model.light_diffuse),
            },
            'specular': {
                'low': -10,
                'high': 10,
                'center': deepcopy(self.sim.model.light_specular),
            }
        }

        self.goal = None 
        if sample_layout: 
            self.randomize_layout()
            # TODO: save sim.model to xml file  
        self.sample_appliance = sample_appliance
        self.sample_layout = sample_layout
        
    def set_augment_kwargs(self, aug_kwargs):
        self.augment_types = aug_kwargs.get('augment_types', [])

        # override default kwargs
        self.body_rand_kwargs.update(
            aug_kwargs.get('body', {})) 
        
        self.texture_rand_kwargs.update(
            aug_kwargs.get('texture', {}))
        if 'texture' in self.augment_types:
            texture_files = glob(self.texture_rand_kwargs['tex_path'])
            assert len(texture_files) > 0, "No texture files found at path: {}".format(self.texture_rand_kwargs['tex_path'])
        
        self.joints_rand_kwargs.update(
            aug_kwargs.get('joint', {}))

        self.light_rand_kwargs.update(
            aug_kwargs.get('light', {}))

    def get_augment_kwargs(self):
        return {
            'augment_types': self.augment_types,
            'body': self.body_rand_kwargs,
            'texture': self.texture_rand_kwargs,
            'joint': self.joints_rand_kwargs,
            'light': self.light_rand_kwargs,
        }

    def get_model_path(self):
        return self._model_path
        
    def randomize_body_pose(self):
        """ NOTE(Mandi): cannot use if randomize_layout() is used, would just set the body to original layout """
        def body_rand(name):
            kwargs = self.body_rand_kwargs.get(name, None)
            assert kwargs is not None, "body {} not found in body_rand_kwargs".format(name)
            pos = np.array(kwargs['pos']['center']) + \
                self.np_random.uniform(low=kwargs['pos']['low'], high=kwargs['pos']['high'])

            euler = np.array(kwargs['euler']['center']) + \
                self.np_random.uniform(low=kwargs['euler']['low'], high=kwargs['euler']['high'])
            
            bid = self.sim.model.body_name2id(name)
            self.sim.model.body_pos[bid] = pos
            self.sim.model.body_quat[bid] = euler2quat(euler)
            return pos, euler

        # dk_pos, _ = body_rand('desk')
        if 'micro0joint' not in self.input_obj_goal.keys():
            body_rand('microwave')
        #hc_pos, _  = body_rand('hingecabinet')
        print('TODO: add back counters')
        # body_rand('counters')

        if 'leftdoorhinge' not in self.input_obj_goal.keys() and \
            'rightdoorhinge' not in self.input_obj_goal.keys():
            body_rand('hingecabinet')
        # self.body_rand_kwargs['slidecabinet']['pos']['center'] = hc_pos
        if 'slidedoor_joint' not in self.input_obj_goal.keys():
            body_rand('slidecabinet') 
        # self.body_rand_kwargs['kettle0']['pos']['center'] = dk_pos
        body_rand('kettle')

    def randomize_texture(self):
        def set_bitmap(tex_id, modder, new_bitmap):
            texture = modder.textures[tex_id]
            curr_bitmap = texture.bitmap
            assert curr_bitmap.dtype == new_bitmap.dtype and curr_bitmap.shape == new_bitmap.shape, \
                 f'Texture ID: {tex_id}: Incoming bitmap shape {new_bitmap.shape} and dtype {new_bitmap.dtype} does not match current bitmap: {curr_bitmap.shape}, {curr_bitmap.dtype}'
            modder.textures[tex_id].bitmap[:] = new_bitmap
            
            if not modder.sim.render_contexts:
                cymj.MjRenderContextOffscreen(modder.sim)
            for render_context in modder.sim.render_contexts:
                render_context.upload_texture(texture.id)
            return 

        tex_ids = self.texture_rand_kwargs.get('tex_ids', [])
        tex_files = glob(self.texture_rand_kwargs['tex_path'])
        assert len(tex_files) > 0, "No texture files found"
        for tex_id in tex_ids:
            tex_info = TEXTURE_ID_TO_INFOS.get(tex_id, None)
            assert tex_info is not None, f'ID {tex_id} not found'
            texture_keys = self.texture_rand_kwargs['tex_names'].get(tex_info['group'], None)
            assert texture_keys is not None, f"Texture group {tex_info['group']} not found"
            found_files = [f for f in tex_files if any([t in f for t in texture_keys])]

            fidx = self.np_random.randint(len(found_files))
            new_tex = PIL.Image.open(found_files[fidx]).convert('RGB')

            if np.asarray(new_tex).shape != tex_info['shape']:
                new_tex = new_tex.resize(
                    (tex_info['shape'][0], tex_info['shape'][1])
                    )

            new_tex = np.asarray(new_tex, dtype=np.uint8)
            set_bitmap(tex_id, self.texture_modder, new_tex)
        return 

    def randomize_object_joints(self):
        object_keys = self.joints_rand_kwargs.get('non_target_objects', [])
        num_objects = self.joints_rand_kwargs.get('num_objects', 0)
        if 'kettle0:Tx' in self.input_obj_goal.keys() or 'kettle0:Ty' in self.input_obj_goal.keys():
            num_objects = min(9, num_objects)
        assert len(object_keys) > 0, "No non-target objects found"
        side_objs = list(self.np_random.choice(object_keys, num_objects, replace=False))
        new_vals = []
        for side_obj_name in side_objs:
            val_range = OBJ_JNT_RANGE[side_obj_name]
            dof_adr = self.obj[side_obj_name]["dof_adr"] 
            new_val = self.np_random.uniform(low=val_range[0], high=val_range[1])
            new_vals.append( (side_obj_name, dof_adr, new_val) )

        env_state = self.get_env_state()
        new_obj_goal = deepcopy(self.original_obj_goals)
        for (side_obj_name, dof_adr, new_val) in new_vals: 
            env_state['qpos'][dof_adr] = new_val
            # NOTE: need to also reset the goal joint for each randomized side object
            goal_adr = self.obj[side_obj_name]["goal_adr"]
            # print(f'{side_obj_name}: old: {new_obj_goal[goal_adr]}, new goal {new_val}')
            new_obj_goal[goal_adr] = new_val
            
        
        self.set_obj_goal(obj_goal=new_obj_goal)

        self.set_state(
                qpos=env_state['qpos'], 
                qvel=env_state['qvel']
                )

        self.set_sim_obsd_state(
                qpos=env_state['qpos'],
                qvel=env_state['qvel']
                )
        return 

    def randomize_lights(self):
        for i in range(4):
            if 'ambient' in self.light_rand_kwargs:
                low = self.light_rand_kwargs['ambient']['low']
                high = self.light_rand_kwargs['ambient']['high']
                center = self.light_rand_kwargs['ambient']['center']
                new_vals = self.np_random.uniform(low, high, size=1)
                self.sim.model.light_ambient[i, :] = center[i] + new_vals
        
            if 'diffuse' in self.light_rand_kwargs:
                low = self.light_rand_kwargs['diffuse']['low']
                high = self.light_rand_kwargs['diffuse']['high']
                center = self.light_rand_kwargs['diffuse']['center']
                new_vals = self.np_random.uniform(low, high, size=1)
                self.sim.model.light_diffuse[i, :] = center[i] + new_vals
            
            if 'specular' in self.light_rand_kwargs:
                low = self.light_rand_kwargs['specular']['low']
                high = self.light_rand_kwargs['specular']['high']
                center = self.light_rand_kwargs['specular']['center']
                new_vals = self.np_random.uniform(low, high, size=1)
                self.sim.model.light_specular[i, :] = center[i] + new_vals

    def randomize_layout(self):
        np_random = self.np_random
        sim = self.sim
        body_offset = {
            'slidecabinet': {'pos':[0,0,0], 'euler':[0,0,0]},
            'hingecabinet': {'pos':[0,0,0], 'euler':[0,0,0]},
            'microwave': {'pos':[0,0,-0.2], 'euler':[0,0,0]},
            'kettle': {'pos':[0,0,0.0], 'euler':[0,0,0]},
        }

        # Counter layout
        layout = {
                'sink': {
                    'L': [-1.620, 0, 0],
                    'R': [0, 0, 0]},
                'island': {
                    'L': [-.020, 0, 0],
                    'R': [1.92, 0, 0]},
        }
        counter_loc = ['L', 'R']

        # Appliance layout
        app_xy = {
            # Front pannel
            'FL':[-.35, 0.28],
            'FR':[.5, 0.28],
            # Left pannel
            'LL':[-0.85, -0.85],
            'LR':[-0.75, -.15],
            # Right pannel
            'RL':[1., -.25],
            'RR':[1., -1.0],
        }
        app_z = {
                'T':[2.6],
                'M':[2.2],
                'B':[1.8],
            }
        app_xyz = {
            # Front pannel
            'FLT': {'pos':app_xy['FL']+app_z['T'], 'euler':[0,0,0], 'accept':True},
            'FRT': {'pos':app_xy['FR']+app_z['T'], 'euler':[0,0,0], 'accept':True},
            'FLM': {'pos':app_xy['FL']+app_z['M'], 'euler':[0,0,0], 'accept':False},
            'FRM': {'pos':app_xy['FR']+app_z['M'], 'euler':[0,0,0], 'accept':False},
            'FLB': {'pos':app_xy['FL']+app_z['B'], 'euler':[0,0,0], 'accept':False},
            'FRB': {'pos':app_xy['FR']+app_z['B'], 'euler':[0,0,0], 'accept':False},
            # Left pannel
            'LLT': {'pos':app_xy['LL']+app_z['T'], 'euler':[0,0,1.57], 'accept':True},
            'LRT': {'pos':app_xy['LR']+app_z['T'], 'euler':[0,0,1.57], 'accept':True},
            'LLM': {'pos':app_xy['LL']+app_z['M'], 'euler':[0,0,1.57], 'accept':True},
            'LRM': {'pos':app_xy['LR']+app_z['M'], 'euler':[0,0,1.57], 'accept':False},
            'LLB': {'pos':app_xy['LL']+app_z['B'], 'euler':[0,0,1.57], 'accept':True},
            'LRB': {'pos':app_xy['LR']+app_z['B'], 'euler':[0,0,1.57], 'accept':True},
            # Right pannel
            'RLT': {'pos':app_xy['RL']+app_z['T'], 'euler':[0,0,-1.57], 'accept':True},
            'RRT': {'pos':app_xy['RR']+app_z['T'], 'euler':[0,0,-1.57], 'accept':True},
            'RLM': {'pos':app_xy['RL']+app_z['M'], 'euler':[0,0,-1.57], 'accept':False},
            'RRM': {'pos':app_xy['RR']+app_z['M'], 'euler':[0,0,-1.57], 'accept':True},
            'RLB': {'pos':app_xy['RL']+app_z['B'], 'euler':[0,0,-1.57], 'accept':True},
            'RRB': {'pos':app_xy['RR']+app_z['B'], 'euler':[0,0,-1.57], 'accept':True},
        }
        app_loc = [*app_xyz] # list of dict keys

        # Randomize counter layouts
        opt = np_random.randint(low=0, high=2)
        sel_grid = counter_loc[opt]
        # Place island
        bid = sim.model.body_name2id('island')
        sim.model.body_pos[bid] = layout['island'][sel_grid]
        # # place sink
        sel_grid = counter_loc[1-opt]
        bid = sim.model.body_name2id('sink')
        sim.model.body_pos[bid] = layout['sink'][sel_grid]
        # Don't mount anything next to sink 
        for side in ['L','R']:
            app_xyz[sel_grid+side+'B']['accept'] = False

        # Randomize Appliances
        for body_name in ['slidecabinet', 'hingecabinet', 'microwave']:
            # Find and empty slot
            empty_slot = False
            while not empty_slot:
                opt = np_random.randint(low=0, high=len(app_loc))
                sel_grid = app_loc[opt] 
                empty_slot = True if app_xyz[sel_grid]['accept'] else False
            bid = sim.model.body_name2id(body_name)
            sim.model.body_pos[bid] = np.array(app_xyz[sel_grid]['pos'])+np.array(body_offset[body_name]['pos'])
            sim.model.body_quat[bid] = euler2quat(np.array(app_xyz[sel_grid]['euler']) + np.array(body_offset[body_name]['euler']))
            # mark occupied
            app_xyz[sel_grid]['accept'] = False
            # handle corner assignments
            if sel_grid in ['LRT', 'FLT']:
                app_xyz['LRT']['accept'] = app_xyz['FLT']['accept'] = False
            if sel_grid in ['FRT', 'RLT']:
                app_xyz['FRT']['accept'] = app_xyz['RLT']['accept'] = False

        # move the kettle on the surface only 
        kwargs = DEFAULT_BODY_RANGE['kettle']
        kettle_pos_high = kwargs['pos']['high']
        if 'kettle0:Tx' in self.input_obj_goal.keys() or 'kettle0:Ty' in self.input_obj_goal.keys():
            kettle_pos_high = [0.2, 0.2, 0] 
            # NOTE: if the current task is kettle pushing, limit the initial variations so that the kettle doesn't start at the goal location 
        pos = np.array(kwargs['pos']['center']) + \
            self.np_random.uniform(low=kwargs['pos']['low'], high=kettle_pos_high)

        euler = np.array(kwargs['euler']['center']) + \
            self.np_random.uniform(low=kwargs['euler']['low'], high=kwargs['euler']['high'])
        bid = self.sim.model.body_name2id('kettle')
        self.sim.model.body_pos[bid] = pos
        self.sim.model.body_quat[bid] = euler2quat(euler)

    def reset(self, reset_qpos=None, reset_qvel=None):
        # random reset of robot initial pos 

        if reset_qpos is None:
            reset_qpos = self.init_qpos.copy() 
            if 'kettle0:Tx' not in self.input_obj_goal.keys() and ('kettle0:Ty' not in self.input_obj_goal.keys()):
                # NOTE: if kettle task, move the arm to closer
                reset_qpos[self.robot_dofs] = [0.101, -1.36, 0, -2.476, 0.3252,  0.8291,  1.6246, 0.04,  0.04 ]
            reset_qpos[self.robot_dofs] += (
                0.05
                * (self.np_random.uniform(size=len(self.robot_dofs)) - 0.5)
                * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            )
            # reset_qpos[self.robot_dofs] = (
            #      0.1
            #     * (self.np_random.uniform(low=-0.5, high=0.5, size=len(self.robot_dofs)))
            #     * (self.robot_ranges[:, 1] - self.robot_ranges[:, 0])
            # )
        super().reset(reset_qpos=reset_qpos, reset_qvel=reset_qvel)
        # if 'layout' in self.augment_types:
        #     self.randomize_layout()

        # if 'body' in self.augment_types:
        #     self.randomize_body_pose()

        if 'texture' in self.augment_types:
            self.randomize_texture()
        
        if 'joint' in self.augment_types:
            self.randomize_object_joints()

        if 'light' in self.augment_types:
            self.randomize_lights()
        
        return self.get_obs()

    def set_sim_obsd_state(self, qpos=None, qvel=None, act=None):
        """
        Set MuJoCo sim_obsd state
        """
        sim = self.sim_obsd
        assert qpos.shape == (sim.model.nq,) and qvel.shape == (sim.model.nv,)
        old_state = sim.get_state()
        if qpos is None:
            qpos = old_state.qpos
        if qvel is None:
            qvel = old_state.qvel
        if act is None:
            act = old_state.act
        new_state = mujoco_py.MjSimState(old_state.time, qpos=qpos, qvel=qvel, act=act, udd_state={})
        sim.set_state(new_state)
        sim.forward()

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["t"] = np.array([sim.data.time])
        obs_dict["robot_jnt"] = sim.data.qpos[self.robot_dofs].copy()
        obs_dict["objs_jnt"] = sim.data.qpos[self.obj["dof_adrs"]].copy()
        obs_dict["robot_vel"] = sim.data.qvel[self.robot_dofs].copy() * self.dt
        obs_dict["objs_vel"] = sim.data.qvel[self.obj["dof_adrs"]].copy() * self.dt
        obs_dict["obj_goal"] = self.obj_goal.copy()
        obs_dict["goal_err"] = (
            obs_dict["obj_goal"] - obs_dict["objs_jnt"]
        )  # mix of translational and rotational erros
        obs_dict["approach_err"] = (
            self.sim.data.site_xpos[self.interact_sid]
            - self.sim.data.site_xpos[self.grasp_sid]
        )
        obs_dict["pose_err"] = self.robot_meanpos - obs_dict["robot_jnt"]
        obs_dict["end_effector"] = self.sim.data.site_xpos[self.grasp_sid]
        obs_dict["qpos"] = self.sim.data.qpos.copy()
        for site in self.obj_interaction_site:
            site_id = self.sim.model.site_name2id(site)
            obs_dict[site + "_err"] = (
                self.sim.data.site_xpos[site_id]
                - self.sim.data.site_xpos[self.grasp_sid]
            )
        if self.focused_goal:
            new_goal_err = np.zeros_like(obs_dict["goal_err"])
            for obj_name in self.input_obj_goal.keys():
                adr = self.obj[obj_name]['goal_adr']
                new_goal_err[adr] = obs_dict["goal_err"][adr]
            # print('old goal err', obs_dict["goal_err"])
            # print('new goal err', new_goal_err)
            obs_dict["goal_err"] = new_goal_err
        return obs_dict

    def make_copy_env(self):
        copy_env = deepcopy(self)

        # match texture
        for tex_id in self.texture_rand_kwargs.get('tex_ids', []):
            copy_env.texture_modder.textures[tex_id].bitmap[:] = self.texture_modder.textures[tex_id].bitmap
        
        # match qpos and qvel
        qpos, qvel = self.get_env_state()['qpos'], self.get_env_state()['qvel']
        copy_env.set_state(qpos=qpos, qvel=qvel)
        copy_env.set_sim_obsd_state(qpos=qpos, qvel=qvel)
        copy_env.set_obj_goal(self.obj_goal)
        
        # match lightings
        copy_env.sim.model.light_specular[:] = self.sim.model.light_specular
        copy_env.sim.model.light_diffuse[:] = self.sim.model.light_diffuse
        copy_env.sim.model.light_ambient[:] = self.sim.model.light_ambient

        # match body pose
        copy_env.sim.model.body_pos[:] = self.sim.model.body_pos
        copy_env.sim.model.body_quat[:] = self.sim.model.body_quat

        return copy_env

    def set_goal(
        self, 
        expert_traj, 
        cameras=['left_cam', 'right_cam'], 
        goal_window=5, 
        frame_size=(256, 256),
        min_success_count=5,
        device_id=0,
        max_trials=10,
        verbose=False,
        ):
        """
        Use for online evaluation, such that the goal image gets applied the sample randomization.
        Call env.set_goal(**kwargs) after every self.reset(). Note that this method calls another reset() to the randomization 
        Input takes in a single expert trajectory, replay the actions to render a goal image
        """
        assert type(expert_traj) is dict, "Expert trajectory must be a dictionary"
        expert_actions = expert_traj['actions']
        horizon = expert_actions.shape[0]
        goal_tstep = self.np_random.randint(low=horizon-goal_window, high=horizon)
        new_camera_imgs = None 
        success_count = 0 
        goal_set = False 

        
        init_state = {key: v[0] for key, v in expert_traj['env_states'].items()}
        self.reset(reset_qpos=init_state['qpos'], reset_qvel=init_state['qvel'])
  
        for trial in range(max_trials):
            goal_env = self.make_copy_env()
            for t, action in enumerate(expert_actions): 
                next_o, rwd, done, next_env_info = goal_env.step(action)
                if next_env_info.get('solved', False):
                    success_count += 1
                if t == goal_tstep and success_count >= min_success_count:
                    goal_set = True
                    curr_frame = goal_env.render_camera_offscreen(
                        sim=goal_env.sim,
                        cameras=cameras,
                        width=frame_size[0],
                        height=frame_size[1],
                        device_id=device_id
                        ) 
                    new_camera_imgs = {cam: curr_frame[i] for i, cam in enumerate(cameras)}
            if verbose:
                print("Trial {}: {}".format(trial, success_count))
            del goal_env
            if goal_set:
                break

        if not goal_set:
            raise ValueError("Failed to complete the task by replaying the expert actions")
        self.goal_imgs = new_camera_imgs
            
        return new_camera_imgs
