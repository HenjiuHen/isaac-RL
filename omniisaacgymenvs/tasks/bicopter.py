import math

import numpy as np
import torch
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.bicopter import Bicopter
from omniisaacgymenvs.tasks.bicopter_view import BicopterView

class BicopterTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._num_actions=4#动作空间：推力大小，旋转方向
        self._num_observations=21
        RLTask.__init__(self, name=name, env=env)

        self.max_thrust= 40.0#最大推力
        self.thrust_lower_limits=torch.zeros(2,device=self._device,dtype=torch.float32)#推力下限       
        self.thrust_upper_limits=self.max_thrust*torch.ones(2,device=self._device,dtype=torch.float32)#推力下限

        return

    def update_config(self, sim_config):
        self._sim_config=sim_config
        self._cfg=sim_config.config
        self._task_cfg=sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self._bicopter_positions=torch.tensor([0.0,0.0,0.5])#初始位置
        self._ball_positions=torch.tensor([0.0,0.0,2.0])#球初始位置
        self.dt = self._task_cfg["sim"]["dt"]

        return       
    
    def set_up_scene(self, scene, ) -> None:
        self.get_bicopter()
        self.get_target()
        super().set_up_scene(scene)
        self._bicopter=BicopterView(
            prim_paths_expr="/World/envs/.*/bicopter",name="bicopter_view"
            )
        self._balls=RigidPrimView(
            prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False
        )
        self._balls._non_root_link=True
        scene.add(self._bicopter)
        scene.add(self._bicopter.propeller)
        scene.add(self._balls)
        return
    
    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("bicopter_view"):
            scene.remove_object("bicopter_view", registry_only=True)
        if scene.object_exists("propeller_view"):
            scene.remove_object("propeller_view", registry_only=True)
        if scene.object_exists("targets_view"):
            scene.remove_object("targets_view", registry_only=True)
        self._bicopter = BicopterView(prim_paths_expr="/World/envs/.*/bicopter", name="bicopter_view")
        self._balls = RigidPrimView(
            prim_paths_expr="/World/envs/.*/ball", name="targets_view", reset_xform_properties=False
        )
        scene.add(self._bicopter)
        scene.add(self._bicopter.propeller)
        scene.add(self._balls)

    def get_bicopter(self):
        bicopter=Bicopter(
            prim_path=self.default_zero_env_path+"/bicopter",name="bicopter",translation=self._bicopter_positions
        )
        self._sim_config.apply_articulation_settings(
            "bicopter", get_prim_at_path(bicopter.prim_path), self._sim_config.parse_actor_config("bicopter")
        )
    
    def get_target(self):
        radius = 0.05
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            name="target_0",
            radius=radius,
            color=color,
            translation=self._ball_positions
        )
        self._sim_config.apply_articulation_settings(
            "ball", get_prim_at_path(ball.prim_path), self._sim_config.parse_actor_config("ball")
        )
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:
        self.root_pos, self.root_rot=self._bicopter.get_world_poses(clone=False)
        self.root_velocities=self._bicopter.get_velocities(clone=False)
        self.dof_pos = self._bicopter.get_joint_positions(clone=False)

        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_linvels = self.root_velocities[:, :3]#线速度
        root_angvels = self.root_velocities[:, 3:]#角速度

        self.obs_buf[..., 0:3] = (self.target_positions - root_positions) / 3
        self.obs_buf[..., 3:7] = root_quats
        self.obs_buf[..., 7:10] = root_linvels / 2
        self.obs_buf[..., 10:13] = root_angvels / math.pi
        self.obs_buf[..., 13:15] = self.dof_pos

        observations = {self._bicopter.name: {"obs_buf": self.obs_buf}}
        return observations
    
    def pre_physics_step(self, actions):
        if not self._env._world.is_playing():
            return
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        actions = actions.clone().to(self._device)
        self.action=actions
        dof_action_speed_scale = 1 * math.pi#角速度
        self.dof_position_targets += self.dt * dof_action_speed_scale * actions[:, 0:2]
        self.dof_position_targets[:] = tensor_clamp(
            self.dof_position_targets, self.dof_lower_limits, self.dof_upper_limits
        )
        thrust_action_speed_scale = 400#推力改变速度
        self.thrusts += self.dt * thrust_action_speed_scale * actions[:, 2:4]
        self.thrusts[:] = tensor_clamp(self.thrusts, self.thrust_lower_limits, self.thrust_upper_limits)

        self.forces[:, 0, 2] = self.thrusts[:, 0]
        self.forces[:, 1, 2] = self.thrusts[:, 1]
        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0
        self.dof_position_targets[reset_env_ids] = self.dof_pos[reset_env_ids]

        self._bicopter.set_joint_positions(self.dof_position_targets)
        self._bicopter.propeller.apply_forces(self.forces, is_global=False)

        return 

    def post_reset(self):
        # control tensors
        self.dof_position_targets = torch.zeros(
            (self._num_envs, self._bicopter.num_dof), dtype=torch.float32, device=self._device, requires_grad=False
        )
        self.thrusts = torch.zeros((self._num_envs, 2), dtype=torch.float32, device=self._device, requires_grad=False)
        self.forces = torch.zeros(
            (self._num_envs, self._bicopter.propeller.count // self._num_envs, 3),
            dtype=torch.float32,
            device=self._device,
            requires_grad=False,
        )

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device)
        self.target_positions[:, 2] = 2.0
        self.actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.root_pos, self.root_rot = self._bicopter.get_world_poses(clone=False)
        self.root_velocities = self._bicopter.get_velocities(clone=False)
        self.dof_pos = self._bicopter.get_joint_positions(clone=False)
        self.dof_vel = self._bicopter.get_joint_velocities(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        dof_limits = self._bicopter.get_dof_limits()
        self.dof_lower_limits = dof_limits[0][:, 0].to(device=self._device)
        self.dof_upper_limits = dof_limits[0][:, 1].to(device=self._device)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_pos[env_ids, :] = torch_rand_float(-0.2, 0.2, (num_resets, self._bicopter.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-1, 1, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-1, 1, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.2, 2, (num_resets, 1), device=self._device).view(-1)
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._bicopter.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._bicopter.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        self._bicopter.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._bicopter.set_velocities(root_velocities[env_ids], indices=env_ids)

        self._balls.set_world_poses(positions=self.target_positions[:, 0:3] + self._env_pos)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def calculate_metrics(self) -> None:
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_angvels = self.root_velocities[:, 3:]
        root_speed=self.root_velocities[:, :3]
        
        # distance to target
        target_dist = torch.sqrt(torch.square(self.target_positions - root_positions).sum(-1))
        pos_reward = 1.0 / (1.0 +3*target_dist * target_dist)  # 2
        self.target_dist = target_dist
        self.root_positions = root_positions


        euler_RPY=get_euler_xyz(root_quats)
        roll=euler_RPY[0]
        pitch=euler_RPY[1]
        roll_rew=roll*roll
        pitch_rew=(2*math.pi-pitch)*(2*math.pi-pitch)
        ver_rew=0.1*(roll_rew+pitch_rew)
        # uprightness
        #ups = quat_axis(root_quats, 2)
        #tiltage = torch.abs(1 - ups[..., 2])
        #up_reward = 1.0 / (1.0 + 10 * tiltage * tiltage)
        ups = quat_axis(root_quats, 2)
        self.orient_z = ups[..., 2]
        up_reward = 2*torch.clamp(ups[..., 2], min=0.0, max=1.0)

        # effort reward
        effort = torch.square(self.actions).sum(-1)
        effort_reward = 0.05 * torch.exp(-0.5 * effort)

        dof_pos_re=(10*self.dof_pos*self.dof_pos)
        dof_pos_rew=torch.exp(-1.0*(dof_pos_re[:,0]+dof_pos_re[:,1]))

        speed = torch.square(root_speed).sum(-1)
        speed_reward = 0.3 * torch.exp(-1.0 * speed)
        # spin reward
        spin = torch.square(root_angvels).sum(-1)
        spin_reward = torch.exp(-1.0 * spin)
        spinnage = torch.abs(root_angvels[..., 2])
        rollnage= torch.abs(root_angvels[..., 0])
        ang=torch.abs(root_angvels)
        rollnage_reward=1.0 / (1.0 + 0.1 * rollnage * rollnage)
        spinnage_reward = 1.0 / (1.0 + 0.1 * spinnage * spinnage)
        ang_reward = (1.0 / (1.0 + 0.1 * ang * ang)).sum(-1)
        RS_reward=rollnage_reward+spinnage_reward+spinnage * spinnage * (-1 / 400)+rollnage * rollnage * (-1 / 400)
        #spin_reward =  .0/(1.0+spin)
        # spinning
        #spinnage = torch.abs(root_angvels[..., 2])
        #spinnage_reward = 1.0 / (1.0 + 0.1 * spinnage * spinnage)

        rew = 2*pos_reward + pos_reward *(up_reward + spin_reward+ang_reward+ang.sum(-1)*ang.sum(-1)*(-1 / 400))-effort_reward
        rew = torch.clip(rew, 0.0, None)
        self.rew_buf[:] = rew

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > 5.0, ones, die)
        die = torch.where(self.root_positions[..., 2] < 0.3, ones, die)
        die = torch.where(self.orient_z < 0.0, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)