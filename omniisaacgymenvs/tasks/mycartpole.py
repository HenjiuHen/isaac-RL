import torch
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
import math
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np


#负责重置环境、执行动作、收集观察结果
class mycartpoleTask(RLTask):
    def __init__(self, name,sim_config, env, offset=None) -> None:
        #name:类的名称，从配置文件yaml中解析
        #sim_config:包含从任务配置文件中解析的任务和物理参数，例如任务环境的数量、物理参数、智能体的物理属性
        #env:环境对象，由脚本定义
        self.update_config(sim_config)#解析任务配置文件
        self._max_episode_lengtjn=500#最大迭代次数

        self._num_observations=4#初始化观测空间
        self._num_actions=1#初始化动作空间

        RLTask.__init__(self,name,env)#调用父类构造函数初始化RL任务通用变量，随机化设置、定义动作和观察空间、环境设置、定义缓冲区

    def update_config(self,sim_config):#解析任务配置文件
        #从任务配置文件提取任务配置信息
        self._sim_config=sim_config
        self._cfg=sim_config.config#config.yaml
        self._task_cfg=sim_config.task_config#task/cartpole.yaml

        #解析任务配置参赛
        self._num_envs=self._task_cfg['env']['numEnvs']#环境数量
        self._env_spacing=self._task_cfg['env']['envSpacing']#？
        self._cartpole_positions=torch.tensor([0.0,0.0,2.0])#初始位置

        #重置和动作相关参赛
        self._reset_dist=self._task_cfg['env']['resetDist']#触发重置距离？
        self._max_push_effort=self._task_cfg['env']['maxEffort']#?

    def set_up_scene(self,scene):#定义场景
        self.get_cartpole()#创建单个环境
        super().set_up_scene(scene)#调用父函数克隆环境
        #创建一个ArticulationView来保存环境集合??
        self._cartpoles=ArticulationView(
            prim_paths_expr="/World/envs/.*/mycartpole",name='mycartpole_view',reset_xform_properties=False#erroe:/World/env/.*/ --> /World/envs/.*/
        )
        scene.add(self._cartpoles)#将ArticulationView注册到场景中以便初始化

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("mycartpole_view"):
            scene.remove_object("mycartpole_view", registry_only=True)
        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/mycartpole", name="mycartpole_view", reset_xform_properties=False
        )
        scene.add(self._cartpoles)

    def get_cartpole(self):
        cartpole=Cartpole(#添加一个机器人
            prim_path=self.default_zero_env_path+"/mycartpole",name="mycartpole",translation=self._cartpole_positions
        )

        #应用配置文件中的铰链设置
        self._sim_config.apply_articulation_settings(
            'Cartpole',get_prim_at_path(cartpole.prim_path),self._sim_config.parse_actor_config('Cartpole')
        )

    def post_reset(self) -> None:
        #获取杆和车的关节自由度指数
        self._cart_dof_idx=self._cartpoles.get_dof_index('cartJoint')
        self._pole_dof_idx=self._cartpoles.get_dof_index('poleJoint')

        #随机化所以环境
        indices=torch.arange(self._cartpoles.count,dtype=torch.int64,device=self._device)
        self.reset_idx(indices)

    def reset_idx(self,env_ids):#重置环境，接受参数应重置环境的索引
        num_resets=len(env_ids)

        #随机化关节位置
        dof_pos=torch.zeros((num_resets,self._cartpoles.num_dof),device=self._device)
        dof_pos[:,self._cart_dof_idx]=1.0*(1.0-2.0*torch.rand(num_resets,device=self._device))
        dof_pos[:,self._pole_dof_idx]=0.125*math.pi*(1.0-2.0*torch.rand(num_resets,device=self._device))

        #随机化关节速度
        dof_vel=torch.zeros((num_resets,self._cartpoles.num_dof),device=self._device)
        dof_vel[:,self._cart_dof_idx]=0.5*(1.0-2.0*torch.rand(num_resets,device=self._device))
        dof_vel[:,self._pole_dof_idx]=0.25*math.pi*(1.0-2.0*torch.rand(num_resets,device=self._device))

        #将随机的关节速度和位置应用在环境中
        indices=env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos,indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel,indices=indices)

        #环境重置后重置重置标识区和步骤标识区
        self.reset_buf[env_ids]=0#0表示不需要重置，1表示需要重置
        self.progress_buf[env_ids]=0#表示步骤数

    def pre_physics_step(self, actions):#模拟步骤，执行动作和重置环境
        if not self._env._world.is_playing():#确保UI中的模拟没有停止 error:self._env_world --> self._env._world
            return
        
        #提取需要重置的环境索引并重置
        reset_env_ids=self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids)>0:
            self.reset_idx(reset_env_ids)
        
        actions=actions.to(self._device)

        #计算动作力
        forces= torch.zeros((self._cartpoles.count,self._cartpoles.num_dof),dtype=torch.float32,device=self._device)#【智能体数量，每个智能体自由度个数】error：dtype=torch.int32 --> dtype=torch.float32
        forces[:,self._cart_dof_idx]=self._max_push_effort*actions[:,0]

        #将动作应用于环境
        indices=torch.arange(self._cartpoles.count,dtype=torch.int32,device=self._device)
        self._cartpoles.set_joint_efforts(forces,indices=indices)

    def get_observations(self) -> dict:#获取状态
        #检索关节位置和速度
        dof_pos=self._cartpoles.get_joint_positions(clone=False)
        dof_vel=self._cartpoles.get_joint_velocities(clone=False)
        cart_pos=dof_pos[:,self._cart_dof_idx]
        cart_vel=dof_vel[:,self._cart_dof_idx]
        pole_pos=dof_pos[:,self._pole_dof_idx]
        pole_vel=dof_vel[:,self._pole_dof_idx]
        
        #填入观测缓冲区
        self.obs_buf[:,0]=cart_pos
        self.obs_buf[:,1]=cart_vel
        self.obs_buf[:,2]=pole_pos
        self.obs_buf[:,3]=pole_vel

        #构建观测字典并返回
        observations={self._cartpoles.name:{'obs_buf':self.obs_buf}}
        return observations
    
    def calculate_metrics(self) -> dict:#计算分数，奖励函数
        cart_pos=self.obs_buf[:,0]
        cart_vel=self.obs_buf[:,1]
        pole_pos=self.obs_buf[:,2]
        pole_vel=self.obs_buf[:,3]

        #根据智能体杆的角度和车的速度计算奖励
        reward=1.0-pole_pos*pole_pos-0.01*torch.abs(cart_vel)-0.5*torch.abs(pole_vel)
        #车移动的太远进行惩罚
        reward=torch.where(torch.abs(cart_pos)>self._reset_dist,torch.ones_like(reward)*-2.0,reward)#true返回torch.ones_like(reward)*-2.0，false返回reward
        #杆移动超过90度进行惩罚
        reward=torch.where(torch.abs(pole_pos)>np.pi/2,torch.ones_like(reward)*-2.0,reward)
        self.rew_buf[:]=reward

    def is_done(self) -> bool:
        cart_pos=self.obs_buf[:,0]
        pole_pos=self.obs_buf[:,2]
        #检测是否满足重置条件，并标记满足条件的环境
        resets=torch.where(torch.abs(cart_pos)>self._reset_dist,1,0)
        resets=torch.where(torch.abs(pole_pos)>math.pi/2,1,resets)
        resets=torch.where(self.progress_buf>self._max_episode_lengtjn,1,resets)

        #放入重置缓冲区
        self.reset_buf[:]=resets