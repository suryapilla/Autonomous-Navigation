import os
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
import imageio
import random
from minigrid.core.world_object import Goal, Key, Door
import itertools
from itertools import product
import pdb


MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door
class robot_env:
    def __init__(self,env,info) -> None:
        self.env = env
        self.info = info
        
    def robot_state(self):
        """_summary_
        This is the initial state of robot

        Args:
            rob_xy_pos (_type_): env.agent_pos
            rob_dir (_type_): env.agent_dir
            key_stat (int, optional): env.carrying is not None
            door_open_stat (int, optional): door.is_open
            door_lock_stat (int, optional): door.is_locked

        Returns:
            _type_: _description_
        """
        self.x_pos = self.env.agent_pos[0]
        
        self.y_pos = self.env.agent_pos[1]
        
        self.orient = self.env.agent_dir # 0--> RIGHT; 1--> DOWN, 2--> LEFT, 3-->UP 
        
        self.key_stat = self.env.carrying is not None
        
        door = self.env.grid.get(self.info["door_pos"][0], self.info["door_pos"][1])
        
        # self.door_open_stat = door.is_open
        
        self.door_lock_stat = door.is_locked
        
        return self.x_pos, self.y_pos, self.orient, self.key_stat, self.door_lock_stat

    def robot_map(self):
        # x:Col number, y:Row number
        self.grid_size = self.env.width
        self.x_goal = self.info["goal_pos"][0]
        self.y_goal = self.info["goal_pos"][1]
        self.x_start = self.env.agent_pos[0]
        self.y_start = self.env.agent_pos[1]
        self.x_key = self.info["key_pos"][0]
        self.y_key = self.info["key_pos"][1]
        self.x_door = self.info["door_pos"][0]
        self.y_door = self.info["door_pos"][1]
        
        self.xy_wall = []
        for i in range(self.env.width):
            for j in range(self.env.height):
                cell_type = self.env.grid.get(i,j)
                # print(cell_type.type)
                if cell_type is not None and cell_type.type == "wall":
                    self.xy_wall.append([i,j])

        return self.x_goal, self.y_goal, self.x_start, self.y_start, self.x_key, self.y_key, self.x_door, self.y_door, self.xy_wall
    
    def isWall(self,pos):
        self.robot_map()
        if pos in self.xy_wall:
            return True
        else:
            return False   

class dp(robot_env):
    def __init__(self,env,info):
        super().__init__(env,info)
        self.robot_map()
        self.init_state = self.robot_state()
        self.T_horiz = self.env.height*self.env.width*4*2*2
        self.len_state_space = self.env.height*self.env.width*4*2*2
        self.val_t = np.ones([self.len_state_space,self.T_horiz])*np.inf
        self.pi = -1*np.ones([self.len_state_space,self.T_horiz],dtype=int)
        self.Q_t = np.ones((self.T_horiz,5))
        self.actions = ["MF","TL","TR","PK","UD"]
        self.N = self.env.width
        
    def state_space(self):
        
        x = np.arange(0, self.N)
        y = np.arange(0, self.N)
        orient = np.arange(0, 4)
        door_stat = np.array([0, 1])
        key_stat = np.array([0, 1])
        state_space_N = np.array(list(product(x, y, orient, key_stat, door_stat)))
        return state_space_N

    # State: x_pos, y_pos, orient, key_stat(0: there, 1 not there), self.door_lock_stat
    # def test(self):
    #     print(self.isWall([3,3]))
        
    def motion_model_robot(self,state,action):
        # 0--> RIGHT; 1--> DOWN, 2--> LEFT, 3-->UP
        self.prev_state = state.copy()
        self.curr_state = state.copy()
        
        if action == "MF" and state[2] == 0:
            if (state[0] + 1 == self.x_key and state[1] == self.y_key) and state[3]==0:
                self.curr_state[0] = state[0]
            elif (state[0] + 1 == self.x_door and state[1] == self.y_door) and state[4]==1:
                self.curr_state[0] = state[0]
            elif self.isWall([state[0]+1, state[1]]):
                self.curr_state[0] = state[0]
            else:
                self.curr_state[0] = min(state[0] + 1, self.N -1)
            
        elif action == "MF" and state[2] == 1:
            if (state[0] == self.x_key and state[1] + 1 == self.y_key) and state[3]==0:
                self.curr_state[1] = state[1]
            elif (state[0] == self.x_door and state[1] + 1 == self.y_door) and state[4]==1:
                self.curr_state[1] = state[1]
            elif self.isWall([state[0], state[1]+1]):
                self.curr_state[1] = state[1]
            else:
                self.curr_state[1] = min(state[1] + 1,self.N -1)
        
        elif action == "MF" and state[2] == 2:
            if (state[0] - 1 == self.x_key and state[1] == self.y_key) and state[3]==0:
                self.curr_state[0] = state[0]
            elif (state[0] - 1 == self.x_door and state[1] == self.y_door) and state[4]==1:
                self.curr_state[0] = state[0]
            elif self.isWall([state[0] - 1,state[1]]):
                self.curr_state[0] = state[0]
            else:
                self.curr_state[0] = max(state[0] - 1,0)
        
        elif action == "MF" and state[2] == 3:
            if (state[0] == self.x_key and state[1] - 1 == self.y_key) and state[3]==0:
                self.curr_state[1] = state[1]
            elif (state[0] == self.x_door and state[1] - 1 == self.y_door) and state[4]==1:
                self.curr_state[1] = state[1]
            elif self.isWall([state[0],state[1]-1]):
                self.curr_state[1] = state[1]
            else:
                self.curr_state[1] = max(state[1] - 1,0)
        
        elif action == "TL":
            self.curr_state[2] = (state[2]-1)%4
        
        elif action == "TR":
            self.curr_state[2] = (state[2]+1)%4
            
        elif action == "PK" :
            if state[2] == 0: #looking right
                if self.x_key == min(state[0] + 1,self.N -1) and self.y_key == state[1]:
                    self.curr_state[3] = 1
            elif state[2] == 1: #looking down
                if self.x_key == state[0] and self.y_key == min(state[1] + 1, self.N -1):
                    self.curr_state[3] = 1
            elif state[2] == 2: #looking left
                if self.x_key == max(state[0]-1,0) and self.y_key == state[1]:
                    self.curr_state[3] = 1
            elif state[2] == 3: #looking up
                if self.x_key == state[0] and self.y_key == max(state[1]-1,0):
                    self.curr_state[3] = 1
            
        elif action == "UD":
            
            if state[2] == 0 and state[3]==1: #looking right
                if self.x_door == min(state[0] + 1,self.N -1) and self.y_door == state[1]:
                    self.curr_state[4] = 0
            elif state[2] == 1 and state[3]==1: #looking down
                if self.x_door == state[0] and self.y_door == min(state[1] + 1,self.N -1):
                    self.curr_state[4] = 0
            elif state[2] == 2 and state[3]==1: #looking left
                if self.x_door == max(state[0]-1,0) and self.y_door == state[1]:
                    self.curr_state[4] = 0
            elif state[2] == 3 and state[3]==1: #looking up
                if self.x_door == state[0] and self.y_door == max(state[1]-1,0):
                    self.curr_state[4] = 0
        
        if action=="UD" or action=="PK":
            self.prev_state[3] = self.curr_state[3]
            self.prev_state[4] = self.curr_state[4]
            return self.prev_state
        
        else:
            return self.curr_state
        
    def stage_cost(self,state,action):
        self.nxt_state = state.copy()
        
        
        if state[2] == 0:
            self.nxt_state[0] = min(self.nxt_state[0] + 1, self.N - 1)

        elif state[2] == 1:
            self.nxt_state[1] = min(self.nxt_state[1] + 1, self.N - 1)


        elif state[2] == 2:
            self.nxt_state[0] = max(self.nxt_state[0] - 1, 0)

        elif state[2] == 3:
            self.nxt_state[1] = max(self.nxt_state[1] - 1, 0)
        
        if action == "MF":
            
            if self.isWall([self.nxt_state[0],self.nxt_state[1]]):
                
                return np.inf
            
            else:
                
                if self.nxt_state[0] == self.x_key and self.nxt_state[1] == self.y_key and state[3] == 0:
                    return np.inf
                
                elif self.nxt_state[0] == self.x_door and self.nxt_state[1] == self.y_door and state[4] == 1:
                    return np.inf
                
                else:
                    return 1


        elif action == "TL" or action == "TR":
            return 1

        elif action == "PK":
            
            if self.nxt_state[0] == self.x_key and self.nxt_state[1] == self.y_key and state[3]==0 and state[4] == 1:
                return 1
            else:
                return np.inf
  
            
        elif action == "UD":
            
            if self.nxt_state[0] == self.x_door and self.nxt_state[1] == self.y_door and state[3]==1 and state[4] == 1:
                return 1
            else:
                return np.inf            

    def algo_dp(self):
        actions = ["MF","TL","TR","PK","UD"]
        self.state_space_N = self.state_space()

        orient, key_stat, door_stat = np.array(list(product(range(4), range(2), range(2)))).T
        goal_state = np.column_stack([self.x_goal*np.ones(orient.shape), self.y_goal*np.ones(orient.shape), orient, key_stat, door_stat])
        goal_idx = np.where((self.state_space_N == goal_state[:, None]).all(-1))[1]
        self.val_t[goal_idx,self.T_horiz - 1] = -1
        self.Q_t[goal_idx,:] = -1

        for i in range(self.T_horiz-2,-1,-1):
                        
            for action in range(5):
                
                x_t = self.state_space_N.copy()
                
                l_t = np.array([self.stage_cost(xi,actions[action]) for xi in self.state_space_N])
                
                x_t_1 = np.array([self.motion_model_robot(X,actions[action]) for X in x_t])
                
                g_index = [np.where((self.state_space_N == elem).all(axis=1))[0][0] for elem in x_t_1]
                
                self.Q_t[:,action] = l_t + self.val_t[g_index,i+1]
                self.Q_t[goal_idx,:] = -1
            
            self.val_t[:,i] = np.min(self.Q_t,axis=1)
            self.pi[:,i] = np.argmin(self.Q_t,axis=1)
            
            self.t = 0
            if np.array_equal(self.val_t[:,i],self.val_t[:,i+1]):
                self.t = i
                print("Converged",i)
                break

        return self.val_t, self.pi, self.state_space_N,self.t
            
def step_cost(action):
    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************
    return 0  # the cost of action


def step(env, action):
    """
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    """
    actions = {
        0: env.actions.forward,
        1: env.actions.left,
        2: env.actions.right,
        3: env.actions.pickup,
        4: env.actions.toggle,
    }

    (obs, reward, terminated, truncated, info) = env.step(actions[action])
    return step_cost(action), terminated


def generate_random_env(seed, task):
    """
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    """
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task, render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def load_env(path):
    """
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    # print(path)
    with open(path, "rb") as f:
        # pdb.set_trace()
        env = pickle.load(f)

    info = {"height": env.height, "width": env.width, "init_agent_pos": env.agent_pos, "init_agent_dir": env.dir_vec}

    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info


def load_random_env(env_folder):
    """
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    """
    env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder) if env_file.endswith(".env")]
    env_path = random.choice(env_list)
    
    with open(env_path, "rb") as f:
        env = pickle.load(f)

    info = {
        "height": env.height,
        "width": env.width,
        "init_agent_pos": env.agent_pos,
        "init_agent_dir": env.dir_vec,
        "door_pos": [],
        "door_open": [],
    }

    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i), Key):
                info["key_pos"] = np.array([j, i])
            elif isinstance(env.grid.get(j, i), Door):
                info["door_pos"].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info["door_open"].append(True)
                else:
                    info["door_open"].append(False)
            elif isinstance(env.grid.get(j, i), Goal):
                info["goal_pos"] = np.array([j, i])

    return env, info, env_path


def save_env(env, path):
    with open(path, "wb") as f:
        pickle.dump(env, f)


def plot_env(env):
    """
    Plot current environment
    ----------------------------------
    """
    img = env.render()
    plt.figure()
    plt.imshow(img)
    plt.pause(1)
    # plt.show()


def draw_gif_from_seq(seq, env, path="./gif/doorkey.gif"):
    """
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    """
    with imageio.get_writer(path, mode="I", duration=0.8) as writer:
        img = env.render()
        writer.append_data(img)
        for act in seq:
            img = env.render()
            step(env, act)
            writer.append_data(img)
    print(f"GIF is written to {path}")
    return
