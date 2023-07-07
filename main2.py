#%%
# from utils2 import *
from utils_b import *

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

env_path = "./envs/random_envs"

# Pick a random environment to genrate single control policy
env, info, env_path = load_random_env(env_path)

# Define the robot class
beeBot = dp(env,info)

# call the value iteration function
v,p,s,t = beeBot.algo_dp()
#%%

# Once the policy p, termination time od dp is received this is used to run all environements to get the map and optimal path

env_folder = "./envs/random_envs"
env_list = [os.path.join(env_folder, env_file) for env_file in os.listdir(env_folder) if env_file.endswith(".env")]
t_horizon = 18431
seq_sav = []

for env_p in range(len(env_list)):
    env, info = load_random_env_all(env_list[env_p])
    rob = dp(env,info)
    
    goal = info["goal_pos"]

    curr_pos = np.array(env.agent_pos)
    curr_state = np.array(rob.robot_state())

    cp = {0:"MF",1:"TL",2:"TR",3:"PK",4:"UD"}

    opt_polic = []
    seq = []
    termi=t
    
    while(termi<t_horizon):

        sta_idx = np.where((s == curr_state).all(axis=1))[0][0]

        action = p[sta_idx,termi]

        opt_polic.append(cp[action])
        seq.append(action)

        x_t_1 = rob.motion_model_robot(curr_state,cp[action])
        curr_pos = np.array([x_t_1[0],x_t_1[1]])
        curr_state = x_t_1
        termi=termi+1
        if(curr_pos[0] == goal[0] and curr_pos[1] == goal[1]):
            
            sta_idx = np.where((s == curr_state).all(axis=1))[0][0]
            action = p[sta_idx,termi]
            opt_polic.append(cp[action])
            seq.append(action)
            break
        
    seq_sav.append(opt_polic)
    print(opt_polic)
    draw_gif_from_seq(seq, env,'./gif/doorkey_rand_8-'+str(env_p)+'-new.gif')

with open('OptimalPaths.txt', 'w') as f:
    f.write(str(seq_sav))