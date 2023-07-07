# from utils2 import *
from utils import *
import yaml

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

with open("config/config.yml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

env_path = config["ENV_PATH"]
fname = config["GIF_PATH"]
# print(env_path)
env, info = load_env(env_path)

###############ENV#######################

beeBot = dp(env,info)

v,p,s,t = beeBot.algo_dp()

goal = info["goal_pos"]

curr_pos = np.array(env.agent_pos)
curr_state = np.array(beeBot.robot_state())
curr_state = np.array(beeBot.robot_state())

cp = {0:"MF",1:"TL",2:"TR",3:"PK",4:"UD"}

opt_polic = []
seq = []

while(t<beeBot.T_horiz):

    sta_idx = np.where((s == curr_state).all(axis=1))[0][0]

    action = p[sta_idx,t]

    opt_polic.append(cp[action])
    seq.append(action)

    x_t_1 = beeBot.motion_model_robot(curr_state,cp[action])
    curr_pos = np.array([x_t_1[0],x_t_1[1]])
    curr_state = x_t_1
    t=t+1
    if(curr_pos[0] == goal[0] and curr_pos[1] == goal[1]):
        sta_idx = np.where((s == curr_state).all(axis=1))[0][0]

        action = p[sta_idx,t]

        opt_polic.append(cp[action])
        seq.append(action)
        break
draw_gif_from_seq(seq, env,fname + '.gif')
