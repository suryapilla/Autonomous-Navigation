# Shortest optimal path using Dynamic Programming

## Objective
To implement dynammic programming for the Door-Key problems.
<p align="center">
<img src="gif/doorkey_rand_8--8.gif" alt="Door-key Problem" width="500"/></br>
</p>

There are 7 test scenes you have to test and include in the report.

|           doorkey-5x5-normal            |
| :-------------------------------------: |
| <img src="envs/known_envs/doorkey-5x5-normal.png"> |

|           doorkey-6x6-normal            |            doorkey-6x6-direct            |            doorkey-6x6-shortcut            |
| :-------------------------------------: | :--------------------------------------: | :----------------------------------------: |
| <img src="envs/known_envs/doorkey-6x6-normal.png"> | <img src="envs/known_envs/doorkey-6x6-direct.png" > | <img src="envs/known_envs/doorkey-6x6-shortcut.png" > |

|           doorkey-8x8-normal            |            doorkey-8x8-direct            |            doorkey-8x8-shortcut            |
| :-------------------------------------: | :--------------------------------------: | :----------------------------------------: |
| <img src="envs/known_envs/doorkey-8x8-normal.png"> | <img src="envs/known_envs/doorkey-8x8-direct.png" > | <img src="envs/known_envs/doorkey-8x8-shortcut.png" > |

## Installation

- Install Python version `3.7 ~ 3.10`
- Install dependencies
```bash
conda create -n env_shortestPath python=3.10
conda activate env_shortestPath
git clone  https://github.com/suryapilla/Autonomous-Navigation.git
cd Autonomous-Navigation
pip install -r requirements.txt
```

## Code
### 1. The below command is for known environments
```
python main.py
```
<p align="center">
<img src="gif/doorkey_rand_8--8.gif" alt="Door-key Problem" width="500"/></br>
</p>

### 2. The below command is for the shortest path in unkown environmnet

```
python main2.py
```
<p align="center">
<img src="gif/doorkey_rand_8-34.gif" alt="Door-key Problem" width="500"/></br>
</p>


