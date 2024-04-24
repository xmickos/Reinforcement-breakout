import sys
from agent import Agent
sys.path.append('/Users/xmickos/miniconda3/lib/python3.10/site-packages')

import os
import torch
import gym
from breakout import *
# ale-import-roms
import ale_py
from model import AtariNet

print(gym.__version__)
print(ale_py.__version__)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

enviroment = DQNBreakout(device=device, render_mode='human')

model = AtariNet(no_actions=4)

model.load_the_model("/Users/xmickos/Documents/DIPLOM/07_04_2024/models/latest_23_04.pt")

agent = Agent(model=model,
              device=device,
              epsilon=0.001,
              nb_warmup=5000,
              nb_actions=4,
              learning_rate=0.00001,
              memory_capacity=200000,
              batch_size=64)

# agent.train(env=enviroment, epochs=200000)

agent.test(env=enviroment)



