import math
import random
import sys
import os
import shutil
import numpy as np
from collections import namedtuple
from itertools import count
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from game import Engine

width, height = 10, 20
engine = Engine(width, height)

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor
ByteTensor = torch.ByteTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

"""
Replay Memory
    Transition - a named tuple representing a single transition in
        our environment
    ReplayMemory - a cyclic buffer of bounded size that holds the
        transitions observed recently. It also implements a ``.sample()``
        method for selecting a random batch of transitions for training.
"""
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.lin1 = nn.Linear(768, 256)
        self.head = nn.Linear(256, engine.nb_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.lin1(x.view(x.size(0), -1)))
        return self.head(x.view(x.size(0), -1))

BATCH_SIZE = 2048
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
CHECKPOINT_FILE = 'checkpoint.pth.tar'


steps_done = 0

model = DQN()
print(model)

loss = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(8000)


def select_action(state):
    """
        Will select an action accordingly to an epsilon
        greedy policy. Simply put, we'll sometimes use our model for choosing
        the action, and sometimes we'll just sample one uniformly. The
        probability of choosing a random action will start at ``EPS_START``
        and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
        controls the rate of the decay.
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        return model(
            Variable(state, volatile=True).type(FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return FloatTensor([[random.randrange(engine.nb_actions)]])


episode_durations = []

last_sync = 0

def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    state_action_values = model(state_batch).gather(1, action_batch)

    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(FloatTensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    if len(loss.data)>0 : return loss.data[0] 
    else : return loss

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        memory = checkpoint['memory']
    except Exception as e:
        pass

    return checkpoint['epoch'], checkpoint['best_score']

if __name__ == '__main__':
    start_epoch = 0
    best_score = 0
    if len(sys.argv) > 1 and sys.argv[1] == 'continue':
        if len(sys.argv) > 2:
            CHECKPOINT_FILE = sys.argv[2]
        if os.path.isfile(CHECKPOINT_FILE):
            print("=> loading checkpoint '{}'".format(CHECKPOINT_FILE))
            start_epoch, best_score = load_checkpoint(CHECKPOINT_FILE)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(CHECKPOINT_FILE, start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(CHECKPOINT_FILE))

    f = open('log.out', 'w+')
    for i_episode in count(start_epoch):
        state = FloatTensor(engine.clear()[None,None,:,:])

        score = 0
        for t in count():
            action = select_action(state).type(LongTensor)

            last_state = state
            state, reward, done = engine.step(action[0,0])
            state = FloatTensor(state[None,None,:,:])
            
            score += int(reward)

            reward = FloatTensor([float(reward)])
            memory.push(last_state, action, state, reward)

            if done:
                if i_episode % 10 == 0:
                    log = 'epoch {0} score {1}'.format(i_episode, score)
                    print(log)
                    f.write(log + '\n')
                    loss = optimize_model()
                    if loss:
                        print('loss: {:.0f}'.format(loss))
                if i_episode % 100 == 0:
                    is_best = True if score > best_score else False
                    save_checkpoint({
                        'epoch' : i_episode,
                        'state_dict' : model.state_dict(),
                        'best_score' : best_score,
                        'optimizer' : optimizer.state_dict(),
                        'memory' : memory
                        }, is_best)
                break

    f.close()
    print('Complete')
