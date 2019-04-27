import sys
import os
import torch
import time
from game import Engine
from dqn_agent import DQN, ReplayMemory, Transition
from torch.autograd import Variable

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

width, height = 10, 20
engine = Engine(width, height)

def load_model(filename):
    model = DQN()
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    return model

def run(model):
    state = FloatTensor(engine.clear()[None,None,:,:])
    score = 0
    while True:
        action = model(Variable(state,
            volatile=True).type(FloatTensor)).data.max(1)[1].view(1,1).type(LongTensor)
        print( model(Variable(state,
            volatile=True).type(FloatTensor)).data)

        state, reward, done = engine.step(action[0,0])
        state = FloatTensor(state[None,None,:,:])

        score += int(reward)

        print(engine)
        print(action)
        time.sleep(.1)

        if done:
            print('score {0}'.format(score))
            break

if len(sys.argv) <= 1:
    print('specify a filename to load the model')
    sys.exit(1)

if __name__ == '__main__':
    filename = sys.argv[1]
    if os.path.isfile(filename):
        print("=> loading model '{}'".format(filename))
        model = load_model(filename).eval()
        run(model)
    else:
        print("=> no file found at '{}'".format(filename))
