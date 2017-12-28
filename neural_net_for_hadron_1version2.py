# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import pybullet as p
import math
import pybullet_data
import time as t

EPISODES = 1000000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) 

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
   
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf",0,0,0)
    p.setGravity(0,0,0)
    hadron = p.loadURDF("/home/arashrobo/Desktop/IRIS_2017/hadron_urdf/urdf/hadron_urdf.urdf")
    p.setRealTimeSimulation(1)


    
    #state_size = env.observation_space.shape[0]
    
    state_size = 26
    action_size = 32
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32
    last = -1
    
    for e in range(EPISODES):
        #state = env.reset()
        #reset
        p.resetSimulation()
        #p.loadURDF("plane.urdf")
        p.loadSDF("stadium.sdf")
        p.setGravity(0,0,-10)
        hadron = p.loadURDF("/home/arashrobo/Desktop/IRIS_2017/hadron_urdf/urdf/hadron_urdf.urdf")
        
        state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        state[0] = p.getBasePositionAndOrientation(hadron)[0][0]
        state[1] = p.getBasePositionAndOrientation(hadron)[0][1]
        state[2] = p.getBasePositionAndOrientation(hadron)[0][2]
        state[3] = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(hadron)[1])[0]
        state[4] = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(hadron)[1])[1]
        state[5] = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(hadron)[1])[2]

        for n in range (0,15):
            state[6+n] = p.getJointState(hadron,n)[0]
            print "state"
            print(state[n])
        state[25] = last      
        #reset
        state = np.reshape(state, [1, state_size])
        for time in range(1000):
            #env.render()
            action = agent.act(state)
            last = action
   
            if(action<16):
                print(action)
                p.setJointMotorControl2(hadron,action,p.VELOCITY_CONTROL,targetVelocity= 5.233333)
                t.sleep(0.03333)
                p.setJointMotorControl2(hadron,action,p.VELOCITY_CONTROL,targetVelocity= 0)

            else:
                print(action)
                p.setJointMotorControl2(hadron,(action-16),p.VELOCITY_CONTROL,targetVelocity= -5.233333)
                t.sleep(0.0333)
                p.setJointMotorControl2(hadron,(action-16),p.VELOCITY_CONTROL,targetVelocity= 0)
                
            p.stepSimulation()
            next_state = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            next_state[0] = p.getBasePositionAndOrientation(hadron)[0][0]
            next_state[1] = p.getBasePositionAndOrientation(hadron)[0][1]
            next_state[2] = p.getBasePositionAndOrientation(hadron)[0][2]
            next_state[3] = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(hadron)[1])[0]
            next_state[4] = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(hadron)[1])[1]
            next_state[5] = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(hadron)[1])[2]

            for n in range (0,15):
                next_state[6+n] = p.getJointState(hadron,n)[0]

            next_state[25] = last 
            
            x0 = p.getBasePositionAndOrientation(hadron)[0][0]
            y0 = p.getBasePositionAndOrientation(hadron)[0][1]
            x1 = 0
            y1 = 0
            reward = math.sqrt(((x1-x0)*(x1-x0))+((y1-y0)*(y1-y0)))
            print ("reward=")
            print (reward)

            c = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(hadron)[1])[0]
            b = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(hadron)[1])[1]
            #print(b)
            
            if ((c>0.785 or c<(-0.785) or b>0.785 or b<-0.785) and time>500):
                print("hi")
                reward = -100
            else:
                if(c>0.785 or c<(-0.785)):
                    reward = reward - math.sqrt(c*c)
                else:
                    reward = reward + math.sqrt(c*c)
                if(b>0.785 or b<(-0.785)):
                    reward = reward - math.sqrt(b*b)    
                else:
                    reward = reward + math.sqrt(c*c)
                    
                
                
            
            #
            #reward = reward if reward>-3 else -10000
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if reward<-1 and time> 500:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
                agent.save("sa.h5")# version 2
                
        if len(agent.memory) > batch_size:
            
            agent.replay(batch_size)

    
        
