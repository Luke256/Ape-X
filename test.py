from keras.models import load_model
import gym
import numpy as np
import tensorflow as tf
import time

if __name__=="__main__":
    #test
    model=load_model("CartPole.h5")
    env=gym.make("CartPole-v0")

    state=env.reset()


    while True:
        action=np.argmax(model.predict(np.array([[state]]))[0])
        state,reward,done,_=env.step(action)

        env.render()
        if done:
            state=env.reset()
    