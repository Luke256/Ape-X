from keras.models import load_model
import gym
import numpy as np
import tensorflow as tf

if __name__=="__main__":
    #test
    model=load_model("CartPole")
    env=gym.make("CartPole-v0")

    state=env.reset()

    while True:
        action=np.argmax(model.predict(state))
        state,reward,done,_=env.step(action)

        if done:
            state=env.reset()