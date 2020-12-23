import multiprocessing as mp
from Actor import Actor
from Leaner import Leaner
import time
from SumTree import SumTree
from keras.models import load_model
import gym
import numpy as np

def A(exp_queue,param_queue,epochs,id,num_actors):
    actor=Actor("CartPole-v0",exp_queue,param_queue,epochs,id,num_actors,update_param_interbal=10)
    actor.run()
    return

def L(exp_queue,param_queue,epochs,memory_size,train_batch_size):
    leaner=Leaner("CartPole-v0",exp_queue,param_queue,epochs,memory_size,train_batch_size,"CartPole")
    leaner.run()
    return
    

if __name__=="__main__":
    num_actors=5
    epochs=10000
    memory_size=100000


    exp_queue=mp.Queue(5000)
    param_queue=mp.Queue(num_actors+2)

    ps=[]
    ps.append(mp.Process(target=L, args=(exp_queue,param_queue,epochs,memory_size,10)))

    for i in range(num_actors):
        ps.append(mp.Process(target=A, args=(exp_queue,param_queue,epochs,i,num_actors)))

    for p in ps:
        p.start()
        time.sleep(1)

    for p in ps:
        print(p)
        p.join()