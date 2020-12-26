import multiprocessing as mp
from Actor import Actor
from Leaner import Leaner
import time
from SumTree import SumTree
from keras.models import load_model
import gym
import numpy as np
import ctypes
from game import GameClass

def A(exp_queue,param_queue,nb_steps,num_actions,id,num_actors,actor_working):
    if id==0:
        actor=Actor("CartPole-v0",exp_queue,param_queue,nb_steps,num_actions,id,num_actors,actor_working)
    else:
        actor=Actor("CartPole-v0",exp_queue,param_queue,nb_steps,num_actions,id,num_actors,actor_working)
    actor.run()
    return

def L(exp_queue,param_queue,num_actions,memory_size,train_batch_size,actor_working,save_name):
    leaner=Leaner("CartPole-v0",exp_queue,param_queue,num_actions,memory_size,train_batch_size,actor_working,save_name
    # ,load_model_path="CartPole.h5"
    )
    leaner.run()
    return
    

if __name__=="__main__":
    num_actors=5
    num_actions=2
    nb_steps=10000
    memory_size=100000
    train_batch_size=10
    save_name="CartPole.h5"


    exp_queue=mp.Queue(5000)
    param_queue=mp.Queue(int(num_actors))
    actor_working=mp.Value(ctypes.c_uint*num_actors)
    for i in range(num_actors):
        actor_working[i]=False


    ps=[]
    ps.append(mp.Process(target=L, args=(exp_queue,param_queue,num_actions,memory_size,train_batch_size,actor_working,save_name)))

    try:
        for i in range(num_actors):
            ps.append(mp.Process(target=A, args=(exp_queue,param_queue,nb_steps,num_actions,i,num_actors,actor_working)))
            ps[i].start()
            time.sleep(0.5)

        for p in ps:
            p.join()


    except KeyboardInterrupt:
        pass