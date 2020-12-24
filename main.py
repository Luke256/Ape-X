import multiprocessing as mp
from Actor import Actor
from Leaner import Leaner
import time
from SumTree import SumTree
from keras.models import load_model
import gym
import numpy as np
import ctypes

def A(exp_queue,param_queue,id,num_actors,leaner_working):
    if id==0:
        actor=Actor("CartPole-v0",exp_queue,param_queue,id,num_actors,leaner_working,update_param_interbal=20,visualize=True)
    else:
        actor=Actor("CartPole-v0",exp_queue,param_queue,id,num_actors,leaner_working,update_param_interbal=20)
    actor.run()
    return

def L(exp_queue,param_queue,epochs,memory_size,train_batch_size,leaner_working):
    leaner=Leaner("CartPole-v0",exp_queue,param_queue,epochs,memory_size,train_batch_size,leaner_working,"CartPole",update_target_interbal=5)
    leaner.run()
    return
    

if __name__=="__main__":
    num_actors=5
    epochs=10000
    memory_size=100000
    train_batch_size=10


    exp_queue=mp.Queue(5000)
    param_queue=mp.Queue(int(num_actors*1.2))
    leaner_working=mp.Value(ctypes.c_uint*1)
    leaner_working[0]=True

    ps=[]
    ps.append(mp.Process(target=L, args=(exp_queue,param_queue,epochs,memory_size,train_batch_size,leaner_working)))

    for i in range(num_actors):
        ps.append(mp.Process(target=A, args=(exp_queue,param_queue,i,num_actors,leaner_working)))

    try:
        for p in ps:
            p.start()
            time.sleep(1)

        for p in ps:
            p.join()


    except KeyboardInterrupt:
        pass