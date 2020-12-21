import multiprocessing as mp


from game import GameClass
from Actor import Actor
from Leaner import Leaner


exp_queue=mp.Queue(5000)
param_queue=mp.Queue(2)

leaner=Leaner(exp_queue,param_queue,10,10,GameClass,myenv=True)
actor=Actor(GameClass,exp_queue,param_queue,100,1,12,myenv=True)