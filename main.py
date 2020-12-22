import multiprocessing as mp


from game import GameClass
from Actor import Actor
from Leaner import Leaner
import time
from SumTree import SumTree

def A(exp_queue,param_queue,epochs,id,num_actors):
    actor=Actor(GameClass,exp_queue,param_queue,epochs,id,num_actors,myenv=True)
    actor.run()
    print("actor_finish")
    return

def L(exp_queue,param_queue):
    leaner=Leaner(GameClass,exp_queue,param_queue,10,10,myenv=True)
    leaner.run()
    print("leaner_fnish")
    return
    

if __name__=="__main__":
    num_actors=1


    exp_queue=mp.Queue(5000)
    param_queue=mp.Queue(num_actors)

    ps=[]
    ps.append(mp.Process(target=L, args=(exp_queue,param_queue)))

    for i in range(num_actors):
        ps.append(mp.Process(target=A, args=(exp_queue,param_queue,100,i,num_actors)))

    for p in ps:
        p.start()
        time.sleep(1)

    for p in ps:
        print(p)
        p.join()


    print("program_end")