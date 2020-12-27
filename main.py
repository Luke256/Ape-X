import multiprocessing as mp
from Actor import Actor
from Leaner import Leaner
import time
from keras.models import load_model
import ctypes

def A(exp_queue,param_queue,nb_steps,warmup_steps,num_actions,id,num_actors,actor_working):
    if id==0:
        actor=Actor("CartPole-v0",exp_queue,param_queue,nb_steps,warmup_steps,num_actions,id,num_actors,actor_working,max_epsilon=0.7)
    else:
        actor=Actor("CartPole-v0",exp_queue,param_queue,nb_steps,warmup_steps,num_actions,id,num_actors,actor_working,max_epsilon=0.7)
    actor.run()
    return

def L(exp_queue,param_queue,num_actions,memory_size,train_batch_size,actor_working,save_name):
    leaner=Leaner("CartPole-v0",exp_queue,param_queue,num_actions,memory_size,train_batch_size,actor_working,save_name)
    leaner.run()
    return
    

if __name__=="__main__":
    num_actors=3 #Actorの数
    num_actions=2 #行動の種類数
    nb_steps=10000 #試行回数
    warmup_steps=200 #ランダムに行動する回数
    memory_size=100000 #Memoryの上限
    train_batch_size=32 #Leanerが一回に使用するデータの数
    save_name="CartPole.h5" #ファイルをセーブする時の名前

    #プロセス間通信用
    exp_queue=mp.Queue(5000)
    param_queue=mp.Queue(int(num_actors))
    actor_working=mp.Value(ctypes.c_uint*num_actors)
    for i in range(num_actors):
        actor_working[i]=False


    #並列処理(Leaner)
    ps=[]
    ps.append(mp.Process(target=L, args=(exp_queue,param_queue,num_actions,memory_size,train_batch_size,actor_working,save_name)))
    ps[0].start()

    try:

        #並列処理(Actor)
        for i in range(num_actors):

            ps.append(mp.Process(target=A, args=(exp_queue,param_queue,nb_steps,warmup_steps,num_actions,i,num_actors,actor_working)))
            ps[i+1].start()
            time.sleep(1)

        for p in ps:
            print(p)
            p.join()


    except KeyboardInterrupt:
        print("end")