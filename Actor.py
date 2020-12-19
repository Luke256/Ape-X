#-*- coding:utf-8 -*-
import gym
from game import GameClass

class Actor:
    def __init__(self,
        env_name, #Gymのenv
        exp_queue, #Memory用のmp.Queue
        param_queue, #Leaner用のmp.Queue
        epochs, #試行回数
        id_, #識別用ID
        num_actors, #Actorの総数
        gamma=0.9, #割引率
        myenv=False, #これをTrueにするとenv_nameにクラス名を渡せる
        epsilon=0.4,
        alpha=7
        ):

        if(myenv):
            self.env=env_name()
        else:
            self.env=gym.make(env_name)

        self.exp_queue=exp_queue
        self.param_queue=param_queue
        self.epochs=epochs
        self.id=id_
        self.gamma=gamma
        self.epsilon=epsilon**(1+id_/num_actors*alpha)




# if __name__=="__main__":
#     # a=Actor(GameClass,2,3,4,5,myenv=True)

