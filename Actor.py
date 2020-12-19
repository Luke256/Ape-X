#-*- coding:utf-8 -*-
import gym
from game import GameClass
from keras.layers import Dense,Input,concatenate,Lambda
from keras import backend as K
from keras import Model
import tensorflow as tf
import time

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
        alpha=7,
        window_length=1 #考慮に入れるフレーム数
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
        self.window_length=window_length

        self.q_model=self.build_network() #価値計算用ネットワーク
        
        while self.param_queue.empty():
            time.sleep(2)
        
        self.q_model.weights=self.param_queue.get()

    def build_network(self):
        #ネットワーク構築
        l_input=Input((self.window_length,)+self.env.observation_space.shape)
        dense=Dense(units=256,activation="relu",name="dense_1_{}".format(self.id))(l_input)
        dense=Dense(units=256,activation="relu",name="dense_2_{}".format(self.id))(dense)
        dense=Dense(units=256,activation="relu",name="dense_3_{}".format(self.id))(dense)
        v=Dense(units=256,activation="relu",name="dense_v_1_{}".format(self.id))(dense)
        v=Dense(units=256,activation="relu",name="dense_v_2_{}".format(self.id))(v)
        adv=Dense(units=256,activation="relu",name="dense_adv_1_{}".format(self.id))(dense)
        adv=Dense(units=256,activation="relu",name="dense_adv_2_{}".format(self.id))(adv)
        y=concatenate([v,adv])
        l_output=Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.compat.v1.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(self.num_actions,))(y)
        model=Model(input=l_input,output=l_output)

        return model

    


        

if __name__=="__main__":
    a=Actor(GameClass,1,1,1,1,1,myenv=True)