#-*- coding:utf-8 -*-
import gym
import keras.backend as K
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Flatten, Input, Lambda, concatenate

from game import GameClass


class Leaner:
    def __init__(self,
        env, #トレーニング環境
        exp_queue, #Memory
        param_queue, #Actorへの提供用
        exp_memory_size, #Memoryの上限(MemoryもLeanerで管理する)
        train_mamory_size, #学習するときのバッチサイズ
        myenv=False, #クラスか名前か
        window_length=4 #考慮に入れるフレーム数
        ):
        #引数の変数を受け取る
        self.exp_queue=exp_queue
        self.param_queue=param_queue
        self.exp_memory_size=exp_memory_size
        self.train_mamory_size=train_mamory_size
        self.window_length=window_length
        if myenv:
            self.env=env()
        else:
            self.env=gym.make(env)
        self.num_actions=self.env.action_space.n

        #ネットワーク定義
        self.main_Q=self.build_network() #行動決定用のQネットワーク
        self.target_Q=self.build_network() #価値計算用のQネットワーク

        #最初の重みをQueueに入れておく
        while not param_queue.full():
            param_queue.put(self.main_Q.get_weights())
        # print(param_queue.full())
        return 

    def build_network(self):
        #ネットワーク構築
        l_input=Input((1,)+self.env.observation_space.shape)
        fltn=Flatten()(l_input)
        dense=Dense(units=256,activation="relu")(fltn)
        dense=Dense(units=256,activation="relu")(dense)
        dense=Dense(units=256,activation="relu")(dense)
        v=Dense(units=256,activation="relu")(dense)
        v=Dense(units=1)(v)
        adv=Dense(units=256,activation="relu")(dense)
        adv=Dense(units=self.num_actions)(adv)
        y=concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + (a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True))), output_shape=(self.num_actions,))(y)
        model=Model(inputs=l_input,outputs=l_output)

        model.compile(optimizer="Adam",
                        loss="mae",
                        metrics=["accuracy"])

        return model

    def run(self):
        print("run")
        return 0



if __name__=="__main__":
    l=Leaner(1,1,1,1,GameClass,myenv=True)
