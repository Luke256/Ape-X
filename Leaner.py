#-*- coding:utf-8 -*-
import gym
import keras.backend as K
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Flatten, Input, Lambda, concatenate
from SumTree import SumTree
import random
import time
import numpy as np

from game import GameClass

class Memory:#経験を優先順位をつけて保存しておく
    # このMenmoryについての説明は
    # https://qiita.com/omuram/items/994ffe8d6deec509ac11#%E3%82%B5%E3%83%B3%E3%83%97%E3%83%AA%E3%83%B3%E3%82%B0
    # がわかりやすい

    def __init__(self,
        capacity):
        self.capacity=capacity

        self.data=SumTree(self.capacity)

    def length(self):
        return self.data.write

    def sample(self,num_samples):
        data_length=self.data.total()
        sampling_interbal=data_length/num_samples

        batch=[] #返すやつ
        for i in range(num_samples):
            l=sampling_interbal*i
            r=sampling_interbal*(i+1)
            p=random.uniform(l,r)

            (idx,p_sample,data)=self.data.get(p)
            # idx:使っている二分木上でのインデックス(優先度を更新するときに使う)
            # p_sample:そのデータの優先度
            # data:データ
            batch.append([idx,p_sample,data])

        return batch

    def add(self,p,data): #p:優先順位(TD誤差)
        self.data.add(p,data)

    def update_p(self,idx,p): #優先順位の更新
        self.data.update(idx,p)


class Leaner:
    def __init__(self,
        env, #トレーニング環境
        exp_queue, #Memory
        param_queue, #Actorへの提供用
        epochs, #試行回数
        exp_memory_size, #Memoryの上限(MemoryもLeanerで管理する)
        train_batch_size, #学習するときのバッチサイズ
        save_name, #保存名
        myenv=False, #クラスか名前か
        gamma=0.9, #割引率
        window_length=4 #考慮に入れるフレーム数
        ):
        #引数の変数を受け取る
        self.exp_queue=exp_queue
        self.param_queue=param_queue
        self.exp_memory_size=exp_memory_size
        self.train_batch_size=train_batch_size
        self.window_length=window_length
        self.epochs=epochs
        self.gamma=gamma
        self.save_name=save_name
        if myenv:
            self.env=env()
        else:
            self.env=gym.make(env)
        self.num_actions=self.env.action_space.n

        #ネットワーク定義
        self.main_Q=self.build_network() #行動決定用のQネットワーク
        self.target_Q=self.build_network() #価値計算用のQネットワーク

        #メモリ作成
        self.memory=Memory(self.exp_memory_size)

        #最初の重みをQueueに入れておく
        while not param_queue.full():
            param_queue.put([self.main_Q.get_weights(),self.target_Q.get_weights()])
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
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + (a[:, 1:] - K.stop_gradient(K.mean(a[:,1:],keepdims=True))), output_shape=(self.num_actions,))(y)
        model=Model(inputs=l_input,outputs=l_output)

        model.compile(optimizer="Adam",
                        loss="mae",
                        metrics=["accuracy"])

        return model

    def run(self):
        t=0 #トータル試行回数

        while self.memory.length()<self.train_batch_size:
            print("Leaner is waiting for enough experience to train")
            while not self.exp_queue.empty():
                batch=self.exp_queue.get()
                self.memory.add(batch[4],batch)
            time.sleep(5)
            
        print("Leaner starts")

        for epoch in range(self.epochs):
            train_batch=self.memory.sample(self.train_batch_size) #学習データの取得

            X=[] #学習データ
            y=[] #教師データ

            try:
                for batch_ in train_batch:
                    batch=batch_[2]
                    X.append(batch[1]) #状況

                    #教師データ作成
                    target=self.main_Q.predict(batch[2])[0] #関係ないところ(実際にしたactionでないもの)はmain_Qの予測で初期化

                    action=np.argmax(target)
                    a=(self.gamma**self.window_length)*self.target_Q.predict(batch[2])[0][action]
                    target[action]=batch[0]+a

                    y.append(target)
            except TypeError:
                print("Leaner failed to lean({}/{})".format(epoch,self.epochs))
                print(self.memory.data.data)
                continue

            X=np.array(X)
            y=np.array(y)

            #価値計算用ネットワークを更新
            self.target_Q.set_weights(self.main_Q.get_weights())
            #行動決定用は学習・重みを更新
            self.main_Q.fit(X,y,epochs=1,verbose=0)

            #TD誤差の計算・更新

            for i in range(len(train_batch)):
                q=self.main_Q.predict(train_batch[i][2][1])[0][train_batch[i][2][3]]
                a=(self.gamma**self.window_length)*self.target_Q.predict(train_batch[i][2][2])[0][np.argmax(self.main_Q.predict(train_batch[i][2][2])[0])]
                td_error=train_batch[i][2][0]+a-q
                train_batch[i][2][4]=td_error
                train_batch[i][1]=td_error
                self.memory.update_p(train_batch[i][0],td_error)

            #重みの共有
            while not self.param_queue.full():
                self.param_queue.put([self.main_Q.get_weights(),self.target_Q.get_weights()])

            #exp_queueから経験を取得・Memoryに入れる
            while not self.exp_queue.empty():
                batch=self.exp_queue.get()
                self.memory.add(batch[4],batch)

            print("Leaner finished leanring({}/{})".format(epoch,self.epochs))

        self.main_Q.save(self.save_name)

        print("Leaner finished")
        return
