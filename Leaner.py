#-*- coding:utf-8 -*-
import gym
import keras.backend as K
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Flatten, Input, Lambda, concatenate
from keras.models import load_model
from SumTree import SumTree
import random
import time
import numpy as np
import sys
from pathlib import Path

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
            # p_sample:そのデータの優先度（実際には使ってない）
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
        num_actions, #行動の種類数
        exp_memory_size, #Memoryの上限(MemoryもLeanerで管理する)
        train_batch_size, #学習するときのバッチサイズ
        actor_working, #Actorが動いているかどうか
        save_name, #保存名
        myenv=False, #クラスか名前か
        gamma=0.99, #割引率
        update_target_interbal=100, #価値計算用のネットワークの更新頻度
        load_model_path=None,
        window_length=3 #考慮に入れるフレーム数
        ):
        #引数の変数を受け取る
        self.exp_queue=exp_queue
        self.param_queue=param_queue
        self.exp_memory_size=exp_memory_size
        self.train_batch_size=train_batch_size
        self.window_length=window_length
        self.gamma=gamma
        self.save_name=save_name
        self.update_target_interbal=update_target_interbal
        self.actor_working=actor_working
        if myenv:
            self.env=env()
        else:
            self.env=gym.make(env)
        self.num_actions=num_actions

        #ネットワーク定義
        self.main_Q=self.build_network() #行動決定用のQネットワーク
        self.target_Q=self.build_network() #価値計算用のQネットワーク
        if load_model_path!=None:
            self.main_Q=load_model(load_model_path)
        self.target_Q.set_weights(self.main_Q.get_weights())

        #メモリ作成
        self.memory=Memory(self.exp_memory_size)

        #最初の重みをQueueに入れておく
        while not param_queue.full():
            param_queue.put([self.main_Q.get_weights(),self.target_Q.get_weights()])
        # print(param_queue.full())
        return 

    def build_network(self):
        #ネットワーク構築
        l_input=Input((1,)+self.env.observation_space.shape,name="Input_Leaner")
        fltn=Flatten()(l_input)
        dense=Dense(units=256,activation="relu")(fltn)
        dense=Dense(units=256,activation="relu")(dense)
        dense=Dense(units=256,activation="relu")(dense)
        v=Dense(units=256,activation="relu")(dense)
        v=Dense(units=1,activation="linear")(v)
        adv=Dense(units=256,activation="relu")(dense)
        adv=Dense(units=self.num_actions,activation="linear")(adv)
        y=concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + (a[:, 1:] - K.stop_gradient(K.mean(a[:,1:],keepdims=True))), output_shape=(self.num_actions,))(y)
        model=Model(inputs=l_input,outputs=l_output)

        model.compile(optimizer="Adam",
                        loss="mse",
                        metrics=["accuracy"])

        return model

    def make_batch(self):
        train_batch=self.memory.sample(self.train_batch_size)
        train_batch=np.array(train_batch)
        batch=train_batch[:,2] #データだけ取り出す

        #このままだと
        #[[a_1,b_1,c_1],[a_2,b_2,c_2]...[a_n,b_n,c_n]]
        #となっているから、これを
        #[[a_1,a_2,...,a_n],[b_1,b_2,...,b_n],[c_1,c_2,...,c_n]]
        #にする
        #(こうすると推論のときに一気にできるので高速に処理ができる)

        rewards_batch=[] #Reward[t]
        state_batch=[] #State[t]
        state_n_batch=[] #State[t+n]
        action_batch=[] #Action[n]

        for i in batch:
            rewards_batch.append(i[0])
            state_batch.append(i[1][0])
            state_n_batch.append(i[2][0])
            action_batch.append(i[3])

        rewards_batch=np.array(rewards_batch)
        state_batch=np.array(state_batch)
        state_n_batch=np.array(state_n_batch)
        action_batch=np.array(action_batch)


        #教師データ作成
        batch_size=len(state_batch)
        y=self.main_Q.predict(state_batch,batch_size=batch_size)

        #予め推論をしておく
        main_predict=self.main_Q.predict(state_n_batch,batch_size=batch_size)
        target_predict=self.target_Q.predict(state_n_batch,batch_size=batch_size)

        for i in range(batch_size):
            action=np.argmax(main_predict[i]) #mainQを使って行動選択
            q=target_predict[i][action] #targetQでQ値を出す

            target=rewards_batch[i]+(self.gamma**self.window_length)*q #Q(State(t),Action(t)) として出すべき値(目標の値)

            td_error=y[i][action_batch[i]]-target #TD誤差を計算
            self.memory.update_p(train_batch[i][0],abs(td_error)) #その繊維の優先度を更新

            y[i][action_batch[i]]=target #教師データに目標の値を代入

        return state_batch,y




    def run(self):
        t=0 #トータル試行回数

        while self.memory.length()<self.train_batch_size:
            print("Leaner is waiting for enough experience to train")
            while not self.exp_queue.empty(): #exp_queueにある経験情報を全てMemoryに追加
                batch=self.exp_queue.get()
                self.memory.add(batch[4],batch)
            time.sleep(5)
            
        print("Leaner starts")

        try:

            while True:
                #Actorが一つも動いていなければ終了
                working=False
                for i in self.actor_working:
                    if i:
                        working=True
                        break
                if not working:
                    break
                
                #exp_queueから経験を取得・Memoryに入れる
                while not self.exp_queue.empty():
                    batch=self.exp_queue.get()
                    self.memory.add(batch[4],batch)

                #サンプル作成
                X,y=self.make_batch()
                X=np.array(X)
                y=np.array(y)

                if t%self.update_target_interbal==0: #価値計算用ネットワークを更新
                    self.target_Q.set_weights(self.main_Q.get_weights())
                #行動決定用は学習・重みを更新
                self.main_Q.fit(X,y,epochs=1,verbose=0)

                #Queueが満杯になるまで入れる
                while not self.param_queue.full():
                    self.param_queue.put([self.main_Q.get_weights(),self.target_Q.get_weights()])



                t+=1

        except KeyboardInterrupt:
            self.main_Q.save(self.save_name)
            print("model has been saved with name'"+self.save_name+"'")

            print("Learning was stopped by user")
            return

        self.main_Q.save(self.save_name)
        print("model has been saved with name '"+self.save_name+"'")

        print("Leaner finished")
        return
