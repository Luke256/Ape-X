#-*- coding:utf-8 -*-
import gym
from game import GameClass
from keras.layers import Dense,Input,concatenate,Lambda,Flatten
from keras import backend as K
from keras import Model
import tensorflow as tf
import time
import numpy as np
from queue import Queue

class Actor:
    def __init__(self,
        env, #Gymのenv
        exp_queue, #Memory用のmp.Queue
        param_queue, #Leaner用のmp.Queue
        epochs, #試行回数
        id_, #識別用ID
        num_actors, #Actorの総数
        gamma=0.9, #割引率
        myenv=False, #これをTrueにするとenvにクラス名を渡せる
        epsilon=0.4,
        alpha=7,
        buffer_size=100, #どのくらい経験がたまったらMemoryに送るか
        update_param_interbal=100, #何回試行したらパラメータを更新するか
        window_length=3 #考慮に入れるフレーム数
        ):

        if(myenv):
            self.env=env()
        else:
            self.env=gym.make(env)

        #各種変数の定義、代入
        self.num_actions=self.env.action_space.n
        self.exp_queue=exp_queue
        self.param_queue=param_queue
        self.epochs=epochs
        self.id=id_
        self.gamma=gamma
        self.epsilon=epsilon**(1+id_/num_actors*alpha)
        self.window_length=window_length
        self.buffer_size=buffer_size
        self.update_param_interbal=update_param_interbal

        self.main_Q=self.build_network() #行動決定用ネットワーク(Actorは行動のみなので価値計算をするQネットワークは不要)
        
        #初期のLeanerの重みを取得
        while self.param_queue.empty():
            time.sleep(2)
        
        # print(self.param_queue.empty())

        self.main_Q.set_weights(self.param_queue.get())
        # print("actor")

    def build_network(self):
        #ネットワーク構築
        # ここのネットワークはLeanerのものと一致させないといけない
        # (コピペが良さそう)
        l_input=Input((1,)+self.env.observation_space.shape)
        fltn=Flatten()(l_input)
        dense=Dense(units=256,activation="relu",name="dense1_{}".format(self.id))(fltn)
        dense=Dense(units=256,activation="relu",name="dense2_{}".format(self.id))(dense)
        dense=Dense(units=256,activation="relu",name="dense3_{}".format(self.id))(dense)
        v=Dense(units=256,activation="relu",name="dense_v1_{}".format(self.id))(dense)
        v=Dense(units=1,name="dense_v2_{}".format(self.id))(v)
        adv=Dense(units=256,activation="relu",name="dense_adv1_{}".format(self.id))(dense)
        adv=Dense(units=self.num_actions,name="dense_adv2_{}".format(self.id))(adv)
        y=concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + (a[:, 1:] - tf.stop_gradient(K.mean(a[:,1:],keepdims=True))), output_shape=(self.num_actions,))(y)
        model=Model(inputs=l_input,outputs=l_output)

        return model

    
    def run(self):
        batch=[] #Memoryに渡す前に経験をある程度蓄積させるためのバッファ
        t=0 #トータル試行回数

        for epoch in range(self.epochs): #トレーニングループ
            state=self.env.reset()
            done=False

            while not done:
                action=self.main_Q.predict(np.array([state])) #行動を決定
                # print(action)
                action=np.argmax(action)

                old_state=state
                state,reward,done,info=self.env.step(action) #実際に行動
                mini_batch=[old_state,state,action,reward] #ミニバッチを作成

                batch.append([mini_batch]) #バッファに新しく格納
                remove_list=[] #溜め切ってデータを送信し終えたデータ(後で削除する)
                for i in range(len(batch)):
                    batch[i].append(mini_batch)
                    if(len(batch[i])>=self.window_length): 
                        #Memoryに送る前処理
                        # この部分はApe-Xで使われるTD誤差の計算式を理解していないと
                        # 読み解くのは少し難しそう
                        # 1.割引率を考慮しながら報酬を足していく
                        batch_rewards=0
                        for j in range(self.window_length-1):
                            batch_rewards+=(self.gamma**j)*batch[i][j+1][3]
                        # S[t]
                        batch_state=batch[i][0][0]
                        #S[t+n]
                        batch_state_n=batch[i][self.window_length-1][0]
                        #A[t]
                        batch_action=batch[i][0][2]

                        # Memoryに送る内容の作成
                        send=[batch_rewards,batch_state,batch_state_n,batch_action]
                        # 送信
                        self.exp_queue.put(send)
                        #不要なので後で捨てておく
                        remove_list.append(i)

                    elif done:# これ以上連続した新しい経験は発生しないのでバッファの中身を全部送る
                        #処理は上のやつとほとんど一緒
                        #(self.window_lengthがlen(batch[i])-1になったくらい)

                        #Memoryに送る前処理
                        batch_rewards=0
                        for j in range(len(batch[i])-1):
                            batch_rewards+=(self.gamma**j)*batch[i][j+1][3]
                        batch_state=batch[i][0][0]
                        batch_state_n=batch[i][len(batch[i])-1][0]
                        batch_action=batch[i][0][2]

                        # Memoryに送る内容の作成
                        send=[batch_rewards,batch_state,batch_state_n,batch_action]
                        # 送信
                        self.exp_queue.put(send)
                        #不要なので後で捨てておく
                        remove_list.append(i)

                #前から削除するとバグりそうので後ろから削除
                remove_list.sort()
                remove_list.reverse()
                for i in remove_list:
                    batch.pop(i)
                
                if t%self.update_param_interbal==0: #Qネットワークの重みを更新
                    while self.param_queue.empty():
                        time.sleep(2)
                    self.main_Q.set_weights(self.param_queue.get())

                t+=1
                
                if done:
                    break

        print("Actor{} finished.".format(self.id))
        return



if __name__=="__main__":
    a=Actor(GameClass,1,1,1,1,1,myenv=True)