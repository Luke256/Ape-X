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
import random
import sys

class Actor:
    def __init__(self,
        env, #Gymのenv
        exp_queue, #Memory用のmp.Queue
        param_queue, #Leaner用のmp.Queue
        nb_steps, #学習回数
        num_actions, #行動の種類数
        id_, #識別用ID
        num_actors, #Actorの総数
        actor_working, #どのActorが動いていてどのActorが動いていないか
        myenv=False, #これをTrueにするとenvにクラス名を渡せる
        gamma=0.9, #割引率
        max_epsilon=0.4,
        alpha=7,
        update_param_interbal=100, #何回試行したらパラメータを更新するか
        visualize=False,
        window_length=3 #考慮に入れるフレーム数
        ):

        if(myenv):
            self.env=env()
        else:
            self.env=gym.make(env)

        #各種変数の定義、代入
        self.num_actions=num_actions
        self.exp_queue=exp_queue
        self.param_queue=param_queue
        self.id=id_
        self.gamma=gamma
        self.window_length=window_length
        self.update_param_interbal=update_param_interbal
        self.visualize=visualize
        self.nb_steps=nb_steps
        self.actor_working=actor_working

        #εの計算
        if num_actors <= 1:
            self.epsilon = max_epsilon ** (1 + alpha)
        else:
            self.epsilon = max_epsilon ** (1 + id_/(num_actors-1)*alpha)

        self.main_Q=self.build_network() #行動決定用ネットワーク
        self.target_Q=self.build_network() #価値計算用ネットワーク(TD誤差の計算のためだけに使う)
        
        #初期のLeanerの重みを取得
        while self.param_queue.empty():
            time.sleep(2)

        param=self.param_queue.get()
        self.main_Q.set_weights(param[0])
        self.target_Q.set_weights(param[1])

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
        v=Dense(units=1,activation="linear",name="dense_v2_{}".format(self.id))(v)
        adv=Dense(units=256,activation="relu",name="dense_adv1_{}".format(self.id))(dense)
        adv=Dense(units=self.num_actions,activation="linear",name="dense_adv2_{}".format(self.id))(adv)
        y=concatenate([v,adv])
        l_output = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + (a[:, 1:] - K.stop_gradient(K.mean(a[:,1:],keepdims=True))), output_shape=(self.num_actions,))(y)
        model=Model(inputs=l_input,outputs=l_output)

        return model


    def send_batch(self,batch):
        #この部分はApe-XのTD誤差の式を理解していないとわかりにくい
        #一応対応する変数の名前は書いておく

        length=len(batch)

        #累計報酬
        batch_rewards=0
        for i in range(length):
            batch_rewards+=(self.gamma**i)*batch[i][3]
        #State[t]
        batch_state=np.array([[batch[0][0]]])
        #State[t+n]
        batch_state_n=np.array([[batch[length-1][1]]])
        #Action[t]
        batch_action=batch[0][2]

        #TD誤差の計算
        action=np.argmax(self.main_Q.predict(batch_state_n)[0])
        q=(self.gamma**length)*self.target_Q.predict(batch_state_n)[0][action]
        td_error=batch_rewards+q+self.main_Q.predict(batch_state)[0][batch_action]

        #exp_queue に送る内容の作成
        send=[batch_rewards,batch_state,batch_state_n,batch_action,abs(td_error)]
        #送信
        self.exp_queue.put(send)
    
    def run(self):
        batch=[] #Memoryに渡す前に経験をある程度蓄積させるためのバッファ
        t=0 #トータル試行回数
        epoch=0

        self.actor_working[self.id]=True
        print("Actor{} starts".format(self.id))

        try:
            while True: #Leanerが生きている限り経験を送る
                state=self.env.reset()
                done=False

                epoch_reward=0
                step=0

                while not done and t<self.nb_steps:

                    #行動を決定
                    action=self.main_Q.predict(np.array([[state]]))
                    action=np.argmax(action[0])
                    if self.epsilon>=random.random() and (not self.visualize):
                        action=random.randrange(0,self.num_actions)

                    old_state=state
                    state,reward,done,info=self.env.step(action) #実際に行動

                    epoch_reward+=reward

                    if not self.visualize: #描画用でないのなら経験を送る
                        mini_batch=[old_state,state,action,reward] #ミニバッチを作成

                        batch.append([mini_batch]) #バッファに新しく格納
                        remove_list=[] #溜め切ってデータを送信し終えたデータ(後で削除する)
                        for i in range(len(batch)):
                            batch[i].append(mini_batch)
                            if(len(batch[i])>=self.window_length):
                                self.send_batch(batch[i]) #経験の送信

                                #不要なので後で捨てておく
                                remove_list.append(i)

                            elif done:# これ以上連続した新しい経験は発生しないのでバッファの中身を全部送る
                                self.send_batch(batch[i]) #経験の送信

                                #不要なので後で捨てておく
                                remove_list.append(i)


                        #前から削除するとバグりそうので後ろから削除
                        remove_list.sort()
                        remove_list.reverse()
                        for i in remove_list:
                            batch.pop(i)

                    else: #visualize=Trueなら描画だけ行う
                        self.env.render()
                        time.sleep(1/30)

                    if t%self.update_param_interbal==0: #Qネットワークの重みを更新
                        failed=0
                        while self.param_queue.empty():
                            if failed>20:
                                print("Actor{} ended.".format(self.id))
                                self.actor_working[self.id]=False
                                return

                            print("Actor{} failed to get new param.".format(self.id))
                            failed+=1
                            time.sleep(5)

                        #重みの取得
                        param=self.param_queue.get()
                        self.main_Q.set_weights(param[0])
                        self.target_Q.set_weights(param[1])

                    t+=1
                    step+=1


                    if done:
                        break

                if t>=self.nb_steps:
                    break

                epoch+=1
                print("Actor{} episode:{} nb_action:{}({}) reward:{} step:{}".format(self.id,epoch,t,self.nb_steps,epoch_reward,step))
        
        except KeyboardInterrupt:
            print("Actor{} ended".format(self.id))
            self.actor_working[self.id]=False
            return

        print("Actor{} ended.".format(self.id))
        self.actor_working[self.id]=False
        return