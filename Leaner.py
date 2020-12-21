import keras.backend as K
import tensorflow as tf
from keras import Model
from keras.layers import Dense, Input, Lambda, concatenate
from game import GameClass
import gym


class Leaner:
    def __init__(self,
        exp_queue, #Memory
        param_queue, #Actorへの提供用
        exp_memory_size, #Memoryの上限(MemoryもLeanerで管理する)
        train_mamory_size, #学習するときのバッチサイズ
        env, #トレーニング環境
        myenv=False, #クラスか名前か
        window_length=4 #考慮に入れるフレーム数
        ):
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

        self.main_Q=self.build_network() #行動決定用のQネットワーク
        self.target_Q=self.build_network() #価値計算用のQネットワーク

        while not param_queue.full():
            param_queue.put([self.main_Q.get_weights(),self.target_Q.get_weights()])

    def build_network(self):
        #ネットワーク構築
        l_input=Input((self.window_length,)+self.env.observation_space.shape)
        dense=Dense(units=256,activation="relu",name="dense1_leaner")(l_input)
        dense=Dense(units=256,activation="relu",name="dense2_leaner")(dense)
        dense=Dense(units=256,activation="relu",name="dense3_leaner")(dense)
        v=Dense(units=256,activation="relu",name="dense_v1_leaner")(dense)
        v=Dense(units=1,name="dense_v2_leaner")(v)
        adv=Dense(units=256,activation="relu",name="dense_adv1_leaner")(dense)
        adv=Dense(units=self.num_actions,name="dense_adv2_leaner")(adv)
        y=concatenate([v,adv])
        l_output=Lambda(lambda a: a - tf.stop_gradient(K.mean(a[:,1:],keepdims=True)), output_shape=(self.num_actions,))(y)
        model=Model(inputs=l_input,outputs=l_output)
        model.compile(optimizer="Adam",
                        loss="mae",
                        metrics=["accuracy"])

        return model


if __name__=="__main__":
    l=Leaner(1,1,1,1,GameClass,myenv=True)