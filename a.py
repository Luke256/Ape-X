import gym

import pickle
import os
import numpy as np
import random
import time
import traceback

import tensorflow as tf

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import *
from keras import backend as K

import rl.core

import multiprocessing as mp

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class PendulumProcessorForDQN(rl.core.Processor):
    def __init__(self, enable_image=False, image_size=84):
        self.image_size = image_size
        self.enable_image = enable_image
        self.mode = "train"
    
    def process_observation(self, observation):
        if not self.enable_image:
            return observation
        return self._get_rgb_state(observation)  # reshazeせずに返す
        
    def process_action(self, action):
        ACT_ID_TO_VALUE = {
            0: [-2.0], 
            1: [-1.0], 
            2: [0.0], 
            3: [+1.0],
            4: [+2.0],
        }
        return ACT_ID_TO_VALUE[action]

    def process_reward(self, reward):
        if self.mode == "test":  # testは本当の値を返す
            return reward
        # return np.clip(reward, -1., 1.)

        # -16.5～0 を -1～1 に正規化
        self.max = 0
        self.min = -16.5
        # min max normarization
        if (self.max - self.min) == 0:
            return 0
        M = 1
        m = -0.5
        return ((reward - self.min) / (self.max - self.min))*(M - m) + m
        

    # 状態（x,y座標）から対応画像を描画する関数
    def _get_rgb_state(self, state):
        img_size = self.image_size

        h_size = img_size/1.1

        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        dr = ImageDraw.Draw(img)

        # 棒の長さ
        l = img_size/2.0 * 3.0/ 2.0

        # 棒のラインの描写
        dr.line(((h_size - l * state[1], h_size - l * state[0]), (h_size, h_size)), (0, 0, 0), 1)

        # 棒の中心の円を描写（それっぽくしてみた）
        buff = img_size/32.0
        dr.ellipse(((h_size - buff, h_size - buff), (h_size + buff, h_size + buff)), 
                   outline=(0, 0, 0), fill=(255, 0, 0))

        # 画像の一次元化（GrayScale化）とarrayへの変換
        pilImg = img.convert("L")
        img_arr = np.asarray(pilImg)

        # 画像の規格化
        img_arr = img_arr/255.0

        return img_arr



#---------------------------------------------------
# manager
#---------------------------------------------------

class ApeXManager():
    def __init__(self, 
        actor_func, 
        num_actors,
        args,
        create_processor_func,
        create_optimizer_func,
        ):

        # 引数整形
        args["save_weights_path"] = args["save_weights_path"] if ("save_weights_path" in args) else ""
        args["load_weights_path"] = args["load_weights_path"] if ("load_weights_path" in args) else ""
        
        self.actor_func = actor_func
        self.num_actors = num_actors
        self.args = args

        # build_compile_model 関数用の引数
        model_args = {
            "input_shape": self.args["input_shape"],
            "enable_image_layer": self.args["enable_image_layer"],
            "nb_actions": self.args["nb_actions"],
            "window_length": self.args["window_length"],
            "enable_dueling_network": self.args["enable_dueling_network"],
            "dense_units_num": self.args["dense_units_num"],
            "metrics": self.args["metrics"],
            "create_optimizer_func": create_optimizer_func,
        }
        self._create_process(model_args, create_processor_func)


    def _create_process(self, model_args, create_processor_func):

        # 各Queueを作成
        experience_q = mp.Queue()
        model_sync_q = [[mp.Queue(), mp.Queue(), mp.Queue()] for _ in range(self.num_actors)]
        self.learner_end_q = [mp.Queue(), mp.Queue()]
        self.actors_end_q = [mp.Queue() for _ in range(self.num_actors)]
        self.learner_logger_q = mp.Queue()
        self.actors_logger_q = mp.Queue()

        # learner ps を作成
        args = (
            model_args,
            self.args,
            experience_q,
            model_sync_q,
            self.learner_end_q,
            self.learner_logger_q,
        )
        self.learner_ps = mp.Process(target=learner_run, args=args)
        
        # actor ps を作成
        self.actors_ps = []
        epsilon = self.args["epsilon"]
        epsilon_alpha = self.args["epsilon_alpha"]
        for i in range(self.num_actors):
            if self.num_actors <= 1:
                epsilon_i = epsilon ** (1 + epsilon_alpha)
            else:
                epsilon_i = epsilon ** (1 + i/(self.num_actors-1)*epsilon_alpha)
            print("Actor{} Epsilon:{}".format(i,epsilon_i))
            self.args["epsilon"] = epsilon_i

            args = (
                i,
                self.actor_func,
                model_args,
                self.args,
                create_processor_func,
                experience_q,
                model_sync_q[i],
                self.actors_logger_q,
                self.actors_end_q[i],
            )
            self.actors_ps.append(mp.Process(target=actor_run, args=args))

        # test用 Actor は子 Process では作らないのでselfにする。
        self.model_args = model_args
        self.create_processor_func = create_processor_func
    
    def __del__(self):
        self.learner_ps.terminate()
        for p in self.actors_ps:
            p.terminate()

    def train(self):

        learner_logs = []
        actors_logs = {}
        
        # プロセスを動かす
        try:
            self.learner_ps.start()
            for p in self.actors_ps:
                p.start()

            # 終了を待つ
            while True:
                time.sleep(1)  # polling time

                # 定期的にログを吸出し
                while not self.learner_logger_q.empty():
                    learner_logs.append(self.learner_logger_q.get(timeout=1))
                while not self.actors_logger_q.empty():
                    log = self.actors_logger_q.get(timeout=1)
                    if log["name"] not in actors_logs:
                        actors_logs[log["name"]] = []
                    actors_logs[log["name"]].append(log)
                
                # 終了判定
                f = True
                for q in self.actors_end_q:
                    if q.empty():
                        f = False
                        break
                if f:
                    break
        except KeyboardInterrupt:
            pass
        except Exception:
            print(traceback.format_exc())
        
        # 定期的にログを吸出し
        while not self.learner_logger_q.empty():
            learner_logs.append(self.learner_logger_q.get(timeout=1))
        while not self.actors_logger_q.empty():
            log = self.actors_logger_q.get(timeout=1)
            if log["name"] not in actors_logs:
                actors_logs[log["name"]] = []
            actors_logs[log["name"]].append(log)
    
        # learner に終了を投げる
        self.learner_end_q[0].put(1)
        
        # learner から最後の状態を取得
        print("Last Learner weights waiting...")
        weights = self.learner_end_q[1].get(timeout=60)
        
        # test用の Actor を作成
        test_actor = Actor(
            -1,
            self.model_args,
            self.args,
            None,
            None,
            processor=self.create_processor_func()
        )
        test_actor.model.set_weights(weights)

        # kill
        self.learner_ps.terminate()
        for p in self.actors_ps:
            p.terminate()

        return test_actor, learner_logs, actors_logs


#---------------------------------------------------
# network
#---------------------------------------------------
def clipped_error_loss(y_true, y_pred):
    err = y_true - y_pred  # エラー
    L2 = 0.5 * K.square(err)
    L1 = K.abs(err) - 0.5

    # エラーが[-1,1]区間ならL2、それ以外ならL1を選択する。
    loss = tf.where((K.abs(err) < 1.0), L2, L1)   # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

def build_compile_model(
    input_shape,         # 入力shape
    enable_image_layer,  # image_layerを入れるか
    window_length,       # window_length
    nb_actions,          # アクション数
    enable_dueling_network,  # dueling_network を有効にするか
    dense_units_num,         # Dense層のユニット数
    create_optimizer_func,
    metrics,                 # compile に渡す metrics
    ):

    c = input_ = Input(shape=(window_length,) + input_shape)

    if enable_image_layer:
        c = Permute((2, 3, 1))(c)  # (window,w,h) -> (w,h,window)

        c = Conv2D(32, (8, 8), strides=(4, 4), padding="same")(c)
        c = Activation("relu")(c)
        c = Conv2D(64, (4, 4), strides=(2, 2), padding="same")(c)
        c = Activation("relu")(c)
        c = Conv2D(64, (3, 3), strides=(1, 1), padding="same")(c)
        c = Activation("relu")(c)
    c = Flatten()(c)

    if enable_dueling_network:
        # value
        v = Dense(dense_units_num, activation="relu")(c)
        v = Dense(1)(v)

        # advance
        adv = Dense(dense_units_num, activation='relu')(c)
        adv = Dense(nb_actions)(adv)

        # 連結で結合
        c = Concatenate()([v,adv])
        c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_actions,))(c)
    else:
        c = Dense(dense_units_num, activation="relu")(c)
        c = Dense(nb_actions, activation="linear")(c)
    
    model = Model(input_, c)

    # compile
    model.compile(
        loss=clipped_error_loss, 
        optimizer=create_optimizer_func(), 
        metrics=metrics)
    
    return model


#---------------------------------------------------
# learner
#---------------------------------------------------

def learner_run(
    model_args, 
    args, 
    experience_q,
    model_sync_q,
    learner_end_q,
    logger_q,
    ):
    learner = Learner(
        model_args=model_args, 
        args=args, 
        experience_q=experience_q,
        model_sync_q=model_sync_q,
    )
    try:
        # model load
        if os.path.isfile(args["load_weights_path"]):
            learner.model.load_weights(args["load_weights_path"])
            learner.target_model.load_weights(args["load_weights_path"])

        # logger用
        t0 = time.time()

        # learner はひたすら学習する
        print("Learner Starts!")
        while True:
            learner.train()

            # logger
            if time.time() - t0 > args["logger_interval"]:
                t0 = time.time()
                logger_q.put({
                    "name": "learner",
                    "train_num": learner.train_num,
                })

            # 終了判定
            if not learner_end_q[0].empty():
                break
    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
    finally:
        print("Learning End. Train Count:{}".format(learner.train_num))

        # model save
        if args["save_weights_path"] != "":
            print("save:" + args["save_weights_path"])
            learner.model.save_weights(args["save_weights_path"], args["save_overwrite"])

        # 最後の状態を manager に投げる
        print("Last Learner weights sending...")
        learner_end_q[1].put(learner.model.get_weights())


class Learner():
    def __init__(self,
        model_args, 
        args, 
        experience_q,
        model_sync_q
        ):
        self.experience_q = experience_q
        self.model_sync_q = model_sync_q

        self.memory_warmup_size = args["remote_memory_warmup_size"]

        per_alpha = args["per_alpha"]
        per_beta_initial = args["per_beta_initial"]
        per_beta_steps = args["per_beta_steps"]
        per_enable_is = args["per_enable_is"]
        memory_capacity = args["remote_memory_capacity"]
        memory_type = args["remote_memory_type"]
        if memory_type == "replay":
            self.memory = ReplayMemory(memory_capacity)
        elif memory_type == "per_greedy":
            self.memory = PERGreedyMemory(memory_capacity)
        elif memory_type == "per_proportional":
            self.memory = PERProportionalMemory(memory_capacity, per_alpha, per_beta_initial, per_beta_steps, per_enable_is)
        elif memory_type == "per_rankbase":
            self.memory = PERRankBaseMemory(memory_capacity, per_alpha, per_beta_initial, per_beta_steps, per_enable_is)
        else:
            raise ValueError('memory_type is ["replay","per_proportional","per_rankbase"]')

        self.gamma = args["gamma"]
        self.batch_size = args["batch_size"]
        self.enable_double_dqn = args["enable_double_dqn"]
        self.target_model_update = args["target_model_update"]
        self.multireward_steps = args["multireward_steps"]

        assert memory_capacity > self.batch_size, "Memory capacity is small.(Larger than batch size)"
        assert self.memory_warmup_size > self.batch_size, "Warmup steps is few.(Larger than batch size)"

        # local
        self.train_num = 0

        # model create
        self.model = build_compile_model(**model_args)
        self.target_model = build_compile_model(**model_args)


    def train(self):
        
        # Actor から要求があれば weights を渡す
        for q in self.model_sync_q:
            if not q[0].empty():
                # 空にする(念のため)
                while not q[0].empty():
                    q[0].get(timeout=1)
                
                # 送る
                q[1].put(self.model.get_weights())
        
        # experience があれば RemoteMemory に追加
        while not self.experience_q.empty():
            exps = self.experience_q.get(timeout=1)
            for exp in exps:
                self.memory.add(exp, exp[4])

        # RemoteMemory が一定数貯まるまで学習しない。
        if len(self.memory) <= self.memory_warmup_size:
            time.sleep(1)  # なんとなく
            return
        
        (indexes, batchs, weights) = self.memory.sample(self.batch_size, self.train_num)
        state0_batch = []
        action_batch = []
        reward_batch = []
        state1_batch = []
        for batch in batchs:
            state0_batch.append(batch[0])
            action_batch.append(batch[1])
            reward_batch.append(batch[2])
            state1_batch.append(batch[3])

        # 更新用に現在のQネットワークを出力(Q network)
        outputs = self.model.predict(np.asarray(state0_batch), self.batch_size)

        if self.enable_double_dqn:
            # TargetNetworkとQNetworkのQ値を出す
            state1_model_qvals_batch = self.model.predict(np.asarray(state1_batch), self.batch_size)
            state1_target_qvals_batch = self.target_model.predict(np.asarray(state1_batch), self.batch_size)

            for i in range(self.batch_size):
                action = np.argmax(state1_model_qvals_batch[i])  # modelからアクションを出す
                maxq = state1_target_qvals_batch[i][action]  # Q値はtarget_modelを使って出す

                td_error = reward_batch[i] + (self.gamma ** self.multireward_steps) * maxq
                td_error *= weights[i]
                td_error_diff = outputs[i][action_batch[i]] - td_error  # TD誤差を取得
                outputs[i][action_batch[i]] = td_error   # 更新

                # TD誤差を更新
                self.memory.update(indexes[i], batchs[i], td_error_diff)

        else:
            # 次の状態のQ値を取得(target_network)
            target_qvals = self.target_model.predict(np.asarray(state1_batch), self.batch_size)

            # Q学習、Q(St,At)=Q(St,At)+α(r+γmax(St+1,At+1)-Q(St,At))
            for i in range(self.batch_size):
                maxq = np.max(target_qvals[i])
                td_error = reward_batch[i] + (self.gamma ** self.multireward_steps) * maxq
                td_error *= weights[i]
                td_error_diff = outputs[i][action_batch[i]] - td_error  # TD誤差を取得
                outputs[i][action_batch[i]] = td_error

                self.memory.update(batchs[i], td_error_diff)

        # 学習
        #self.model.train_on_batch(np.asarray(state0_batch), np.asarray(outputs))
        self.model.fit(np.asarray(state0_batch), np.asarray(outputs), batch_size=self.batch_size, epochs=1, verbose=0)
        self.train_num += 1

        # target networkの更新
        if self.train_num % self.target_model_update == 0:
            self.target_model.set_weights(self.model.get_weights())



#---------------------------------------------------
# actor
#---------------------------------------------------

class ActorLogger(rl.callbacks.Callback):
    def __init__(self, index, logger_q, interval):
        self.index = index
        self.interval = interval
        self.logger_q = logger_q

    def on_train_begin(self, logs):
        self.t0 = time.time()
        self.reward_min = None
        self.reward_max = None

    def on_episode_end(self, episode, logs):
        if self.reward_min is None:
            self.reward_min = logs["episode_reward"]
            self.reward_max = logs["episode_reward"]
        else:
            if self.reward_min > logs["episode_reward"]:
                self.reward_min = logs["episode_reward"]
            if self.reward_max < logs["episode_reward"]:
                self.reward_max = logs["episode_reward"]
        if time.time() - self.t0 > self.interval:
            self.t0 = time.time()
            self.logger_q.put({
                "name": "actor" + str(self.index),
                "reward_min": self.reward_min,
                "reward_max": self.reward_max,
                "nb_steps": logs["nb_steps"],
            })
            self.reward_min = None
            self.reward_max = None
    
def actor_run(
    actor_index,
    actor_func, 
    model_args,
    args,
    create_processor_func,
    experience_q, 
    model_sync_q, 
    logger_q,
    actors_end_q,
    ):
    print("Actor{} Starts!".format(actor_index))
    try:
        actor = Actor(
            actor_index,
            model_args,
            args,
            experience_q,
            model_sync_q,
            processor=create_processor_func()
        )

        # model load
        if os.path.isfile( args["load_weights_path"] ):
            actor.model.load_weights(args["load_weights_path"])

        # logger用
        callbacks = [
            ActorLogger(actor_index, logger_q, args["logger_interval"])
        ]

        # run
        actor_func(actor_index, actor, callbacks=callbacks)
    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
    finally:
        print("Actor{} End!".format(actor_index))
        actors_end_q.put(1)


from collections import deque
class Actor(rl.core.Agent):
    def __init__(self, 
        actor_index,
        model_args,
        args,
        experience_q,
        model_sync_q,
        **kwargs):
        super(Actor, self).__init__(**kwargs)

        self.actor_index = actor_index
        self.experience_q = experience_q
        self.model_sync_q = model_sync_q

        self.nb_actions = args["nb_actions"]
        self.input_shape = args["input_shape"]
        self.window_length = args["window_length"]
        self.actor_model_sync_interval = args["actor_model_sync_interval"]
        self.gamma = args["gamma"]
        self.epsilon = args["epsilon"]
        self.multireward_steps = args["multireward_steps"]
        self.action_interval = args["action_interval"]

        # local memory
        self.local_memory = deque()
        self.local_memory_update_size = args["local_memory_update_size"]

        self.learner_train_num = 0

        # reset
        self.reset_states()
        
        # model
        self.model = build_compile_model(**model_args)
        self.compiled = True

    def reset_states(self):
        self.recent_action = 0
        self.repeated_action = 0
        self.recent_reward = [0 for _ in range(self.multireward_steps)]
        self.recent_observations = [np.zeros(self.input_shape) for _ in range(self.window_length+self.multireward_steps)]

    def compile(self, optimizer, metrics=[]):
        self.compiled = True

    def load_weights(self, filepath):
        print("WARNING: Not Loaded. Please use 'load_weights_path' param.")

    def save_weights(self, filepath, overwrite=False):
        print("WARNING: Not Saved. Please use 'save_weights_path' param.")

    def forward(self, observation):
        self.recent_observations.append(observation)  # 最後に追加
        self.recent_observations.pop(0)  # 先頭を削除

        if self.training:
            # 結果を送る
            # multi step learning、nstepの報酬を割引しつつ加算する
            reward = 0
            for i, r in enumerate(reversed(self.recent_reward)):
                reward += r * (self.gamma ** i)

            state0 = self.recent_observations[:self.window_length]
            state1 = self.recent_observations[-self.window_length:]

            # priority のために TD-error をだす。
            state0_qvals = self.model.predict(np.asarray([state0]), 1)[0]
            state1_qvals = self.model.predict(np.asarray([state1]), 1)[0]
            
            maxq = np.max(state1_qvals)

            td_error = reward + self.gamma * maxq
            td_error = state0_qvals[self.recent_action] - td_error
        
            # local memoryに追加
            self.local_memory.append((state0, self.recent_action, reward, state1, td_error))

        
        # フレームスキップ(action_interval毎に行動を選択する)
        action = self.repeated_action
        if self.step % self.action_interval == 0:
            
            # 行動を決定
            if self.training:

                # ϵ-greedy法
                if self.epsilon > np.random.uniform(0, 1):
                    # ランダム
                    action = np.random.randint(0, self.nb_actions)
                else:
                    # model入力用にwindow長さ分取得
                    obs = np.asarray([self.recent_observations[-self.window_length:]])

                    # 最大のQ値を取得
                    q_values = self.model.predict(obs, batch_size=1)[0]
                    action = np.argmax(q_values)
            else:
                # model入力用にwindow長さ分取得
                obs = np.asarray([self.recent_observations[-self.window_length:]])

                # Q値が最大のもの
                q_values = self.model.predict(obs, batch_size=1)[0]
                action = np.argmax(q_values)
            
            # リピート用
            self.repeated_action = action

        # backword用に保存
        self.recent_action = action
        
        return action


    def backward(self, reward, terminal):
        self.recent_reward.append(reward)  # 最後に追加
        self.recent_reward.pop(0)  # 先頭を削除

        if self.training:
            # 一定間隔で model を learner からsyncさせる
            if self.step % self.actor_model_sync_interval == 0:
                # 要求を送る
                self.model_sync_q[0].put(1)  # 要求
            
            # weightが届いていれば更新
            if not self.model_sync_q[1].empty():
                weights = self.model_sync_q[1].get(timeout=1)
                # 空にする(念のため)
                while not self.model_sync_q[1].empty():
                    self.model_sync_q[1].get(timeout=1)
                self.model.set_weights(weights)

            # localメモリが一定量超えていれば RemoteMemory に送信
            if len(self.local_memory) > self.local_memory_update_size:
                # 共有qに送るのは重いので配列化
                data = []
                while len(self.local_memory) > 0:
                    data.append(self.local_memory.pop())
                self.experience_q.put(data)

        return []

    @property
    def layers(self):
        return self.model.layers[:]

#--------------------------------------------------------

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity= capacity
        self.index = 0
        self.buffer = []

    def add(self, experience, td_error):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = experience
        self.index = (self.index + 1) % self.capacity

    def update(self, idx, experience, td_error):
        pass

    def sample(self, batch_size, steps):
        batchs = random.sample(self.buffer, batch_size)

        indexes = np.empty(batch_size, dtype='float32')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)


#--------------------------------------------------------

import heapq
class _head_wrapper():
    def __init__(self, data):
        self.d = data
    def __eq__(self, other):
        return True

class PERGreedyMemory():
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience, td_error):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()
        
        # priority は最初は最大を選択
        priority = abs(td_error)
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-priority, experience))

    def update(self, idx, experience, td_error):
        priority = abs(td_error)
        # heapqは最小値を出すためマイナス
        experience = _head_wrapper(experience)
        heapq.heappush(self.buffer, (-priority, experience))
    
    def sample(self, batch_size, step):
        # 取り出す(学習後に再度追加)
        batchs = [heapq.heappop(self.buffer)[1].d for _ in range(batch_size)]

        indexes = np.empty(batch_size, dtype='float32')
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)

#---------------------------------------------------

#copy from https://github.com/jaromiru/AI-blog/blob/5aa9f0b/SumTree.py
import numpy

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class PERProportionalMemory():
    def __init__(self, capacity, alpha, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.alpha = alpha

        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

    def add(self, experience, td_error):
        priority = (abs(td_error) + 0.0001) ** self.alpha
        self.tree.add(priority, experience)

    def update(self, index, experience, td_error):
        priority = (abs(td_error) + 0.0001) ** self.alpha
        self.tree.update(index, priority)

    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
    
        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        total = self.tree.total()
        section = total / batch_size
        for i in range(batch_size):
            r = section*i + random.random()*section
            (idx, priority, experience) = self.tree.get(r)

            indexes.append(idx)
            batchs.append(experience)

            if self.enable_is:
                # 重要度サンプリングを計算
                weights[i] = (self.capacity * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes ,batchs, weights)

    def __len__(self):
        return self.tree.write

#------------------------------------

import bisect
class _bisect_wrapper():
    def __init__(self, data):
        self.d = data
        self.priority = 0
        self.p = 0
    def __lt__(self, o):  # a<b
        return self.priority > o.priority

class PERRankBaseMemory():
    def __init__(self, capacity, alpha, beta_initial, beta_steps, enable_is):
        self.capacity = capacity
        self.buffer = []
        self.alpha = alpha
        
        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

    def add(self, experience, td_error):
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は最後の要素を削除
            self.buffer.pop()
        
        priority = (abs(td_error) + 0.0001)  # priority を計算
        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)

    def update(self, index, experience, td_error):
        priority = (abs(td_error) + 0.0001)  # priority を計算

        experience = _bisect_wrapper(experience)
        experience.priority = priority
        bisect.insort(self.buffer, experience)


    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする。
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps

        total = 0
        for i, o in enumerate(self.buffer):
            o.index = i
            o.p = (len(self.buffer) - i) ** self.alpha 
            total += o.p
            o.p_total = total

        # 合計を均等に割り、その範囲内からそれぞれ乱数を出す。
        index_lst = []
        section = total / batch_size
        rand = []
        for i in range(batch_size):
            rand.append(section*i + random.random()*section)
        
        rand_i = 0
        for i in range(len(self.buffer)):
            if rand[rand_i] < self.buffer[i].p_total:
                index_lst.append(i)
                rand_i += 1
                if rand_i >= len(rand):
                    break

        for i, index in enumerate(reversed(index_lst)):
            o = self.buffer.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.d)
            indexes.append(index)

            if self.enable_is:
                # 重要度サンプリングを計算
                priority = o.p
                weights[i] = (self.capacity * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)


#-------------------------------------------
ENV_NAME = "Pendulum-v0"
def create_processor():
    return PendulumProcessorForDQN(enable_image=False)

def create_processor_image():
    return PendulumProcessorForDQN(enable_image=True, image_size=84)

def create_optimizer():
    return Adam()


def plot_logs(learner_logs, actors_logs):
    n = len(actors_logs) + 1
    train_num = [ l["train_num"] for l in learner_logs]
    actor_names = []
    actor_steps = []
    actor_reward_min = []
    actor_reward_max = []
    for name, logs in actors_logs.items():
        actor_names.append(name)
        steps = []
        reward_min = []
        reward_max = []
        for log in logs:
            steps.append(log["nb_steps"])
            reward_min.append(log["reward_min"])
            reward_max.append(log["reward_max"])
        actor_steps.append(steps)
        actor_reward_min.append(reward_min)
        actor_reward_max.append(reward_max)

    plt.subplot(n,1,1)
    plt.ylabel("Trains,Steps")
    plt.plot(train_num, label="train_num")
    for i in range(len(actor_names)):
        plt.plot(actor_steps[i], label=actor_names[i])
    #-- legend
    from matplotlib.font_manager import FontProperties
    plt.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.95)
    plt.legend(bbox_to_anchor=(1.00, 1), loc='upper left', borderaxespad=0, fontsize=8)

    # reward
    for i in range(len(actor_names)):
        plt.subplot(n,1,i+2)
        plt.plot(actor_reward_min[i], label="min")
        plt.plot(actor_reward_max[i], label="max")
        plt.ylabel("Actor" + str(i) + " Reward")
    plt.show()


def actor_func(index, actor, callbacks):
    env = gym.make(ENV_NAME)
    if index == 0:
        verbose = 1
    else:
        verbose = 0
    actor.fit(env, nb_steps=1000_000, visualize=False, verbose=verbose, callbacks=callbacks)


def main_no_image():
    
    env = gym.make(ENV_NAME)

    # 引数
    args = {
        # model関係
        "input_shape": env.observation_space.shape, 
        "enable_image_layer": False, 
        "nb_actions": 5, 
        "window_length": 1,     # 入力フレーム数
        "dense_units_num": 32,  # Dense層のユニット数
        "metrics": [],          # optimizer用
        "enable_dueling_network": True,  # dueling_network有効フラグ
        
        # learner 関係
        "remote_memory_capacity": 50_000,    # 確保するメモリーサイズ
        "remote_memory_warmup_size": 1000,    # 初期のメモリー確保用step数(学習しない)
        "remote_memory_type": "per_proportional", # メモリの種類
        "per_alpha": 0.8,        # PERの確率反映率
        "per_beta_initial": 0.0,     # IS反映率の初期値
        "per_beta_steps": 100_000,   # IS反映率の上昇step数
        "per_enable_is": False,      # ISを有効にするかどうか
        "batch_size": 16,            # batch_size
        "target_model_update": 1500, #  target networkのupdate間隔
        "enable_double_dqn": True,   # DDQN有効フラグ

        # actor 関係
        "local_memory_update_size": 50,    # LocalMemoryからRemoteMemoryへ投げるサイズ
        "actor_model_sync_interval": 500,  # learner から model を同期する間隔
        "gamma": 0.99,      # Q学習の割引率
        "epsilon": 0.4,        # ϵ-greedy法
        "epsilon_alpha": 4,    # ϵ-greedy法
        "multireward_steps": 3, # multistep reward
        "action_interval": 1,   # アクションを実行する間隔
        
        # その他
        "load_weights_path": "",  # 保存ファイル名
        "save_weights_path": "",  # 読み込みファイル名
        "save_overwrite": True,   # 上書き保存するか
        "logger_interval": 5,    # ログ取得間隔(秒)
    }

    manager = ApeXManager(
        actor_func=actor_func, 
        num_actors=2, 
        args=args, 
        create_processor_func=create_processor,
        create_optimizer_func=create_optimizer,
    )
    
    test_actor, learner_logs, actors_logs = manager.train()
    
    #--- plot
    plot_logs(learner_logs, actors_logs)

    #--- test
    test_actor.processor.mode = "test"
    test_actor.test(env, nb_episodes=5, visualize=True)

    

def main_image():
    
    env = gym.make(ENV_NAME)

    # 引数
    args = {
        # model関係
        "input_shape": (84,84), 
        "enable_image_layer": True, 
        "nb_actions": 5, 
        "window_length": 4,     # 入力フレーム数
        "dense_units_num": 64,  # Dense層のユニット数
        "metrics": [],          # optimizer用
        "enable_dueling_network": True,  # dueling_network有効フラグ
        
        # learner 関係
        "remote_memory_capacity": 100_000,    # 確保するメモリーサイズ
        "remote_memory_warmup_size": 200,    # 初期のメモリー確保用step数(学習しない)
        "remote_memory_type": "per_proportional", # メモリの種類
        "per_alpha": 0.8,        # PERの確率反映率
        "per_beta_initial": 0.0,     # IS反映率の初期値
        "per_beta_steps": 100_000,   # IS反映率の上昇step数
        "per_enable_is": False,      # ISを有効にするかどうか
        "batch_size": 16,            # batch_size
        "target_model_update": 1500, #  target networkのupdate間隔
        "enable_double_dqn": True,   # DDQN有効フラグ

        # actor 関係
        "local_memory_update_size": 50,    # LocalMemoryからRemoteMemoryへ投げるサイズ
        "actor_model_sync_interval": 500,  # learner から model を同期する間隔
        "gamma": 0.99,      # Q学習の割引率
        "epsilon": 0.4,        # ϵ-greedy法
        "epsilon_alpha": 1,    # ϵ-greedy法
        "multireward_steps": 1, # multistep reward
        "action_interval": 1,   # アクションを実行する間隔
        
        # その他
        "load_weights_path": "",  # 保存ファイル名
        "save_weights_path": "",  # 読み込みファイル名
        "save_overwrite": True,   # 上書き保存するか
        "logger_interval": 5,    # ログ取得間隔(秒)
    }

    manager = ApeXManager(
        actor_func=actor_func, 
        num_actors=1, 
        args=args, 
        create_processor_func=create_processor_image,
        create_optimizer_func=create_optimizer,
    )
    
    test_actor, learner_logs, actors_logs = manager.train()
    
    #--- plot
    plot_logs(learner_logs, actors_logs)

    #--- test
    test_actor.processor.mode = "test"
    test_actor.test(env, nb_episodes=5, visualize=True)

    
if __name__ == '__main__':
    # コメントアウトで切り替え
    main_no_image()
    #main_image()