import os
import argparse
import torch
import dill
import pdb
import numpy as np
import os.path as osp
import logging
import time
from torch import nn, optim, utils
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm
import pickle
import json

from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation

import matplotlib.pyplot as plt
from evaluation.visualization import visualize_prediction

class MID():
    def __init__(self, config):
        self.config = config
        self.sampling_method = self.config.sampling_method
        self.sampling_steps = self.config.sampling_steps
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            self.train_dataset.augment = self.config.augment
            for node_type, data_loader in self.train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                for batch in pbar:
                    self.optimizer.zero_grad()
                    train_loss = self.model.get_loss(batch, node_type)
                    pbar.set_description(f"Epoch {epoch}, {node_type} MSE: {train_loss.item():.2f}")
                    train_loss.backward()
                    self.optimizer.step()

            self.train_dataset.augment = False
            if epoch % self.config.eval_every == 0:
                self.model.eval()

                node_type = "PEDESTRIAN"
                eval_ade_batch_errors = []
                eval_fde_batch_errors = []

                ph = self.hyperparams['prediction_horizon']
                max_hl = self.hyperparams['maximum_history_length']

                for i, scene in enumerate(self.eval_scenes):
                    print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                    for t in tqdm(range(0, scene.timesteps, 10)):
                        timesteps = np.arange(t,t+10)
                        batches = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                                       pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                                       min_ht=8, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                                       max_ft=12, hyperparams=self.hyperparams, max_batch_size=50)
                        if batches is None:
                            continue
                        
                        for test_batch, nodes, timesteps_o in zip(*batches):
                            
                            traj_pred = self.model.generate(test_batch, node_type, num_points=12, sample=20,bestof=True, sampling=self.sampling_method, step=self.sampling_steps)  # B * 20 * 12 * 2

                            predictions = traj_pred
                            predictions_dict = {}
                            for i, ts in enumerate(timesteps_o):
                                if ts not in predictions_dict.keys():
                                    predictions_dict[ts] = dict()
                                predictions_dict[ts][nodes[i]] = np.transpose(predictions[:, [i]], (1, 0, 2, 3))

                            batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                                scene.dt,
                                                                                max_hl=max_hl,
                                                                                ph=ph,
                                                                                node_type_enum=self.eval_env.NodeType,
                                                                                kde=False,
                                                                                map=None,
                                                                                best_of=True,
                                                                                prune_ph_to_future=True)

                            eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                            eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))



                ade = np.mean(eval_ade_batch_errors)
                fde = np.mean(eval_fde_batch_errors)

                if self.config.dataset == "eth":
                    ade = ade/0.6
                    fde = fde/0.6
                elif self.config.dataset == "sdd":
                    ade = ade * 50
                    fde = fde * 50

                print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
                self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")

                self.model.train()

                # モデルを保存
                checkpoint = {
                    'encoder': self.registrar.model_dict,
                    'ddpm': self.model.state_dict()
                }
                torch.save(checkpoint, osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"))


    def eval(self):
        epoch = self.config.eval_at

        self.log.info(f"Sampling: {sampling_method} Stride: {sampling_steps}")

        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []
        ph = self.hyperparams['prediction_horizon']
        max_hl = self.hyperparams['maximum_history_length']

        diversity_list=[]
        average_time = []
        for i, scene in enumerate(self.eval_scenes):
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t,t+10)
                batches = get_timesteps_data(env=self.eval_env, scene=scene, t=timesteps, node_type=node_type, state=self.hyperparams['state'],
                            pred_state=self.hyperparams['pred_state'], edge_types=self.eval_env.get_edge_types(),
                            min_ht=8, max_ht=self.hyperparams['maximum_history_length'], min_ft=12,
                            max_ft=12, hyperparams=self.hyperparams, max_batch_size=50)

                if batches is None:
                    continue

                for test_batch, nodes, timesteps_o in zip(*batches):

                    # Batchsizeを決める
                    batch_size = len(test_batch[0])
                    
                    # 計測開始
                    start_time = time.time()

                    traj_pred = self.model.generate(test_batch, node_type, num_points=12, sample=20,bestof=True, sampling=self.sampling_method, step=self.sampling_steps) # B * 20 * 12 * 2

                    # 計測終了&実行時間の測定
                    end_time = time.time()
                    execution_time = (end_time - start_time)
                    average_time.append(execution_time)

                    predictions = traj_pred
                    predictions_dict = {}
                    for j, ts in enumerate(timesteps_o):
                        if ts not in predictions_dict.keys():
                            predictions_dict[ts] = dict()
                        predictions_dict[ts][nodes[j]] = np.transpose(predictions[:, [j]], (1, 0, 2, 3))
                    
                    batch_error_dict = evaluation.compute_batch_statistics(predictions_dict,
                                                                        scene.dt,
                                                                        max_hl=max_hl,
                                                                        ph=ph,
                                                                        node_type_enum=self.eval_env.NodeType,
                                                                        kde=False,
                                                                        map=None,
                                                                        best_of=True,
                                                                        prune_ph_to_future=True)

                    eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[node_type]['ade']))
                    eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[node_type]['fde']))
                    batch_diversity = np.mean(batch_error_dict[node_type]['diversity'])
                    diversity_list.append(batch_diversity)

                    # 可視化するための関数
                    if self.config.visualize:
                        fig, ax = plt.subplots(figsize=(10, 6))  # プロットサイズを設定
                        visualize_prediction(ax, predictions_dict, scene.dt, max_hl, ph)  # 予測の可視化
                        piture_out_dir = f"{self.model_dir}/out_picture/step{sampling_steps}"
                        if not os.path.exists(piture_out_dir):
                            os.makedirs(piture_out_dir)
                        plt.savefig(f"{piture_out_dir}/{i}_{t}.png", dpi=300)  # 画像として保存
                        plt.close(fig)

        ade = np.mean(eval_ade_batch_errors)
        fde = np.mean(eval_fde_batch_errors)
            
        # mに調整する
        if self.config.dataset == "eth":
            ade = ade/0.6
            fde = fde/0.6
        elif self.config.dataset == "sdd":
            ade = ade * 50
            fde = fde * 50
        
        print("#######################################")
        print("Time: ", np.mean(average_time))
        print("Diversity: ", np.mean(diversity_list))
        print(f"ADE: {ade} FDE: {fde})")


    def _build(self):
        self._build_dir()

        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        self._build_train_loader()
        self._build_eval_loader()

        self._build_optimizer()

        #self._build_offline_scene_graph()
        #pdb.set_trace()
        print("> Everything built. Have fun :)")

    def _build_dir(self):
        self.model_dir = osp.join("./experiments",self.config.exp_name)
        self.log_writer = SummaryWriter(log_dir=self.model_dir)
        os.makedirs(self.model_dir,exist_ok=True)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir, log_name)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        handler = logging.FileHandler(log_dir)
        handler.setLevel(logging.INFO)
        self.log.addHandler(handler)

        self.log.info("Config:")
        self.log.info(self.config)
        self.log.info("\n")
        self.log.info("Eval on:")
        self.log.info(self.config.dataset)
        self.log.info("\n")

        self.train_data_path = osp.join(self.config.data_dir,self.config.dataset + "_train.pkl")
        self.eval_data_path = osp.join(self.config.data_dir,self.config.dataset + "_test.pkl")
        print("> Directory built!")

    def _build_optimizer(self):
        self.optimizer = optim.Adam([{'params': self.registrar.get_all_but_name_match('map_encoder').parameters()},
                                     {'params': self.model.parameters()}
                                    ],
                                    lr=self.config.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.98)
        print("> Optimizer built!")

    def _build_encoder_config(self):

        self.hyperparams = get_traj_hypers()
        self.hyperparams['enc_rnn_dim_edge'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_edge_influence'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_history'] = self.config.encoder_dim//2
        self.hyperparams['enc_rnn_dim_future'] = self.config.encoder_dim//2
        # registar
        self.registrar = ModelRegistrar(self.model_dir, "cuda")

        if self.config.eval_mode:
            epoch = self.config.eval_at
            checkpoint_dir = osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt")
            self.checkpoint = torch.load(osp.join(self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"), map_location = "cpu")

            self.registrar.load_models(self.checkpoint['encoder'])


        with open(self.train_data_path, 'rb') as f:
            self.train_env = dill.load(f, encoding='latin1')
        with open(self.eval_data_path, 'rb') as f:
            self.eval_env = dill.load(f, encoding='latin1')

    def _build_encoder(self):
        self.encoder = Trajectron(self.registrar, self.hyperparams, "cuda")

        self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()


    def _build_model(self):
        """ Define Model """
        config = self.config
        model = AutoEncoder(config, encoder = self.encoder)

        self.model = model.cuda()
        if self.config.eval_mode:
            self.model.load_state_dict(self.checkpoint['ddpm'])

        print("> Model built!")

    def _build_train_loader(self):
        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, 'rb') as f:
            train_env = dill.load(f, encoding='latin1')

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)


        self.train_scenes = self.train_env.scenes
        self.train_scenes_sample_probs = self.train_env.scenes_freq_mult_prop if config.scene_freq_mult_train else None

        self.train_dataset = EnvironmentDataset(train_env,
                                           self.hyperparams['state'],
                                           self.hyperparams['pred_state'],
                                           scene_freq_mult=self.hyperparams['scene_freq_mult_train'],
                                           node_freq_mult=self.hyperparams['node_freq_mult_train'],
                                           hyperparams=self.hyperparams,
                                           min_history_timesteps=8,
                                           min_future_timesteps=self.hyperparams['prediction_horizon'],
                                           return_robot=not self.config.incl_robot_node)
        self.train_data_loader = dict()
        for node_type_data_set in self.train_dataset:
            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory = True,
                                                         batch_size=self.config.batch_size,
                                                         shuffle=True,
                                                         num_workers=self.config.preprocess_workers)
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader


    def _build_eval_loader(self):
        config = self.config
        self.eval_scenes = []
        eval_scenes_sample_probs = None

        if config.eval_every is not None:
            with open(self.eval_data_path, 'rb') as f:
                self.eval_env = dill.load(f, encoding='latin1')

            for attention_radius_override in config.override_attention_radius:
                node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

            if self.eval_env.robot_type is None and self.hyperparams['incl_robot_node']:
                self.eval_env.robot_type = self.eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes
            eval_scenes_sample_probs = self.eval_env.scenes_freq_mult_prop if config.scene_freq_mult_eval else None
            self.eval_dataset = EnvironmentDataset(self.eval_env,
                                              self.hyperparams['state'],
                                              self.hyperparams['pred_state'],
                                              scene_freq_mult=self.hyperparams['scene_freq_mult_eval'],
                                              node_freq_mult=self.hyperparams['node_freq_mult_eval'],
                                              hyperparams=self.hyperparams,
                                              min_history_timesteps=self.hyperparams['minimum_history_length'],
                                              min_future_timesteps=self.hyperparams['prediction_horizon'],
                                              return_robot=not config.incl_robot_node)
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                             collate_fn=collate,
                                                             pin_memory=True,
                                                             batch_size=config.eval_batch_size,
                                                             shuffle=True,
                                                             num_workers=config.preprocess_workers)
                self.eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print("> Dataset built!")

    # def _build_offline_scene_graph(self):
    #     if self.hyperparams['offline_scene_graph'] == 'yes':
    #         print(f"Offline calculating scene graphs")
    #         for i, scene in enumerate(self.train_scenes):
    #             scene.calculate_scene_graph(self.train_env.attention_radius,
    #                                         self.hyperparams['edge_addition_filter'],
    #                                         self.hyperparams['edge_removal_filter'])
    #             print(f"Created Scene Graph for Training Scene {i}")

    #         for i, scene in enumerate(self.eval_scenes):
    #             scene.calculate_scene_graph(self.eval_env.attention_radius,
    #                                         self.hyperparams['edge_addition_filter'],
    #                                         self.hyperparams['edge_removal_filter'])
    #             print(f"Created Scene Graph for Evaluation Scene {i}")
