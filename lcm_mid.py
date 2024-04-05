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
from diffusers import DDPMScheduler, LCMScheduler, DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union

from dataset import EnvironmentDataset, collate, get_timesteps_data, restore
from models.lcm_autoencoder import AutoEncoder
from models.trajectron import Trajectron
from utils.model_registrar import ModelRegistrar
from utils.trajectron_hypers import get_traj_hypers
import evaluation
import inspect
import easydict

import matplotlib.pyplot as plt
from evaluation.visualization import visualize_prediction


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (
            np.arange(1, ddim_timesteps + 1) * step_ratio
        ).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        """
        device上に移動させる関数

        Parameters
        ----------
        device : string
            デバイス名

        Returns
        -------
        """
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        """
        ddim_stepを行う関数

        Parameters
        ----------
        pred_x0 :
        pred_noise :
        timestep_index :
        Returns
        -------
        """
        alpha_cumprod_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape
        )
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        pred_x_0 = (
            alphas[timesteps].unsqueeze(1).unsqueeze(2) * sample
            - sigmas[timesteps].unsqueeze(1).unsqueeze(2) * model_output
        )
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5

    return c_skip, c_out


def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99) -> None:
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    Parameters
    ----------
    target_params :
        the target parameter sequence.
    source_params :
        the source parameter sequence.
    rate : int
        the EMA rate (closer to 1 means slower).
    """
    print(type())
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Parameters
    ----------
    num_inference_steps : Optional[int] = None

    device: Optional[Union[str, torch.device]] = None

    timesteps: Optional[List[int]] = None

    Returns
    -------
    timesteps : torch.Tensor

    num_inference_steps : int

    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


class MID:
    def __init__(self, config: easydict.EasyDict) -> None:
        """
        Initialize MID

        Parameters
        ----------
        config : easydict.EasyDict
            configuration of this method
        """
        self.config = config
        self.sampling_method = self.config.sampling_method
        self.sampling_steps = self.config.sampling_steps
        self.device = torch.device(self.config.device)
        torch.backends.cudnn.benchmark = True
        self._build()

    def train(self) -> None:
        """
        Training Method.
        """

        # 学習したEncoderの重みもセットする。
        # self.encoder.load_state_dict(self.checkpoint['encoder'])

        # ノイズスケジューラーを定義
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.05,
            beta_schedule="linear",
        )
        betas = noise_scheduler.betas.to(self.device)
        alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod).to(self.device)
        sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod).to(self.device)

        solver = DDIMSolver(
            noise_scheduler.alphas_cumprod.numpy(),
            timesteps=noise_scheduler.config.num_train_timesteps,
            ddim_timesteps=self.config.num_ddim_timesteps,
        ).to(self.device)

        for epoch in range(1, self.config.epochs + 1):
            self.train_dataset.augment = self.config.augment

            for node_type, data_loader in self.train_data_loader.items():
                pbar = tqdm(data_loader, ncols=80)
                for batch in pbar:

                    # batchからデータセットを取り出す。
                    (
                        first_history_index,
                        x_t,
                        y_t,
                        x_st_t,
                        y_st_t,
                        neighbors_data_st,
                        neighbors_edge_value,
                        robot_traj_st_t,
                        map,
                    ) = batch

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(y_t)
                    bsz = y_t.shape[0]

                    # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
                    topk = (
                        noise_scheduler.config.num_train_timesteps
                        // self.config.num_ddim_timesteps
                    )
                    index = torch.randint(
                        0, self.config.num_ddim_timesteps, (bsz,), device=self.device
                    ).long()
                    start_timesteps = solver.ddim_timesteps[index]
                    timesteps = start_timesteps - topk
                    timesteps = torch.where(
                        timesteps < 0, torch.zeros_like(timesteps), timesteps
                    )

                    # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
                    c_skip_start, c_out_start = scalings_for_boundary_conditions(
                        start_timesteps
                    )
                    c_skip_start, c_out_start = [
                        append_dims(x, y_t.ndim) for x in [c_skip_start, c_out_start]
                    ]
                    c_skip, c_out = scalings_for_boundary_conditions(timesteps)
                    c_skip, c_out = [append_dims(x, y_t.ndim) for x in [c_skip, c_out]]

                    # 20.4.5. Add noise to the latents according to the noise magnitude at each timestep
                    noisy_model_input = noise_scheduler.add_noise(
                        y_t, noise, start_timesteps
                    )

                    # 20.4.6. Sample a random guidance scale w from U[w_min, w_max] and embed it
                    w = (self.config.w_max - self.config.w_min) * torch.rand(
                        (bsz,)
                    ) + self.config.w_min
                    w_embedding = guidance_scale_embedding(
                        w, embedding_dim=self.config.embedding_dim
                    )
                    w_embedding = w_embedding.to(device=self.device, dtype=y_t.dtype)

                    # 画像だと、(bsz, 1, 1, 1)になっている.注意する。
                    w_reshaped = w.reshape(bsz, 1, 1).to(
                        device=self.device, dtype=y_t.dtype
                    )

                    # CFG計算のために必要なので、unconditionのembeddingも用意する。(zeroにしただけ。ちょっとうまくいかなかったので、他の関数がうまくいくか確かめるために、一旦condと同じでやってみる。)
                    uncond_emb, dynamics_uncond = self.model_teacher.encode_test(
                        batch, node_type, orzero=True
                    )

                    # 周辺情報をEncodingしておく。ts_emb(temporal-social embedding)
                    ts_emb, dynamics_cond = self.model_teacher.encode_test(
                        batch, node_type
                    )

                    noisy_model_input = noisy_model_input.to(self.device)
                    ts_emb = ts_emb.to(self.device)
                    uncond_emb = uncond_emb.to(self.device)

                    # # 生徒モデルでノイズを予測する。
                    noise_pred = self.model_student.get_noise(
                        noisy_model_input, ts_emb, betas[start_timesteps], w_embedding
                    )

                    pred_x_0 = predicted_origin(
                        noise_pred,
                        start_timesteps,
                        noisy_model_input,
                        self.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    model_pred = (
                        c_skip_start * noisy_model_input + c_out_start * pred_x_0
                    )

                    # 教師モデルでノイズを予測する。(ただし、cond, uncondの両方を予測する。
                    cond_teacher_output = self.model_teacher.get_noise(
                        noisy_model_input, ts_emb, betas[start_timesteps]
                    )
                    cond_pred_x0 = predicted_origin(
                        cond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        self.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    uncond_teacher_output = self.model_teacher.get_noise(
                        noisy_model_input, uncond_emb, betas[start_timesteps]
                    )
                    uncond_pred_x0 = predicted_origin(
                        uncond_teacher_output,
                        start_timesteps,
                        noisy_model_input,
                        self.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    # CFGスケールを考慮して最終的な予測
                    pred_x0 = cond_pred_x0 + w_reshaped * (
                        cond_pred_x0 - uncond_pred_x0
                    )
                    pred_noise = cond_teacher_output + w_reshaped * (
                        cond_teacher_output - uncond_teacher_output
                    )

                    x_prev = solver.ddim_step(pred_x0, pred_noise, index)

                    # EMAモデルの予測(少し方法を変えるので、get_pred_emaという関数を使うことにする。
                    target_noise_pred = self.model_ema.get_noise(
                        x_prev.float(), ts_emb, betas[start_timesteps], w_embedding
                    )

                    pred_x_0 = predicted_origin(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        self.config.prediction_type,
                        alpha_schedule,
                        sigma_schedule,
                    )

                    target = c_skip * x_prev + c_out * pred_x_0

                    # どちらかの方法でLoss計算
                    if self.config.loss_type == "l2":
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                        # print(loss)
                        # 毎回Lossを出すようにする。
                    elif self.config.loss_type == "huber":
                        loss = torch.mean(
                            torch.sqrt(
                                (model_pred.float() - target.float()) ** 2
                                + self.config.huber_c**2
                            )
                            - self.config.huber_c
                        )

                    # バックプロパゲーション
                    # optimizerには、生徒モデルだけのパラメータが変更されるようにしてある。
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # EMAモデルのアップデート
                    update_ema(
                        self.model_ema.parameters(),
                        self.model_student.parameters(),
                        self.config.ema_decay,
                    )

            # モデルの評価
            self.model_ema.eval()

            guidance_scale = self.config.guidance_scale

            node_type = "PEDESTRIAN"
            eval_ade_batch_errors = []
            eval_fde_batch_errors = []
            ph = self.hyperparams["prediction_horizon"]
            max_hl = self.hyperparams["maximum_history_length"]

            self.scheduler = LCMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.05,
                beta_schedule="linear",
            )
            self.scheduler.betas = self.scheduler.betas.to(self.device)

            for i, scene in enumerate(self.eval_scenes):
                print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
                for t in tqdm(range(0, scene.timesteps, 10)):
                    # eval用のデータセットを作成
                    data_timesteps = np.arange(t, t + 10)
                    batches = get_timesteps_data(
                        env=self.eval_env,
                        scene=scene,
                        t=data_timesteps,
                        node_type=node_type,
                        state=self.hyperparams["state"],
                        pred_state=self.hyperparams["pred_state"],
                        edge_types=self.eval_env.get_edge_types(),
                        min_ht=8,
                        max_ht=self.hyperparams["maximum_history_length"],
                        min_ft=12,
                        max_ft=12,
                        hyperparams=self.hyperparams,
                        max_batch_size=100,
                    )

                    if batches is None:
                        continue

                    for test_batch, nodes, timesteps_o in zip(*batches):

                        encoded_cond, dynamics_cond = self.model_ema.encode_test(
                            test_batch, node_type
                        )

                        # 最初のランダムノイズ作成: x_T
                        batch_size = len(test_batch[0])
                        guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(
                            batch_size * 1
                        )
                        timestep_cond = guidance_scale_embedding(
                            guidance_scale_tensor,
                            embedding_dim=self.config.embedding_dim,
                        )
                        timestep_cond = timestep_cond.to(self.device)

                        # 同じシーンに対して20個生成する。
                        traj_list = []
                        for j in range(20):

                            generator = torch.Generator().manual_seed(
                                self.config.seed + j
                            )
                            x_T = self.prepare_latents(
                                batch_size,
                                12,
                                2,
                                device=self.device,
                                dtype=torch.float32,
                                generator=generator,
                            )

                            # スケジューラに決めてもらう。
                            timesteps, num_inference_steps = retrieve_timesteps(
                                self.scheduler,
                                100 // self.sampling_steps,
                                device=self.device,
                            )

                            extra_step_kwargs = self.prepare_extra_step_kwargs(
                                generator, eta=0.0
                            )

                            # 7. Denoising loop
                            num_warmup_steps = (
                                len(timesteps)
                                - num_inference_steps * self.scheduler.order
                            )
                            self._num_timesteps = len(timesteps)

                            for t in timesteps:

                                index = [t.item()] * batch_size

                                noise_pred_cond = self.model_ema.get_noise(
                                    x_T,
                                    encoded_cond,
                                    self.scheduler.betas[index],
                                    timestep_cond,
                                )
                                x_T = self.scheduler.step(
                                    noise_pred_cond,
                                    t,
                                    x_T,
                                    **extra_step_kwargs,
                                    return_dict=False,
                                )[0]

                            traj_list.append(x_T)

                        traj_list = torch.stack(traj_list)
                        predicted_y_pos = dynamics_cond.integrate_samples(traj_list)
                        predictions = predicted_y_pos.cpu().detach().numpy()

                        predictions_dict = {}
                        for i, ts in enumerate(timesteps_o):
                            if ts not in predictions_dict.keys():
                                predictions_dict[ts] = dict()
                            predictions_dict[ts][nodes[i]] = np.transpose(
                                predictions[:, [i]], (1, 0, 2, 3)
                            )

                        batch_error_dict = evaluation.compute_batch_statistics(
                            predictions_dict,
                            scene.dt,
                            max_hl=max_hl,
                            ph=ph,
                            node_type_enum=self.eval_env.NodeType,
                            kde=False,
                            map=None,
                            best_of=True,
                            prune_ph_to_future=True,
                        )

                        eval_ade_batch_errors = np.hstack(
                            (eval_ade_batch_errors, batch_error_dict[node_type]["ade"])
                        )
                        eval_fde_batch_errors = np.hstack(
                            (eval_fde_batch_errors, batch_error_dict[node_type]["fde"])
                        )

            ade = np.mean(eval_ade_batch_errors)
            fde = np.mean(eval_fde_batch_errors)

            if self.config.dataset == "eth":
                ade = ade / 0.6
                fde = fde / 0.6
            elif self.config.dataset == "sdd":
                ade = ade * 50
                fde = fde * 50
            print(f"Epoch {epoch} Best Of 20: ADE: {ade} FDE: {fde}")
            self.log.info(f"Best of 20: Epoch {epoch} ADE: {ade} FDE: {fde}")

            # 10epochごとにモデルを保存する。
            if epoch % 1 == 0:

                # Saving model
                checkpoint = {
                    "encoder": self.registrar.model_dict,
                    "ddpm": self.model_ema.state_dict(),
                }
                torch.save(
                    checkpoint,
                    osp.join(
                        self.model_dir_lcm, f"{self.config.dataset}_epoch{epoch}.pt"
                    ),
                )

            self.model_student.train()

    def eval(self) -> None:
        """
        Evaluation Method.
        """
        guidance_scale = self.config.guidance_scale

        node_type = "PEDESTRIAN"
        eval_ade_batch_errors = []
        eval_fde_batch_errors = []
        ph = self.hyperparams["prediction_horizon"]
        max_hl = self.hyperparams["maximum_history_length"]

        self.scheduler = LCMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.05,
            beta_schedule="linear",
        )
        betas = self.scheduler.betas.to(self.device)

        diversity_list = []
        average_time = []

        for i, scene in enumerate(self.eval_scenes):
            print(f"----- Evaluating Scene {i + 1}/{len(self.eval_scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                # making evaluation dataset
                data_timesteps = np.arange(t, t + 10)
                batches = get_timesteps_data(
                    env=self.eval_env,
                    scene=scene,
                    t=data_timesteps,
                    node_type=node_type,
                    state=self.hyperparams["state"],
                    pred_state=self.hyperparams["pred_state"],
                    edge_types=self.eval_env.get_edge_types(),
                    min_ht=8,
                    max_ht=self.hyperparams["maximum_history_length"],
                    min_ft=12,
                    max_ft=12,
                    hyperparams=self.hyperparams,
                    max_batch_size=25,
                )
                if batches is None:
                    continue

                for test_batch, nodes, timesteps_o in zip(*batches):

                    # Batchsizeを決める
                    batch_size = len(test_batch[0])

                    # Calculate Execution time
                    start_time = time.time()

                    # Encoding Past Pedestrian's Trajectory by Trajectron++
                    encoded_cond, dynamics_cond = self.model_ema.encode_test(
                        test_batch, node_type
                    )

                    # Setting Guidance Scale
                    guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(
                        batch_size * 1
                    )

                    # Concatenate timestep_embedding and guidancescale_embedding
                    timestep_cond = guidance_scale_embedding(
                        guidance_scale_tensor, embedding_dim=self.config.embedding_dim
                    ).to(self.device)

                    # Generating 20 Future Trajectories
                    traj_list = []
                    for j in range(20):

                        # Generating initial noise
                        generator = torch.Generator(device=self.device).manual_seed(
                            self.config.seed * j
                        )
                        x_T = self.prepare_latents(
                            batch_size,
                            12,
                            2,
                            device=self.device,
                            dtype=torch.float32,
                            generator=generator,
                        )

                        extra_step_kwargs = self.prepare_extra_step_kwargs(
                            generator, eta=0.0
                        )

                        # retrieve timesteps
                        timesteps, num_inference_steps = retrieve_timesteps(
                            self.scheduler, 100 // self.sampling_steps, self.device
                        )

                        # denoising loop
                        for step in timesteps:

                            index = [step.item()] * batch_size

                            # predict noise
                            noise_pred_cond = self.model_ema.get_noise(
                                x_T, encoded_cond, betas[index], timestep_cond
                            )

                            # denoising
                            x_T = self.scheduler.step(
                                noise_pred_cond,
                                step,
                                x_T,
                                **extra_step_kwargs,
                                return_dict=False,
                            )[0]

                        traj_list.append(x_T)

                    traj_list = torch.stack(traj_list)
                    predicted_y_pos = dynamics_cond.integrate_samples(traj_list)
                    predictions = predicted_y_pos.cpu().detach().numpy()

                    # Calculate Execution time
                    end_time = time.time()
                    execution_time = end_time - start_time
                    average_time.append(execution_time)

                    predictions_dict = {}
                    for k, ts in enumerate(timesteps_o):
                        if ts not in predictions_dict.keys():
                            predictions_dict[ts] = dict()
                        predictions_dict[ts][nodes[k]] = np.transpose(
                            predictions[:, [k]], (1, 0, 2, 3)
                        )

                    # Calculate batch error(ADE, FDE)
                    batch_error_dict = evaluation.compute_batch_statistics(
                        predictions_dict,
                        scene.dt,
                        max_hl=max_hl,
                        ph=ph,
                        node_type_enum=self.eval_env.NodeType,
                        kde=False,
                        map=None,
                        best_of=True,
                        prune_ph_to_future=True,
                    )
                    eval_ade_batch_errors = np.hstack(
                        (eval_ade_batch_errors, batch_error_dict[node_type]["ade"])
                    )
                    eval_fde_batch_errors = np.hstack(
                        (eval_fde_batch_errors, batch_error_dict[node_type]["fde"])
                    )

                    # Calculate diversity of trajectories
                    batch_diversity = np.mean(batch_error_dict[node_type]["diversity"])
                    diversity_list.append(batch_diversity)

                    # Visualize results of trajectories
                    if self.config.visualize:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        visualize_prediction(ax, predictions_dict, scene.dt, max_hl, ph)
                        piture_out_dir = f"{self.model_dir_lcm}/out_picture/step{self.sampling_steps}"
                        if not os.path.exists(piture_out_dir):
                            os.makedirs(piture_out_dir)
                        plt.savefig(f"{piture_out_dir}/{i}_{t}.png", dpi=300)
                        plt.close(fig)

            ade = np.mean(eval_ade_batch_errors)
            fde = np.mean(eval_fde_batch_errors)

            # Adjust the unit to meter
            if self.config.dataset == "eth":
                ade = ade / 0.6
                fde = fde / 0.6
            elif self.config.dataset == "sdd":
                ade = ade * 50
                fde = fde * 50

        print("#######################################")
        print("Time: ", np.mean(average_time))
        print("Diversity: ", np.mean(diversity_list))
        print(f"ADE: {ade} FDE: {fde}")

    def _build(self) -> None:
        """
        Building Method
        """
        self._build_dir()

        self._build_encoder_config()
        self._build_encoder()
        self._build_model()
        if self.config.eval_mode == False:
            self._build_train_loader()
            self._build_optimizer()
        self._build_eval_loader()

        print("> Everything built.")

    def _build_dir(self) -> None:
        """
        Build Directory
        """
        if self.config.eval_mode:
            # In evaluation mode, lcm model are evaluated
            # self.model_dir_lcm = osp.join("./experiments/lcm_", self.config.exp_name)
            self.model_dir_lcm = "./experiments/lcm_eth_8"
        else:
            # In training mode, baseline model are distilled
            # and saved as a lcm model
            self.model_dir = "./experiments/baseline_eth_8"
            self.model_dir_lcm = "./experiments/lcm_eth_8"

        # self.log_writer = SummaryWriter(log_dir=self.model_dir)

        # initialize logging
        os.makedirs(self.model_dir_lcm, exist_ok=True)
        log_name = "{}.log".format(time.strftime("%Y-%m-%d-%H-%M"))
        log_name = f"{self.config.dataset}_{log_name}"

        log_dir = osp.join(self.model_dir_lcm, log_name)
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

        self.train_data_path = osp.join(
            self.config.data_dir, self.config.dataset + "_train.pkl"
        )
        self.eval_data_path = osp.join(
            self.config.data_dir, self.config.dataset + "_test.pkl"
        )
        print("> Directory built!")

    def _build_optimizer(self) -> None:
        """
        Build optimizer
        """
        self.optimizer = optim.AdamW(
            [{"params": self.model_student.parameters()}],
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0.0,
        )
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1)
        print("> Optimizer built!")

    def _build_encoder_config(self) -> None:
        """
        Build optimizer
        """
        self.hyperparams = get_traj_hypers()
        self.hyperparams["enc_rnn_dim_edge"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_edge_influence"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_history"] = self.config.encoder_dim // 2
        self.hyperparams["enc_rnn_dim_future"] = self.config.encoder_dim // 2

        epoch = self.config.epoch_baseline
        if self.config.eval_mode:
            self.registrar = ModelRegistrar(self.model_dir_lcm, "cuda")
            checkpoint_dir = osp.join(
                self.model_dir_lcm, f"{self.config.dataset}_epoch{epoch}.pt"
            )
        else:
            self.registrar = ModelRegistrar(self.model_dir, "cuda")
            checkpoint_dir = osp.join(
                self.model_dir, f"{self.config.dataset}_epoch{epoch}.pt"
            )
        self.checkpoint = torch.load(checkpoint_dir, map_location="cpu")

        self.registrar.load_models(self.checkpoint["encoder"])

        with open(self.train_data_path, "rb") as f:
            self.train_env = dill.load(f, encoding="latin1")
        with open(self.eval_data_path, "rb") as f:
            self.eval_env = dill.load(f, encoding="latin1")

    def _build_encoder(self) -> None:
        """
        Build Encoder
        """
        self.encoder = Trajectron(self.registrar, self.hyperparams, "cuda")
        self.encoder.set_environment(self.train_env)
        self.encoder.set_annealing_params()

    def _build_model(self) -> None:
        """
        Build Model

        In LCM distillation, we use Exponential Average Movement(EMA).

        """
        # Define Models.
        config = self.config
        if self.config.eval_mode == False:
            model_teacher = AutoEncoder(config, encoder=self.encoder)
            model_student = AutoEncoder(config, encoder=self.encoder)
        model_ema = AutoEncoder(config, encoder=self.encoder)

        # Move Models to a device.
        if self.config.eval_mode == False:
            self.model_teacher = model_teacher.to(self.device)
            self.model_student = model_student.to(self.device)
        self.model_ema = model_ema.to(self.device)

        new_state_dict = {}
        for key in self.checkpoint["ddpm"].keys():
            new_key = key.replace(
                "diffusion.net.", "net."
            )  # 'diffusion.net.pos_emb.pe' -> 'net.pos_emb.pe'
            new_state_dict[new_key] = self.checkpoint["ddpm"][key]

        # Load the parameter weights
        if self.config.eval_mode == False:
            self.model_teacher.load_state_dict(new_state_dict, strict=False)
            self.model_student.load_state_dict(new_state_dict, strict=False)
        self.model_ema.load_state_dict(new_state_dict, strict=False)

        print("> Model built!")

    def _build_train_loader(self) -> None:
        """
        Build data loader of training.
        """
        config = self.config
        self.train_scenes = []

        with open(self.train_data_path, "rb") as f:
            train_env = dill.load(f, encoding="latin1")

        for attention_radius_override in config.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(
                " "
            )
            train_env.attention_radius[(node_type1, node_type2)] = float(
                attention_radius
            )

        self.train_scenes = self.train_env.scenes
        self.train_scenes_sample_probs = (
            self.train_env.scenes_freq_mult_prop
            if config.scene_freq_mult_train
            else None
        )

        self.train_dataset = EnvironmentDataset(
            train_env,
            self.hyperparams["state"],
            self.hyperparams["pred_state"],
            scene_freq_mult=self.hyperparams["scene_freq_mult_train"],
            node_freq_mult=self.hyperparams["node_freq_mult_train"],
            hyperparams=self.hyperparams,
            min_history_timesteps=8,
            min_future_timesteps=self.hyperparams["prediction_horizon"],
            return_robot=not self.config.incl_robot_node,
        )
        self.train_data_loader = dict()
        for node_type_data_set in self.train_dataset:
            node_type_dataloader = utils.data.DataLoader(
                node_type_data_set,
                collate_fn=collate,
                pin_memory=True,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.preprocess_workers,
            )
            self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    def _build_eval_loader(self) -> None:
        """
        Build data loader of evaluation.
        """
        config = self.config
        self.eval_scenes = []
        eval_scenes_sample_probs = None

        if config.eval_every is not None:
            with open(self.eval_data_path, "rb") as f:
                self.eval_env = dill.load(f, encoding="latin1")

            for attention_radius_override in config.override_attention_radius:
                node_type1, node_type2, attention_radius = (
                    attention_radius_override.split(" ")
                )
                self.eval_env.attention_radius[(node_type1, node_type2)] = float(
                    attention_radius
                )

            if self.eval_env.robot_type is None and self.hyperparams["incl_robot_node"]:
                self.eval_env.robot_type = self.eval_env.NodeType[
                    0
                ]  # TODO: Make more general, allow the user to specify?
                for scene in self.eval_env.scenes:
                    scene.add_robot_from_nodes(self.eval_env.robot_type)

            self.eval_scenes = self.eval_env.scenes
            eval_scenes_sample_probs = (
                self.eval_env.scenes_freq_mult_prop
                if config.scene_freq_mult_eval
                else None
            )
            self.eval_dataset = EnvironmentDataset(
                self.eval_env,
                self.hyperparams["state"],
                self.hyperparams["pred_state"],
                scene_freq_mult=self.hyperparams["scene_freq_mult_eval"],
                node_freq_mult=self.hyperparams["node_freq_mult_eval"],
                hyperparams=self.hyperparams,
                min_history_timesteps=self.hyperparams["minimum_history_length"],
                min_future_timesteps=self.hyperparams["prediction_horizon"],
                return_robot=not config.incl_robot_node,
            )
            self.eval_data_loader = dict()
            for node_type_data_set in self.eval_dataset:
                node_type_dataloader = utils.data.DataLoader(
                    node_type_data_set,
                    collate_fn=collate,
                    pin_memory=True,
                    batch_size=config.eval_batch_size,
                    shuffle=True,
                    num_workers=config.preprocess_workers,
                )
                self.eval_data_loader[node_type_data_set.node_type] = (
                    node_type_dataloader
                )

        print("> Dataset built!")

    def prepare_extra_step_kwargs(
        self, generator: torch._C.Generator, eta: float
    ) -> Dict[str, torch._C.Generator]:
        """


        Parameters
        ----------
        generator : torch._C.Generator

        eta : float

        Returns
        -------
        extra_step_kwargs : dict

        """
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size: int,
        sequence_length: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch._C.Generator = None,
    ) -> torch.Tensor:
        """
        Make initial noise by generator.

        Parameters
        ----------
        batch_size :
            training batch size
        sequence_length : int
            Lenght of predicting timestep. By default, 12(0.4s per step, so predict 4.8s future trajectory)
        feature_dim : int
            Dimention of future trajectory. By default, 2(x cordinate, y cordinate)
        device : torch.device
        dtype : torch.dtype
        generator : torch._C.Generator

        Returns
        -------
        latent :
            initial noize.
        """
        # 形状の定義: (batch_size, sequence_length, feature_dim)
        shape = (batch_size, sequence_length, feature_dim)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents
