# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class TaxelMLPEncoder(nn.Module):
    """Finger-wise MLP encoder for flattened taxel tails.

    The env keeps the policy observation layout as:
        [base_obs, taxel_tail]
    where taxel_tail is flattened by finger. This encoder preserves base_obs,
    encodes each finger's taxel block, and returns:
        [base_obs, encoded_taxel_feature]
    """

    def __init__(
        self,
        input_size: int,
        base_obs_dim: int,
        num_fingers: int = 5,
        finger_units: list[int] | tuple[int, ...] = (256, 128, 32),
        final_units: list[int] | tuple[int, ...] | None = (128,),
        share_finger_encoder: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.base_obs_dim = int(base_obs_dim)
        self.num_fingers = int(num_fingers)
        if self.base_obs_dim <= 0 or self.base_obs_dim >= self.input_size:
            raise ValueError(
                f"base_obs_dim must be in (0, input_size), got "
                f"base_obs_dim={self.base_obs_dim}, input_size={self.input_size}"
            )
        if self.num_fingers <= 0:
            raise ValueError(f"num_fingers must be positive, got {self.num_fingers}")

        self.taxel_tail_dim = self.input_size - self.base_obs_dim
        if self.taxel_tail_dim % self.num_fingers != 0:
            raise ValueError(
                f"taxel_tail_dim={self.taxel_tail_dim} is not divisible by "
                f"num_fingers={self.num_fingers}"
            )

        self.per_finger_dim = self.taxel_tail_dim // self.num_fingers
        self.finger_units = list(finger_units)
        if len(self.finger_units) == 0:
            raise ValueError("finger_units must contain at least one output dimension")

        self.share_finger_encoder = bool(share_finger_encoder)
        if self.share_finger_encoder:
            self.shared_finger_mlp = MLP(units=self.finger_units, input_size=self.per_finger_dim)
            self.finger_mlps = None
        else:
            self.shared_finger_mlp = None
            self.finger_mlps = nn.ModuleList(
                [MLP(units=self.finger_units, input_size=self.per_finger_dim) for _ in range(self.num_fingers)]
            )

        finger_feature_dim = self.finger_units[-1] * self.num_fingers
        self.final_units = list(final_units) if final_units is not None else []
        if self.final_units:
            self.final_mlp = MLP(units=self.final_units, input_size=finger_feature_dim)
            self.output_dim = self.base_obs_dim + self.final_units[-1]
        else:
            self.final_mlp = None
            self.output_dim = self.base_obs_dim + finger_feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        base_obs = obs[:, :self.base_obs_dim]
        taxel_tail = obs[:, self.base_obs_dim:]
        finger_obs = taxel_tail.reshape(obs.shape[0], self.num_fingers, self.per_finger_dim)

        if self.share_finger_encoder:
            encoded = self.shared_finger_mlp(finger_obs.reshape(-1, self.per_finger_dim))
            encoded = encoded.reshape(obs.shape[0], self.num_fingers, -1)
        else:
            encoded = torch.stack(
                [mlp(finger_obs[:, finger_id]) for finger_id, mlp in enumerate(self.finger_mlps)],
                dim=1,
            )

        taxel_feature = encoded.reshape(obs.shape[0], -1)
        if self.final_mlp is not None:
            taxel_feature = self.final_mlp(taxel_feature)
        return torch.cat([base_obs, taxel_feature], dim=-1)


class ProprioAdaptTConv(nn.Module):
    def __init__(self, frame_shape): # frame_shape: 42 or 47_w_tactile
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(frame_shape, frame_shape),
            nn.ReLU(inplace=True),
            nn.Linear(frame_shape, frame_shape),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(frame_shape, frame_shape, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_shape, frame_shape, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_shape, frame_shape, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(frame_shape * 3, 40)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 30, frame_shape)
        x = x.permute((0, 2, 1))  # (N, frame_shape, 30)
        x = self.temporal_aggregation(x)  # (N, frame_shape, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


class DepthCNNEncoder(nn.Module):
    """
    CNN encoder for depth images.
    Processes depth images of shape [batch, height, width] -> [batch, feature_dim]
    """
    def __init__(self, height, width, units=[32, 64, 128, 256], feature_dim=256):
        super(DepthCNNEncoder, self).__init__()
        self.height = height
        self.width = width
        
        # Add channel dimension: [batch, height, width] -> [batch, 1, height, width]
        layers = []
        in_channels = 1
        
        for out_channels in units:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate feature map size after convolutions
        # Assuming stride=2 for each conv layer
        h_out = height // (2 ** len(units))
        w_out = width // (2 ** len(units))
        self.feature_map_size = h_out * w_out * units[-1]
        
        # Projection to final feature dimension
        self.projection = nn.Linear(self.feature_map_size, feature_dim)
        
    def forward(self, x):
        # x shape: [batch, height, width]
        # Add channel dimension: [batch, 1, height, width]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        # Apply CNN layers
        x = self.conv_layers(x)  # [batch, channels, h_out, w_out]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch, channels * h_out * w_out]
        
        # Project to feature dimension
        x = self.projection(x)  # [batch, feature_dim]
        
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['proprio_adapt']
        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'])

            if self.priv_info_stage2:
                self.adapt_tconv = ProprioAdaptTConv(input_shape[0]//3)

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self.adapt_tconv(obs_dict['proprio_hist'])
                # during supervised training, extrin has gt label
                extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.env_mlp(obs_dict['priv_info'])
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
        # breakpoint()
        x = self.actor_mlp(obs)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result


class ActorCriticAsymmetric(nn.Module):
    """Asymmetric Actor-Critic: actor uses only deployable obs, critic uses obs + priv_info.

    For sim2real: actor never sees privileged information, so it can be deployed directly.
    Critic has full state information for better value estimation during training.
    """
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')  # actor obs dim
        self.units = kwargs.pop('actor_units')
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        self.priv_info_dim = kwargs['priv_info_dim']

        actor_obs_dim = input_shape[0]
        critic_obs_dim = actor_obs_dim + self.priv_info_dim

        # Separate actor and critic MLPs
        actor_out_size = self.units[-1]
        critic_units = kwargs.get('critic_units', self.units)  # default: same as actor
        critic_out_size = critic_units[-1]

        self.actor_mlp = MLP(units=self.units, input_size=actor_obs_dim)
        self.critic_mlp = MLP(units=critic_units, input_size=critic_obs_dim)

        # Actor output heads
        self.mu = torch.nn.Linear(actor_out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                                  requires_grad=True)

        # Critic output head
        self.value = torch.nn.Linear(critic_out_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        """For deployment: only uses actor (no priv_info needed)."""
        obs = obs_dict['obs']
        x = self.actor_mlp(obs)
        mu = self.mu(x)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']

        # Actor: only deployable obs
        actor_x = self.actor_mlp(obs)
        mu = self.mu(actor_x)
        sigma = self.sigma

        # Critic: obs + privileged info
        priv_info = obs_dict.get('priv_info', None)
        if priv_info is not None:
            critic_input = torch.cat([obs, priv_info], dim=-1)
        else:
            # Fallback for inference without priv_info (shouldn't happen during training)
            critic_input = torch.cat([obs, torch.zeros(obs.shape[0], self.priv_info_dim, device=obs.device)], dim=-1)
        critic_x = self.critic_mlp(critic_input)
        value = self.value(critic_x)

        return mu, mu * 0 + sigma, value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': None,
            'extrin_gt': None,
        }
        return result


class ActorCriticTaxelAsymmetric(nn.Module):
    """Asymmetric actor-critic with a finger-wise taxel MLP encoder.

    This is opt-in via the PPO network config and is intended for taxel
    ablations only. It keeps the raw env observation unchanged, but compresses
    the taxel tail before feeding actor/critic MLPs.
    """

    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        self.priv_info_dim = kwargs['priv_info_dim']
        taxel_cfg = kwargs.get('taxel_encoder', {})

        actor_obs_dim = input_shape[0]
        self.taxel_encoder = TaxelMLPEncoder(
            input_size=actor_obs_dim,
            base_obs_dim=taxel_cfg.get('base_obs_dim', 523),
            num_fingers=taxel_cfg.get('num_fingers', 5),
            finger_units=taxel_cfg.get('finger_units', [256, 128, 32]),
            final_units=taxel_cfg.get('final_units', [128]),
            share_finger_encoder=taxel_cfg.get('share_finger_encoder', False),
        )
        encoded_obs_dim = self.taxel_encoder.output_dim
        critic_obs_dim = encoded_obs_dim + self.priv_info_dim

        actor_out_size = self.units[-1]
        critic_units = kwargs.get('critic_units', self.units)
        critic_out_size = critic_units[-1]

        self.actor_mlp = MLP(units=self.units, input_size=encoded_obs_dim)
        self.critic_mlp = MLP(units=critic_units, input_size=critic_obs_dim)

        self.mu = torch.nn.Linear(actor_out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                                  requires_grad=True)
        self.value = torch.nn.Linear(critic_out_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        obs = self.taxel_encoder(obs_dict['obs'])
        x = self.actor_mlp(obs)
        return self.mu(x)

    def _actor_critic(self, obs_dict):
        obs = self.taxel_encoder(obs_dict['obs'])

        actor_x = self.actor_mlp(obs)
        mu = self.mu(actor_x)
        sigma = self.sigma

        priv_info = obs_dict.get('priv_info', None)
        if priv_info is None:
            priv_info = torch.zeros(obs.shape[0], self.priv_info_dim, device=obs.device)
        critic_input = torch.cat([obs, priv_info], dim=-1)
        critic_x = self.critic_mlp(critic_input)
        value = self.value(critic_x)

        return mu, mu * 0 + sigma, value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': None,
            'extrin_gt': None,
        }
        return result


class ActorCriticAsymmetricVisual(nn.Module):
    """Asymmetric Actor-Critic with a CNN depth encoder.

    Mirrors `ActorCriticAsymmetric` exactly except:
      - both actor AND critic share a `DepthCNNEncoder` that produces a
        cnn_feature_dim feature from the depth image and concatenates it to
        the proprio obs vector;
      - critic additionally concatenates raw priv_info (NOT compressed via
        env_mlp), same as ActorCriticAsymmetric, so the value baseline is
        accurate and zero-priv-info-at-deploy is fine — the actor never
        sees priv_info to begin with.

    Inputs (obs_dict):
        'obs'         : (B, actor_obs_dim) deployable obs vector
        'depth_image' : (B, H, W) depth tensor in [0, 1]
        'priv_info'   : (B, priv_info_dim) — critic-only, may be missing at deploy
    """
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')          # actor obs dim (proprio only)
        self.units = kwargs.pop('actor_units')
        _ = kwargs.pop('priv_mlp_units', None)  # consumed for compat; unused (no env_mlp)
        self.priv_info_dim = kwargs['priv_info_dim']

        # ---- Depth CNN encoder (shared by actor and critic) ----
        self.use_visual = kwargs.get('use_visual', False)
        if not self.use_visual:
            raise ValueError(
                "ActorCriticAsymmetricVisual requires use_visual=True; "
                "use ActorCriticAsymmetric for non-visual obs."
            )
        visual_config = kwargs.get('visual_config', {})
        depth_height = visual_config.get('depth_height', 480)
        depth_width = visual_config.get('depth_width', 640)
        cnn_units = visual_config.get('cnn_units', [32, 64, 128, 256])
        cnn_feature_dim = visual_config.get('cnn_feature_dim', 256)
        self.visual_encoder = DepthCNNEncoder(
            height=depth_height, width=depth_width,
            units=cnn_units, feature_dim=cnn_feature_dim,
        )

        # ---- Actor / Critic input dims ----
        # Actor sees: proprio obs + depth features (deployable)
        actor_obs_dim = input_shape[0] + cnn_feature_dim
        # Critic sees: actor input + raw priv_info
        critic_obs_dim = actor_obs_dim + self.priv_info_dim

        actor_out_size = self.units[-1]
        critic_units = kwargs.get('critic_units', self.units)
        critic_out_size = critic_units[-1]

        self.actor_mlp = MLP(units=self.units, input_size=actor_obs_dim)
        self.critic_mlp = MLP(units=critic_units, input_size=critic_obs_dim)

        # Heads
        self.mu = torch.nn.Linear(actor_out_size, actions_num)
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
            requires_grad=True,
        )
        self.value = torch.nn.Linear(critic_out_size, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels if isinstance(m.kernel_size, tuple) else m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    def _encode(self, obs_dict):
        """Build the actor input vector: proprio obs ⊕ cnn(depth)."""
        obs = obs_dict['obs']
        depth = obs_dict['depth_image']
        visual_features = self.visual_encoder(depth)
        return torch.cat([obs, visual_features], dim=-1)

    @torch.no_grad()
    def act(self, obs_dict):
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        return {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }

    @torch.no_grad()
    def act_inference(self, obs_dict):
        """For deployment: only uses actor (no priv_info needed)."""
        actor_input = self._encode(obs_dict)
        x = self.actor_mlp(actor_input)
        mu = self.mu(x)
        return mu

    def _actor_critic(self, obs_dict):
        actor_input = self._encode(obs_dict)
        # Actor
        actor_x = self.actor_mlp(actor_input)
        mu = self.mu(actor_x)
        sigma = self.sigma
        # Critic: actor input + raw priv_info
        priv_info = obs_dict.get('priv_info', None)
        if priv_info is None:
            priv_info = torch.zeros(
                actor_input.shape[0], self.priv_info_dim,
                device=actor_input.device, dtype=actor_input.dtype,
            )
        critic_input = torch.cat([actor_input, priv_info], dim=-1)
        critic_x = self.critic_mlp(critic_input)
        value = self.value(critic_x)
        return mu, mu * 0 + sigma, value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        return {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': None,
            'extrin_gt': None,
        }


class ActorCriticVisual(nn.Module):
    """
    Actor-Critic network with CNN encoder for visual (depth) observations.
    """
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs.get('proprio_adapt', False)
        
        # Visual encoder configuration
        self.use_visual = kwargs.get('use_visual', False)
        if self.use_visual:
            visual_config = kwargs.get('visual_config', {})
            depth_height = visual_config.get('depth_height', 480)
            depth_width = visual_config.get('depth_width', 640)
            cnn_units = visual_config.get('cnn_units', [32, 64, 128, 256])
            cnn_feature_dim = visual_config.get('cnn_feature_dim', 256)
            
            self.visual_encoder = DepthCNNEncoder(
                height=depth_height,
                width=depth_width,
                units=cnn_units,
                feature_dim=cnn_feature_dim
            )
            mlp_input_shape += cnn_feature_dim
        
        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'])

            if self.priv_info_stage2:
                self.adapt_tconv = ProprioAdaptTConv(input_shape[0]//3)

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels if isinstance(m.kernel_size, tuple) else m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        extrin, extrin_gt = None, None
        
        # Process visual observations if available
        if self.use_visual and 'depth_image' in obs_dict:
            depth_images = obs_dict['depth_image']  # [batch, height, width]
            visual_features = self.visual_encoder(depth_images)  # [batch, cnn_feature_dim]
            obs = torch.cat([obs, visual_features], dim=-1)
        
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self.adapt_tconv(obs_dict['proprio_hist'])
                # during supervised training, extrin has gt label
                extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.env_mlp(obs_dict['priv_info'])
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
        
        x = self.actor_mlp(obs)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result

class ActorCriticResidual(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        # ---------------------- config ----------------------
        self.residual_scale = kwargs.pop("residual_scale", 1.0)
        base_ckpt = kwargs.pop("base_ckpt", None)
        assert base_ckpt is not None, "Residual model requires base_ckpt path"

        # ---------------------- base model ----------------------
        base_kwargs = dict(kwargs)
        print(base_kwargs)
        self.base_actions_num = kwargs.get("base_actions_num", 28)  # base actions
        self.base_obs_dim = (kwargs.get("base_obs_dim", 390), )
        self.base_priv_info_dim = kwargs.get("base_priv_info_dim", 25)
        base_kwargs["actions_num"] = self.base_actions_num
        base_kwargs["input_shape"] = self.base_obs_dim
        base_kwargs["priv_info_dim"] = self.base_priv_info_dim
        self.base_model = ActorCritic(base_kwargs)
        # load base checkpoint
        base_state = torch.load(base_ckpt, map_location="cpu")
        self.base_model.load_state_dict(base_state["model"])
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        self.base_model.eval()

        # ---------------------- residual model ----------------------
        res_kwargs = dict(kwargs)
        self.res_actions_num = kwargs.get("res_actions_num", self.base_actions_num)
        res_kwargs["actions_num"] = self.res_actions_num
        self.residual_model = ActorCritic(res_kwargs)

        

        

    @torch.no_grad()
    def act(self, obs_dict, is_train=True):
        base_obs_dict = {
            "obs": obs_dict["obs"][:, :self.base_obs_dim[0]],
            "priv_info": obs_dict["priv_info"][:, :self.base_priv_info_dim]
        }
        base_mu, _, _, _, _ = self.base_model._actor_critic(base_obs_dict)  # [N, 28]

        # residual model
        mu_r, logstd_r, value, extrin, extrin_gt = self.residual_model._actor_critic(obs_dict)  # [N, 28]

        # residual addition
        mu_r = self.residual_scale * torch.tanh(mu_r)
        mu = torch.cat([base_mu, mu_r], dim=-1)  # [N, 56]
        sigma = torch.exp(logstd_r)
        sigma = torch.cat([sigma, sigma], dim=-1)  # [N, 56]

        distr = torch.distributions.Normal(mu, sigma)

        if is_train:
            action = distr.sample()
        else:
            action = mu  # deterministic during evaluation

        neglogp = -distr.log_prob(action).sum(dim=-1)

        return {
            "actions": action,
            "values": value,
            "mus": mu,
            "sigmas": sigma,
            "residual_mu": mu_r,
            "base_mu": base_mu,
            "prev_neglogp": neglogp,
            "neglogpacs": neglogp,
            "extrin": extrin,
            "extrin_gt": extrin_gt,
        }

    @torch.no_grad()
    def act_inference(self, obs_dict):
        base_obs_dict = {
            "obs": obs_dict["obs"][:, :self.base_obs_dim[0]],
            "priv_info": obs_dict["priv_info"][:, :self.base_priv_info_dim]
        }
        base_mu, _, _, _, _ = self.base_model._actor_critic(base_obs_dict)
        mu_r, _, _, _, _ = self.residual_model._actor_critic(obs_dict)
        mu_r = self.residual_scale * torch.tanh(mu_r)
        return torch.cat([base_mu, mu_r], dim=-1)

    def forward(self, input_dict):
        prev_actions = input_dict.get("prev_actions", None)
        base_input_dict = {
            "obs": input_dict["obs"][:, :self.base_obs_dim[0]],
            "priv_info": input_dict["priv_info"][:, :self.base_priv_info_dim],
            "prev_actions": prev_actions,
        }

        # base model (no grad)
        with torch.no_grad():
            base_mu, _, _, _, _ = self.base_model._actor_critic(base_input_dict)

        # residual model
        mu_r, logstd_r, value, extrin, extrin_gt = self.residual_model._actor_critic(input_dict)
        mu_r = self.residual_scale * torch.tanh(mu_r)

        mu = torch.cat([base_mu, mu_r], dim=-1)
        sigma = torch.exp(logstd_r)
        sigma = torch.cat([sigma, sigma], dim=-1)

        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(dim=-1) if prev_actions is not None else None

        return {
            "prev_neglogp": prev_neglogp,
            "values": value,
            "entropy": entropy,
            "mus": mu,
            "sigmas": sigma,
            "residual_mu": mu_r,
            "base_mu": base_mu,
            "extrin": extrin,
            "extrin_gt": extrin_gt,
        }
class ActorCriticBimanualSplit(nn.Module):
    """Bimanual split-arm Actor-Critic.

    Designed for the Franka-Sharpa bimanual force env where:
      - obs = cat([obs_right, obs_left]) with obs_dim per side = 410
      - priv_info = cat([priv_right, priv_left]) with priv_dim per side = 40
      - actions = cat([act_right, act_left]) with action_dim per side = 29

    Key property: actor is fully split into two independent MLPs, one per arm.
    The right-arm action head cannot be perturbed by left-arm obs/feature drift,
    and vice versa. This fixes the cross-arm signal pollution that causes one
    arm (typically the support/static one) to keep oscillating during training.

    Critic is kept shared because the RL target is joint: reward = (R_r+R_l)/2.

    Required kwargs:
      actions_num      : total action dim (58 for bimanual)
      input_shape      : (total_actor_obs_dim,) e.g. (820,)
      actor_units      : list, hidden sizes for each per-arm actor MLP
      priv_mlp_units   : list, hidden sizes for critic-side priv encoder (shared)
      priv_info_dim    : total priv_info dim (80 for bimanual)
      side_obs_dim     : actor obs dim for one side (default: input_shape[0] // 2)
      side_action_dim  : action dim for one side (default: actions_num // 2)
      side_priv_dim    : priv_info dim for one side (default: priv_info_dim // 2)
    """

    def __init__(self, kwargs):
        nn.Module.__init__(self)

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        self.priv_info_dim = kwargs['priv_info_dim']

        total_actor_obs = input_shape[0]

        # Per-side dimensions (default to half; overridable via kwargs)
        self.side_obs_dim = kwargs.get('side_obs_dim', total_actor_obs // 2)
        self.side_action_dim = kwargs.get('side_action_dim', actions_num // 2)
        self.side_priv_dim = kwargs.get('side_priv_dim', self.priv_info_dim // 2)

        # Sanity checks
        assert total_actor_obs == 2 * self.side_obs_dim, \
            f"obs dim mismatch: total={total_actor_obs}, side*2={2*self.side_obs_dim}"
        assert actions_num == 2 * self.side_action_dim, \
            f"action dim mismatch: total={actions_num}, side*2={2*self.side_action_dim}"
        assert self.priv_info_dim == 2 * self.side_priv_dim, \
            f"priv dim mismatch: total={self.priv_info_dim}, side*2={2*self.side_priv_dim}"

        actor_out_size = self.units[-1]

        # ── Two fully independent actor MLPs, one per arm ──
        self.actor_mlp_right = MLP(units=self.units, input_size=self.side_obs_dim)
        self.actor_mlp_left  = MLP(units=self.units, input_size=self.side_obs_dim)

        # Per-side action heads (mu)
        self.mu_right = nn.Linear(actor_out_size, self.side_action_dim)
        self.mu_left  = nn.Linear(actor_out_size, self.side_action_dim)

        # Per-side sigma (learnable log-std, independent across arms)
        self.sigma_right = nn.Parameter(
            torch.zeros(self.side_action_dim, dtype=torch.float32), requires_grad=True)
        self.sigma_left = nn.Parameter(
            torch.zeros(self.side_action_dim, dtype=torch.float32), requires_grad=True)

        # ── Shared critic: sees full state ──
        # critic_input = [obs_right, obs_left, priv_right, priv_left] = total_actor_obs + priv_info_dim
        critic_units = kwargs.get('critic_units', self.units)
        critic_obs_dim = total_actor_obs + self.priv_info_dim
        self.critic_mlp = MLP(units=critic_units, input_size=critic_obs_dim)
        self.value = nn.Linear(critic_units[-1], 1)

        # Weight init
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma_right, 0)
        nn.init.constant_(self.sigma_left, 0)

    # ──────────────────────────────────────────────────────────
    #  Forward helpers
    # ──────────────────────────────────────────────────────────

    def _split_obs(self, obs: torch.Tensor):
        """Split the flat obs tensor into (obs_right, obs_left)."""
        return obs[:, :self.side_obs_dim], obs[:, self.side_obs_dim:]

    def _split_priv(self, priv_info: torch.Tensor):
        """Split the flat priv_info tensor into (priv_right, priv_left)."""
        return priv_info[:, :self.side_priv_dim], priv_info[:, self.side_priv_dim:]

    def _compute_mu_sigma(self, obs: torch.Tensor):
        """Run both per-arm actors and concat the outputs."""
        obs_r, obs_l = self._split_obs(obs)

        feat_r = self.actor_mlp_right(obs_r)
        feat_l = self.actor_mlp_left(obs_l)

        mu_r = self.mu_right(feat_r)
        mu_l = self.mu_left(feat_l)
        mu = torch.cat([mu_r, mu_l], dim=-1)  # [N, 58]

        # Broadcast per-arm sigma to batch dim via mu*0 trick (matches parent class)
        sigma_r = mu_r * 0 + self.sigma_right
        sigma_l = mu_l * 0 + self.sigma_left
        sigma = torch.cat([sigma_r, sigma_l], dim=-1)  # [N, 58]

        return mu, sigma

    def _compute_value(self, obs: torch.Tensor, priv_info):
        """Shared critic; sees the full bimanual state."""
        if priv_info is None:
            priv_info = torch.zeros(
                obs.shape[0], self.priv_info_dim,
                device=obs.device, dtype=obs.dtype)
        critic_in = torch.cat([obs, priv_info], dim=-1)
        x = self.critic_mlp(critic_in)
        return self.value(x)

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        priv_info = obs_dict.get('priv_info', None)

        mu, sigma = self._compute_mu_sigma(obs)  # sigma here is log-std
        value = self._compute_value(obs, priv_info)

        return mu, sigma, value

    # ──────────────────────────────────────────────────────────
    #  Public API (matches ActorCriticAsymmetric)
    # ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, obs_dict):
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        return {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }

    @torch.no_grad()
    def act_inference(self, obs_dict):
        """Deployment: actor only; priv_info not needed."""
        obs = obs_dict['obs']
        mu, _ = self._compute_mu_sigma(obs)
        return mu

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        return {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': None,
            'extrin_gt': None,
        }
