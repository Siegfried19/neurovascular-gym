"""Minimal single-task TD-MPC2 training and latent-MPPI planning agent.

Copied from reference/contact-aware-tdmpc2/contact_aware_tdmpc2/tdmpc2/agent.py.
Changes, everything else verbatim:

* optional auxiliary prediction heads (``aux_heads``; G13's force head is one)
  get their own optimizer group and add a weighted loss term in the update;
* ``update(replay)`` samples and calls ``update_on_batch(batch)``, so a fixed
  batch can be fed directly (tests);
* ``aux_predict`` evaluates a head on the imagined next latent;
* ``act_batch`` / ``_plan_batch`` plan for N envs in one call (parallel
  training), each env with its own warm start; ``act`` and ``_plan`` are unchanged;
* optional action box (``bounds``, the residual box of ``planning.residual``):
  every action the planner samples or takes from the policy, and the action it
  returns, is kept inside [low, high]; the model, Q and the update do not change.
  Without a box the code path is the original one.

Original note: algorithmically this follows the official implementation at
https://github.com/nicklashansen/tdmpc2 (commit e9f5932). Environment suites,
multi-task/offline learning, pixels, Hydra, TorchRL, W&B, and compilation were
removed because they are unrelated to the stEVE state-control task.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .aux_heads import AuxHeadSpec, build_heads, head_loss
from .world_model import WorldModel, soft_cross_entropy, two_hot_inv


class RunningScale(torch.nn.Module):
    """Running 5th-to-95th-percentile value scale used by TD-MPC2's actor."""

    def __init__(self, tau: float, device: torch.device) -> None:
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("value", torch.ones(1, device=device))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1)
        if flat.numel() < 2:
            estimate = torch.ones_like(self.value)
        else:
            percentiles = torch.quantile(
                flat, torch.tensor([0.05, 0.95], device=flat.device)
            )
            estimate = (percentiles[1] - percentiles[0]).clamp(min=1.0).reshape(1)
        self.value.lerp_(estimate, self.tau)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values / self.value


Box = Tuple[torch.Tensor, torch.Tensor]       # (low, high), broadcastable to the actions


def _inside(actions: torch.Tensor, box: Box) -> torch.Tensor:
    return torch.maximum(torch.minimum(actions, box[1]), box[0])


class TDMPC2Agent:
    """TD-MPC2 agent for one low-dimensional continuous-control task."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: Mapping[str, Any],
        *,
        episode_length: int,
        device: torch.device,
        aux_heads: Sequence[AuxHeadSpec] = (),
    ) -> None:
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.config = dict(config)
        self.device = device
        model = WorldModel(self.observation_dim, self.action_dim, self.config)
        self.aux_specs = tuple(aux_heads)
        model.aux_heads = build_heads(
            model.latent_dim, int(self.config["mlp_dim"]), self.aux_specs
        )
        self.model = model.to(device)
        self.model.train(False)

        learning_rate = float(self.config["lr"])
        encoder_scale = float(self.config["enc_lr_scale"])
        model_groups = [
            {
                "params": self.model.encoder.parameters(),
                "lr": learning_rate * encoder_scale,
            },
            {"params": self.model.dynamics.parameters()},
            {"params": self.model.reward_head.parameters()},
            {"params": self.model.termination_head.parameters()},
            {"params": self.model.q_ensemble.parameters()},
        ]
        if self.aux_specs:
            model_groups.append({"params": self.model.aux_heads.parameters()})
        self.model_optimizer = torch.optim.Adam(model_groups, lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(
            self.model.policy.parameters(), lr=learning_rate, eps=1e-5
        )
        self.scale = RunningScale(float(self.config["tau"]), device)

        fraction = float(episode_length) / float(self.config["discount_denom"])
        discount = (fraction - 1.0) / fraction
        self.discount = float(
            np.clip(
                discount,
                float(self.config["discount_min"]),
                float(self.config["discount_max"]),
            )
        )
        self.horizon = int(self.config["horizon"])
        self.previous_mean = torch.zeros(
            self.horizon, self.action_dim, device=self.device
        )
        self.update_count = 0

    def _tensor_observation(self, observation: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        if tensor.shape != (self.observation_dim,):
            raise ValueError(
                f"Expected observation ({self.observation_dim},), got {tuple(tensor.shape)}"
            )
        return tensor.unsqueeze(0)

    def _box(self, bounds) -> Optional[Box]:
        """(low, high) as tensors on the device; None stays None."""
        if bounds is None:
            return None
        low, high = (torch.as_tensor(np.asarray(b, dtype=np.float32), device=self.device)
                     for b in bounds)
        return low.clamp(-1.0, 1.0), high.clamp(-1.0, 1.0)

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        *,
        first_step: bool = False,
        eval_mode: bool = False,
        bounds=None,
    ) -> np.ndarray:
        """``bounds``: optional (low, high) action box (see the module docstring)."""
        observation_tensor = self._tensor_observation(observation)
        box = self._box(bounds)
        if bool(self.config.get("mpc", True)):
            if box is None:
                action = self._plan(
                    observation_tensor, first_step=first_step, eval_mode=eval_mode
                )
            else:
                action = self._plan(
                    observation_tensor, first_step=first_step, eval_mode=eval_mode, box=box
                )
        else:
            latent = self.model.encode(observation_tensor)
            action, policy_info = self.model.pi(latent, deterministic=eval_mode)
            action = policy_info["mean"] if eval_mode else action
            action = action[0]
            if box is not None:
                action = _inside(action, box)
        return action.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def _estimate_value(
        self, latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        returns = torch.zeros(
            latent.shape[0], 1, device=self.device, dtype=latent.dtype
        )
        discount = 1.0
        termination = torch.zeros_like(returns)
        for step in range(self.horizon):
            reward_logits = self.model.reward(latent, actions[step])
            reward = two_hot_inv(
                reward_logits,
                self.model.num_bins,
                self.model.value_min,
                self.model.value_max,
            )
            latent = self.model.next(latent, actions[step])
            returns.add_(discount * (1.0 - termination) * reward)
            discount *= self.discount
            if bool(self.config.get("episodic", True)):
                termination = torch.clamp(
                    termination + (self.model.termination(latent) > 0.5).float(),
                    max=1.0,
                )
        final_action, _ = self.model.pi(latent)
        final_q = self.model.q(latent, final_action, reduction="avg")
        return returns + discount * (1.0 - termination) * final_q

    @torch.no_grad()
    def _plan(
        self,
        observation: torch.Tensor,
        *,
        first_step: bool,
        eval_mode: bool,
        box: Optional[Box] = None,
    ) -> torch.Tensor:
        latent_single = self.model.encode(observation)
        num_samples = int(self.config["num_samples"])
        num_policy = min(int(self.config["num_pi_trajs"]), num_samples)
        num_elites = int(self.config["num_elites"])
        if not 0 < num_elites <= num_samples:
            raise ValueError("num_elites must be in [1, num_samples]")

        policy_actions: Optional[torch.Tensor] = None
        if num_policy > 0:
            policy_actions = torch.empty(
                self.horizon, num_policy, self.action_dim, device=self.device
            )
            policy_latent = latent_single.repeat(num_policy, 1)
            for step in range(self.horizon):
                policy_actions[step], _ = self.model.pi(policy_latent)
                if box is not None:
                    policy_actions[step] = _inside(policy_actions[step], box)
                policy_latent = self.model.next(
                    policy_latent, policy_actions[step]
                )

        latent = latent_single.repeat(num_samples, 1)
        mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.full_like(mean, float(self.config["max_std"]))
        if not first_step:
            mean[:-1].copy_(self.previous_mean[1:])
        actions = torch.empty(
            self.horizon, num_samples, self.action_dim, device=self.device
        )
        if policy_actions is not None:
            actions[:, :num_policy].copy_(policy_actions)

        score = None
        elite_actions = None
        for _ in range(int(self.config["iterations"])):
            random_actions = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                self.horizon,
                num_samples - num_policy,
                self.action_dim,
                device=self.device,
            )
            if box is None:
                actions[:, num_policy:].copy_(random_actions.clamp(-1.0, 1.0))
            else:
                actions[:, num_policy:].copy_(_inside(random_actions, box))
            values = torch.nan_to_num(self._estimate_value(latent, actions), nan=0.0)
            elite_indices = torch.topk(values.squeeze(-1), num_elites).indices
            elite_values = values[elite_indices]
            elite_actions = actions[:, elite_indices]
            max_value = elite_values.max(dim=0).values
            score = torch.exp(
                float(self.config["temperature"]) * (elite_values - max_value)
            )
            score = score / (score.sum(dim=0, keepdim=True) + 1e-9)
            weights = score.unsqueeze(0)
            mean = (weights * elite_actions).sum(dim=1) / (
                weights.sum(dim=1) + 1e-9
            )
            variance = (weights * (elite_actions - mean.unsqueeze(1)).pow(2)).sum(
                dim=1
            ) / (weights.sum(dim=1) + 1e-9)
            std = variance.sqrt().clamp(
                float(self.config["min_std"]), float(self.config["max_std"])
            )

        assert score is not None and elite_actions is not None
        self.previous_mean.copy_(mean)
        if eval_mode:
            # Deterministic final selection; planning samples remain reproducible
            # when evaluate.py seeds PyTorch at each episode.
            action = mean[0]
        else:
            elite_index = torch.multinomial(score.squeeze(-1), 1).squeeze(0)
            action = elite_actions[0, elite_index] + std[0] * torch.randn_like(std[0])
        if box is not None:
            return _inside(action, box)
        return action.clamp(-1.0, 1.0)

    @torch.no_grad()
    def act_batch(
        self,
        observations: np.ndarray,
        *,
        first_step: Sequence[bool],
        previous_means: torch.Tensor,
        eval_mode: bool = False,
        bounds: Optional[Sequence[Any]] = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """``act`` for N envs at once. ``previous_means`` (N, horizon, action_dim)
        holds each env's warm start; the new ones are returned with the actions
        and the caller keeps them. ``self.previous_mean`` is not touched.
        ``bounds``: optional per env (low, high) box or None (no box)."""
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        if observation_tensor.ndim != 2 or observation_tensor.shape[1] != self.observation_dim:
            raise ValueError(
                f"Expected observations (N, {self.observation_dim}), "
                f"got {tuple(observation_tensor.shape)}"
            )
        first = torch.as_tensor(np.asarray(first_step, dtype=bool), device=self.device)
        means = previous_means.to(self.device)
        box = self._batch_box(bounds, observation_tensor.shape[0])
        if bool(self.config.get("mpc", True)):
            if box is None:
                action, means = self._plan_batch(
                    observation_tensor, first_step=first, previous_means=means,
                    eval_mode=eval_mode,
                )
            else:
                action, means = self._plan_batch(
                    observation_tensor, first_step=first, previous_means=means,
                    eval_mode=eval_mode, box=box,
                )
        else:
            latent = self.model.encode(observation_tensor)
            action, policy_info = self.model.pi(latent, deterministic=eval_mode)
            action = policy_info["mean"] if eval_mode else action
            if box is not None:
                action = _inside(action, box)
        return action.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32), means

    def _batch_box(self, bounds: Optional[Sequence[Any]], n: int) -> Optional[Box]:
        """Per-env boxes as (N, action_dim) tensors; envs without a box get [-1, 1].
        None when no env has a box."""
        if bounds is None or all(b is None for b in bounds):
            return None
        if len(bounds) != n:
            raise ValueError(f"bounds for {len(bounds)} envs, {n} observations")
        low = torch.full((n, self.action_dim), -1.0, device=self.device)
        high = torch.full((n, self.action_dim), 1.0, device=self.device)
        for i, b in enumerate(bounds):
            if b is not None:
                low[i], high[i] = self._box(b)
        return low, high

    def _planning_noise(self, shape: Sequence[int]) -> torch.Tensor:
        return torch.randn(*shape, device=self.device)

    @torch.no_grad()
    def _plan_batch(
        self,
        observations: torch.Tensor,
        *,
        first_step: torch.Tensor,
        previous_means: torch.Tensor,
        eval_mode: bool,
        box: Optional[Box] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``_plan`` for N observations: same steps, with an env axis after the
        horizon axis. Env i only sees its own samples, elites and warm start
        (tests/test_parallel.py checks it against one env at a time and, for
        N = 1, against ``_plan``). ``box``: (low, high), each (N, action_dim)."""
        n = observations.shape[0]
        horizon, action_dim = self.horizon, self.action_dim
        latent_single = self.model.encode(observations)
        num_samples = int(self.config["num_samples"])
        num_policy = min(int(self.config["num_pi_trajs"]), num_samples)
        num_elites = int(self.config["num_elites"])
        if not 0 < num_elites <= num_samples:
            raise ValueError("num_elites must be in [1, num_samples]")

        actions = torch.empty(
            horizon, n, num_samples, action_dim, device=self.device
        )
        if num_policy > 0:
            policy_latent = latent_single.repeat_interleave(num_policy, dim=0)
            if box is not None:
                policy_box = (box[0].repeat_interleave(num_policy, dim=0),
                              box[1].repeat_interleave(num_policy, dim=0))
            for step in range(horizon):
                policy_action, _ = self.model.pi(policy_latent)
                if box is not None:
                    policy_action = _inside(policy_action, policy_box)
                actions[step, :, :num_policy] = policy_action.view(n, num_policy, action_dim)
                policy_latent = self.model.next(policy_latent, policy_action)

        latent = latent_single.repeat_interleave(num_samples, dim=0)
        mean = torch.zeros(horizon, n, action_dim, device=self.device)
        std = torch.full_like(mean, float(self.config["max_std"]))
        warm = ~first_step
        mean[:-1, warm] = previous_means[warm, 1:].transpose(0, 1)
        rows = torch.arange(n, device=self.device).unsqueeze(1)

        score = None
        elite_actions = None
        for _ in range(int(self.config["iterations"])):
            noise = self._planning_noise(
                (horizon, n, num_samples - num_policy, action_dim)
            )
            if box is None:
                actions[:, :, num_policy:] = (
                    mean.unsqueeze(2) + std.unsqueeze(2) * noise
                ).clamp(-1.0, 1.0)
            else:
                actions[:, :, num_policy:] = _inside(
                    mean.unsqueeze(2) + std.unsqueeze(2) * noise,
                    (box[0][None, :, None], box[1][None, :, None]),
                )
            values = torch.nan_to_num(
                self._estimate_value(
                    latent, actions.view(horizon, n * num_samples, action_dim)
                ),
                nan=0.0,
            ).view(n, num_samples)
            elite_indices = torch.topk(values, num_elites, dim=1).indices
            elite_values = values.gather(1, elite_indices)
            elite_actions = actions[:, rows, elite_indices]        # (H, n, elites, A)
            max_value = elite_values.max(dim=1, keepdim=True).values
            score = torch.exp(
                float(self.config["temperature"]) * (elite_values - max_value)
            )
            score = score / (score.sum(dim=1, keepdim=True) + 1e-9)
            weights = score.unsqueeze(0).unsqueeze(-1)
            mean = (weights * elite_actions).sum(dim=2) / (
                weights.sum(dim=2) + 1e-9
            )
            variance = (weights * (elite_actions - mean.unsqueeze(2)).pow(2)).sum(
                dim=2
            ) / (weights.sum(dim=2) + 1e-9)
            std = variance.sqrt().clamp(
                float(self.config["min_std"]), float(self.config["max_std"])
            )

        assert score is not None and elite_actions is not None
        if eval_mode:
            action = mean[0]
        else:
            elite_index = torch.multinomial(score, 1).squeeze(1)
            action = elite_actions[0, rows.squeeze(1), elite_index] + std[0] * torch.randn_like(
                std[0]
            )
        if box is not None:
            return _inside(action, box), mean.transpose(0, 1).contiguous()
        return action.clamp(-1.0, 1.0), mean.transpose(0, 1).contiguous()

    @torch.no_grad()
    def _td_target(
        self,
        next_latent: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        next_action, _ = self.model.pi(next_latent)
        next_q = self.model.q(
            next_latent, next_action, reduction="min", target=True
        )
        return reward + self.discount * (1.0 - terminated) * next_q

    def update(self, replay_buffer) -> Dict[str, float]:
        return self.update_on_batch(replay_buffer.sample(self.device))

    def update_on_batch(self, batch) -> Dict[str, float]:
        observations, actions = batch.observations, batch.actions
        rewards, terminated = batch.rewards, batch.terminated
        rho = float(self.config["rho"])
        weights = torch.pow(
            torch.tensor(rho, device=self.device),
            torch.arange(self.horizon, device=self.device),
        )

        with torch.no_grad():
            next_latent_targets = self.model.encode(observations[1:])
            td_targets = self._td_target(
                next_latent_targets, rewards, terminated
            )

        self.model.train(True)
        latent = self.model.encode(observations[0])
        latent_rollout = [latent]
        consistency_loss = torch.zeros((), device=self.device)
        for step in range(self.horizon):
            latent = self.model.next(latent, actions[step])
            consistency_loss = consistency_loss + weights[step] * F.mse_loss(
                latent, next_latent_targets[step]
            )
            latent_rollout.append(latent)
        latent_rollout_tensor = torch.stack(latent_rollout, dim=0)

        rollout_latent = latent_rollout_tensor[:-1]
        reward_logits = self.model.reward(rollout_latent, actions)
        q_logits = self.model.q(rollout_latent, actions, reduction="all")
        termination_logits = self.model.termination(
            latent_rollout_tensor[1:], logits=True
        )

        reward_element = soft_cross_entropy(
            reward_logits,
            rewards,
            self.model.num_bins,
            self.model.value_min,
            self.model.value_max,
        ).mean(dim=(1, 2))
        reward_loss = (reward_element * weights).sum() / self.horizon

        value_element = soft_cross_entropy(
            q_logits,
            td_targets.unsqueeze(0).expand(self.model.num_q, -1, -1, -1),
            self.model.num_bins,
            self.model.value_min,
            self.model.value_max,
        ).mean(dim=(0, 2, 3))
        value_loss = (value_element * weights).sum() / self.horizon
        consistency_loss = consistency_loss / self.horizon
        termination_loss = F.binary_cross_entropy_with_logits(
            termination_logits, terminated
        )
        total_loss = (
            float(self.config["consistency_coef"]) * consistency_loss
            + float(self.config["reward_coef"]) * reward_loss
            + float(self.config["value_coef"]) * value_loss
            + float(self.config["termination_coef"]) * termination_loss
        )
        aux_info = {}
        for spec in self.aux_specs:
            # Heads read the imagined next latents f(z_t, a_t), t = 0..H-1.
            next_latents = latent_rollout_tensor[1:]
            if spec.gradient == "detached":
                next_latents = next_latents.detach()
            prediction = self.model.aux_heads[spec.name](next_latents)
            aux_element = head_loss(
                spec, prediction, batch.labels[spec.label]
            ).mean(dim=(1, 2))
            aux_loss = (aux_element * weights).sum() / self.horizon
            total_loss = total_loss + float(spec.weight) * aux_loss
            aux_info[f"aux_{spec.name}_loss"] = float(aux_loss.detach().cpu())

        self.model_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in self.model_optimizer.param_groups
                for parameter in group["params"]
            ],
            float(self.config["grad_clip_norm"]),
        )
        self.model_optimizer.step()

        policy_info = self._update_policy(latent_rollout_tensor.detach())
        self.model.soft_update_target_q(float(self.config["tau"]))
        self.model.train(False)
        self.update_count += 1
        return {
            "total_loss": float(total_loss.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "termination_loss": float(termination_loss.detach().cpu()),
            "model_grad_norm": float(model_grad_norm.detach().cpu()),
            **aux_info,
            **policy_info,
        }

    @torch.no_grad()
    def aux_predict(
        self, name: str, observation: np.ndarray, action: np.ndarray
    ) -> float:
        """Head ``name`` on the imagined next latent f(E(o), a)."""
        latent = self.model.encode(self._tensor_observation(observation))
        action_tensor = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).reshape(1, self.action_dim)
        next_latent = self.model.next(latent, action_tensor)
        return float(self.model.aux_heads[name](next_latent)[0, 0].item())

    def _update_policy(self, latent_rollout: torch.Tensor) -> Dict[str, float]:
        self.policy_optimizer.zero_grad(set_to_none=True)
        action, info = self.model.pi(latent_rollout)

        critic_parameters = list(self.model.q_ensemble.parameters())
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        q_values = self.model.q(latent_rollout, action, reduction="avg")
        for parameter in critic_parameters:
            parameter.requires_grad_(True)

        self.scale.update(q_values[0])
        normalized_q = self.scale(q_values)
        weights = torch.pow(
            torch.tensor(float(self.config["rho"]), device=self.device),
            torch.arange(latent_rollout.shape[0], device=self.device),
        )
        objective = (
            float(self.config["entropy_coef"]) * info["scaled_entropy"]
            + normalized_q
        )
        policy_loss = -(objective.mean(dim=(1, 2)) * weights).mean()
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.policy.parameters(), float(self.config["grad_clip_norm"])
        )
        self.policy_optimizer.step()
        return {
            "policy_loss": float(policy_loss.detach().cpu()),
            "policy_grad_norm": float(policy_grad_norm.detach().cpu()),
            "policy_entropy": float(info["entropy"].detach().mean().cpu()),
            "policy_scale": float(self.scale.value.detach().cpu()),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "scale": self.scale.state_dict(),
            "previous_mean": self.previous_mean.detach().cpu(),
            "update_count": self.update_count,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "discount": self.discount,
        }

    def load_state_dict(
        self, state: Mapping[str, Any], *, load_optimizers: bool = True
    ) -> None:
        if int(state["observation_dim"]) != self.observation_dim or int(
            state["action_dim"]
        ) != self.action_dim:
            raise ValueError("Checkpoint observation/action dimensions do not match")
        self.model.load_state_dict(state["model"])
        if load_optimizers:
            self.model_optimizer.load_state_dict(state["model_optimizer"])
            self.policy_optimizer.load_state_dict(state["policy_optimizer"])
        self.scale.load_state_dict(state["scale"])
        self.previous_mean.copy_(state["previous_mean"].to(self.device))
        self.update_count = int(state.get("update_count", 0))
        self.model.train(False)
