"""The agent's update, rearranged so that it can be compiled (``training.compile_update``).

Why (measured 2026-11-15, work log): one ``TDMPC2Agent.update_on_batch`` sends about 1170 small
operations to the GPU, which is busy a quarter of the time; the rest is one CPU core handing them
out. Training speed is capped by that core (about 32 env steps/s on the rented EPYC machine,
whatever the number of simulators). Compiled, the same arithmetic is a handful of fused kernels
replayed in one call.

What is different from ``update_on_batch``: nothing in the mathematics. The two losses are written
as functions of tensors only (``_model_loss``, ``_policy_loss``): no value is read back from the
GPU inside them, no branch depends on the data, nothing is modified in place. Everything that
breaks a compiler is done between and after them, as before: ``backward``, gradient clipping, the
two Adam steps, the value-scale update, the target critics' soft update, and one read-back of all
the reported numbers at the end.

Modes (``training.compile_update``):

* ``false`` (default): this module is not used; ``update_on_batch`` runs as always.
* ``"eager"``: the rearranged update, not compiled. It draws the same random numbers in the same
  order as ``update_on_batch`` and gives the same result; ``tests/test_fast_update.py`` holds the
  two side by side. This is what proves the rearrangement.
* ``true``: the two loss functions go through ``torch.compile(mode="reduce-overhead")``. The
  compiler fuses operations and draws its random numbers its own way, so the result is the same
  update statistically, not bit for bit (rounding differs in the last digits; the noise in the
  policy's sample and the critics' dropout are other draws of the same distributions). Needs a C
  compiler on the machine (``gcc``; ``scripts/remote/setup_machine.sh`` installs it). The first
  updates take tens of seconds (compilation). If compilation fails, a warning is printed once and
  the update goes on in ``"eager"`` mode.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aux_heads import head_loss
from .world_model import SimNorm, soft_cross_entropy

# Modules whose behaviour does not depend on train / eval mode. Only Dropout does, so switching the
# mode of the Dropout modules alone equals ``model.train(...)`` and skips walking ~250 modules four
# times per update. Any other module type in the model: refuse, rather than guess.
_MODE_FREE = (nn.Linear, nn.LayerNorm, nn.Mish, nn.Identity, nn.Softplus, nn.Sigmoid, nn.Sequential,
              nn.ModuleList, nn.ModuleDict, SimNorm)

MODEL_METRICS = ("total_loss", "consistency_loss", "reward_loss", "value_loss", "termination_loss",
                 "termination_recall", "_ends", "termination_false_positive", "_not_ends")


class FastUpdate:
    """One agent's rearranged update. ``mode``: "eager" or "compiled"."""

    def __init__(self, agent, mode: str) -> None:
        if mode not in ("eager", "compiled"):
            raise ValueError(f"FastUpdate mode must be 'eager' or 'compiled', got {mode!r}")
        self.agent = agent
        self.model = agent.model
        self.mode = mode
        target = set(self.model.target_q_ensemble.modules())
        self._dropouts: List[nn.Module] = []
        for module in self.model.modules():
            if module is self.model or module in target:
                continue
            if isinstance(module, nn.Dropout):
                self._dropouts.append(module)
            elif not isinstance(module, _MODE_FREE):
                raise TypeError(f"training.compile_update: do not know whether {type(module).__name__} "
                                "behaves differently in train and eval mode")
        self._model_params = [p for group in agent.model_optimizer.param_groups for p in group["params"]]
        self._policy_params = list(self.model.policy.parameters())
        self._critic_params = list(self.model.q_ensemble.parameters())
        self._target_params = list(self.model.target_q_ensemble.parameters())
        config = agent.config
        self._smooth_coef = float(config.get("policy_smooth_coef", 0.0) or 0.0)
        self._bound_coef = float(config.get("policy_mean_bound_coef", 0.0) or 0.0)
        self._coef = torch.zeros((), device=agent.device)
        self._model_fn = self._model_loss
        self._policy_fn = self._policy_loss
        if mode == "compiled":
            self._model_fn = torch.compile(self._model_loss, mode="reduce-overhead")
            self._policy_fn = torch.compile(self._policy_loss, mode="reduce-overhead")

    # ------------------------------------------------------------ the two losses --
    def _model_loss(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor,
                    terminated: torch.Tensor, labels: Sequence[torch.Tensor]):
        """World model, critics and heads: (loss, the latent rollout detached, the reported numbers).
        Same operations in the same order as ``TDMPC2Agent.update_on_batch`` up to its backward."""
        agent, model, config = self.agent, self.model, self.agent.config
        horizon = agent.horizon
        device = observations.device
        weights = torch.pow(torch.tensor(float(config["rho"]), device=device),
                            torch.arange(horizon, device=device))
        with torch.no_grad():
            next_latent_targets = model.encode(observations[1:])
            td_targets = agent._td_target(next_latent_targets, rewards, terminated)

        latent = model.encode(observations[0])
        latent_rollout = [latent]
        consistency_loss = torch.zeros((), device=device)
        for step in range(horizon):
            latent = model.next(latent, actions[step])
            consistency_loss = consistency_loss + weights[step] * F.mse_loss(latent, next_latent_targets[step])
            latent_rollout.append(latent)
        latent_rollout_tensor = torch.stack(latent_rollout, dim=0)

        rollout_latent = latent_rollout_tensor[:-1]
        reward_logits = model.reward(rollout_latent, actions)
        q_logits = model.q(rollout_latent, actions, reduction="all")
        termination_logits = model.termination(latent_rollout_tensor[1:], logits=True)

        reward_element = soft_cross_entropy(reward_logits, rewards, model.num_bins, model.value_min,
                                            model.value_max).mean(dim=(1, 2))
        reward_loss = (reward_element * weights).sum() / horizon
        value_element = soft_cross_entropy(q_logits, td_targets.unsqueeze(0).expand(model.num_q, -1, -1, -1),
                                           model.num_bins, model.value_min, model.value_max).mean(dim=(0, 2, 3))
        value_loss = (value_element * weights).sum() / horizon
        consistency_loss = consistency_loss / horizon
        termination_loss = F.binary_cross_entropy_with_logits(termination_logits, terminated)
        total_loss = (float(config["consistency_coef"]) * consistency_loss
                      + float(config["reward_coef"]) * reward_loss
                      + float(config["value_coef"]) * value_loss
                      + float(config["termination_coef"]) * termination_loss)
        aux_losses = []
        for spec, label in zip(agent.aux_specs, labels):
            next_latents = latent_rollout_tensor[1:]
            if spec.gradient == "detached":
                next_latents = next_latents.detach()
            prediction = model.aux_heads[spec.name](next_latents)
            aux_element = head_loss(spec, prediction, label).mean(dim=(1, 2))
            aux_loss = (aux_element * weights).sum() / horizon
            total_loss = total_loss + float(spec.weight) * aux_loss
            aux_losses.append(aux_loss)

        with torch.no_grad():
            # recall and false-positive rate of the termination head; a batch without a terminal
            # step (or without a non-terminal one) divides by zero here and is reported as None
            said_end = termination_logits > 0.0
            is_end = terminated > 0.5
            ends, not_ends = is_end.sum(), (~is_end).sum()
            recall = (said_end & is_end).sum() / ends
            false_pos = (said_end & ~is_end).sum() / not_ends
            metrics = torch.stack([total_loss.detach(), consistency_loss.detach(), reward_loss.detach(),
                                   value_loss.detach(), termination_loss.detach(), recall,
                                   ends.to(recall.dtype), false_pos, not_ends.to(recall.dtype)]
                                  + [a.detach() for a in aux_losses])
        return total_loss, latent_rollout_tensor.detach(), metrics

    def _policy_loss(self, latent_rollout: torch.Tensor, actions: torch.Tensor,
                     source: Optional[torch.Tensor], expert: Optional[torch.Tensor],
                     scale: torch.Tensor, coef: torch.Tensor):
        """The policy prior's loss: (loss, the reported numbers, the value scale after this batch).
        ``scale`` is the running value scale before this batch, ``coef`` = bc_coef * bc_scale (a
        tensor, so the schedule can change it without a recompilation). The critics' parameters
        must not require gradients while this runs (the caller switches them, as
        ``_update_policy`` does)."""
        agent, model, config = self.agent, self.model, self.agent.config
        action, info = model.pi(latent_rollout)
        q_values = model.q(latent_rollout, action, reduction="avg")

        with torch.no_grad():                                   # RunningScale.update, not in place
            flat = q_values[0].detach().float().reshape(-1)
            if flat.numel() < 2:
                estimate = torch.ones_like(scale)
            else:
                percentiles = torch.quantile(flat, torch.tensor([0.05, 0.95], device=flat.device))
                estimate = (percentiles[1] - percentiles[0]).clamp(min=1.0).reshape(1)
            new_scale = torch.lerp(scale, estimate, agent.scale.tau)
        normalized_q = q_values / new_scale
        weights = torch.pow(torch.tensor(float(config["rho"]), device=latent_rollout.device),
                            torch.arange(latent_rollout.shape[0], device=latent_rollout.device))
        objective = float(config["entropy_coef"]) * info["scaled_entropy"] + normalized_q
        policy_loss = -(objective.mean(dim=(1, 2)) * weights).mean()
        reported = [info["entropy"].detach().mean()]

        target = actions if agent.bc_target == "action" else (
            None if expert is None else torch.nan_to_num(expert, nan=0.0))
        wanted = float(config.get("bc_coef", 0.0) or 0.0) > 0
        mask = agent._bc_mask(source, expert) if (wanted and target is not None) else None
        if mask is not None:
            # update_on_batch skips the cloning when the batch has no row to clone or coef is 0;
            # here the term is then exactly 0 (mask all zero, or times coef = 0): the same loss
            horizon = target.shape[0]
            limit = 1.0 if agent.policy_mean_bound is None else float(np.tanh(agent.policy_mean_bound))
            miss = (info["mean"][:horizon] - target.clamp(-limit, limit)).abs()
            if agent.bc_loss_kind == "smooth_l1":
                beta = agent.bc_loss_beta
                gap = torch.where(miss < beta, 0.5 * miss.pow(2) / beta, miss - 0.5 * beta).mean(dim=-1)
            else:
                gap = miss.pow(2).mean(dim=-1)
            rows = mask.sum()
            bc_loss = (gap * mask).sum() / rows.clamp(min=1.0)
            policy_loss = policy_loss + coef * bc_loss
            reported += [bc_loss.detach(), ((miss.mean(dim=-1) * mask).sum() / rows.clamp(min=1.0)).detach(),
                         mask.mean(), rows]
        if agent.self_bc_coef > 0 and source is not None:
            # training.self_bc_coef: the same gap to the replayed action on the kept own successes
            # (replay source 3), as in ``_update_policy``; exactly 0 when the batch has none
            own = (source == 3).float()
            horizon = actions.shape[0]
            miss = (info["mean"][:horizon] - actions.clamp(-1.0, 1.0)).abs()
            if agent.bc_loss_kind == "smooth_l1":
                beta = agent.bc_loss_beta
                gap = torch.where(miss < beta, 0.5 * miss.pow(2) / beta, miss - 0.5 * beta).mean(dim=-1)
            else:
                gap = miss.pow(2).mean(dim=-1)
            own_rows = own.sum()
            self_bc_loss = (gap * own).sum() / own_rows.clamp(min=1.0)
            policy_loss = policy_loss + agent.self_bc_coef * self_bc_loss
            reported += [self_bc_loss.detach(), own.mean(), own_rows]
        if self._smooth_coef > 0 and latent_rollout.shape[0] > 1:
            smooth_loss = (info["mean"][1:] - info["mean"][:-1]).pow(2).mean()
            policy_loss = policy_loss + self._smooth_coef * smooth_loss
            reported.append(smooth_loss.detach())
        if agent.policy_mean_bound is not None:
            size = info["mean_pre"].abs()
            reported += [(size > agent.policy_mean_bound).float().mean().detach(), size.max().detach()]
            if self._bound_coef > 0:
                bound_loss = torch.relu(size - agent.policy_mean_bound).pow(2).mean()
                policy_loss = policy_loss + self._bound_coef * bound_loss
                reported.append(bound_loss.detach())
        return policy_loss, torch.stack([policy_loss.detach()] + reported), new_scale

    # ------------------------------------------------------------------ one update --
    def _call(self, name: str, *args):
        """Run a loss function; a compilation that fails falls back to the uncompiled one, once."""
        try:
            return getattr(self, name)(*args)
        except Exception as error:                                   # noqa: BLE001 - any compiler failure
            if self.mode != "compiled":
                raise
            warnings.warn("training.compile_update: compilation failed, the update continues uncompiled "
                          f"(about as fast as with compile_update off). {type(error).__name__}: {error}",
                          RuntimeWarning, stacklevel=2)
            self.mode = "eager"
            self._model_fn, self._policy_fn = self._model_loss, self._policy_loss
            return getattr(self, name)(*args)

    def _set_dropout(self, on: bool) -> None:
        for module in self._dropouts:
            module.training = on

    def update(self, batch) -> Dict[str, Any]:
        agent, model, config = self.agent, self.model, self.agent.config
        clip = float(config["grad_clip_norm"])
        source, expert = getattr(batch, "source", None), getattr(batch, "expert", None)
        labels = tuple(batch.labels[spec.label] for spec in agent.aux_specs)
        if self.mode == "compiled":
            torch.compiler.cudagraph_mark_step_begin()

        self._set_dropout(True)
        total_loss, rollout, model_numbers = self._call("_model_fn", batch.observations, batch.actions,
                                                        batch.rewards, batch.terminated, labels)
        model_numbers = model_numbers.clone()           # a compiled function reuses its output buffers
        agent.model_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self._model_params, clip)
        agent.model_optimizer.step()

        coef = float(config.get("bc_coef", 0.0) or 0.0) * float(getattr(agent, "bc_scale", 1.0))
        self._coef.fill_(coef)
        agent.policy_optimizer.zero_grad(set_to_none=True)
        for parameter in self._critic_params:
            parameter.requires_grad_(False)
        try:
            policy_loss, policy_numbers, new_scale = self._call("_policy_fn", rollout, batch.actions, source,
                                                                expert, agent.scale.value, self._coef)
        finally:
            for parameter in self._critic_params:
                parameter.requires_grad_(True)
        with torch.no_grad():
            agent.scale.value.copy_(new_scale)
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self._policy_params, clip)
        agent.policy_optimizer.step()

        with torch.no_grad():                           # WorldModel.soft_update_target_q, in one call
            torch._foreach_lerp_(self._target_params, self._critic_params, float(config["tau"]))
        self._set_dropout(False)
        agent.update_count += 1

        # the only read-back from the GPU
        numbers = torch.cat([model_numbers, policy_numbers, model_grad_norm.detach().reshape(1),
                             policy_grad_norm.detach().reshape(1), agent.scale.value.detach().reshape(1)]).cpu()
        values = [float(v) for v in numbers]
        m = dict(zip(MODEL_METRICS, values))
        cursor = len(MODEL_METRICS)
        info: Dict[str, Any] = {
            "total_loss": m["total_loss"], "consistency_loss": m["consistency_loss"],
            "reward_loss": m["reward_loss"], "value_loss": m["value_loss"],
            "termination_loss": m["termination_loss"],
            "termination_recall": m["termination_recall"] if m["_ends"] > 0 else None,
            "termination_false_positive": m["termination_false_positive"] if m["_not_ends"] > 0 else None,
            "model_grad_norm": values[-3],
        }
        for spec in agent.aux_specs:
            info[f"aux_{spec.name}_loss"] = values[cursor]
            cursor += 1
        info["policy_loss"] = values[cursor]
        info["policy_grad_norm"] = values[-2]
        info["policy_entropy"] = values[cursor + 1]
        info["policy_scale"] = values[-1]
        cursor += 2
        target_known = agent.bc_target == "action" or expert is not None
        if float(config.get("bc_coef", 0.0) or 0.0) > 0 and target_known:
            bc_loss, bc_abs, bc_share, rows = values[cursor:cursor + 4]
            cursor += 4
            if coef > 0 and rows > 0:
                info.update(bc_loss=bc_loss, bc_share=bc_share, bc_coef=coef, bc_abs_error=bc_abs)
        if agent.self_bc_coef > 0 and source is not None:
            self_bc_loss, self_bc_share, own_rows = values[cursor:cursor + 3]
            cursor += 3
            if own_rows > 0:
                info.update(self_bc_loss=self_bc_loss, self_bc_share=self_bc_share)
        if self._smooth_coef > 0 and rollout.shape[0] > 1:
            info.update(policy_smooth_loss=values[cursor], policy_smooth_coef=self._smooth_coef)
            cursor += 1
        if agent.policy_mean_bound is not None:
            info.update(policy_mean_over_bound=values[cursor], policy_mean_abs_max=values[cursor + 1])
            cursor += 2
            if self._bound_coef > 0:
                info["policy_mean_bound_loss"] = values[cursor]
        return info
