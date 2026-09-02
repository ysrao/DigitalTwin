"""PPO and Double DQN agents for the per-tier multi-tier twin.

The twin's control is per tier: the single centralized policy emits one branch
per tier (the slice-mix template that tier runs) plus one global steering
branch. That is a multi-categorical action, not an index into a flat table, so
the network is a shared trunk with one softmax head per branch — the
"action branching" architecture. It is still a single agent: one observation,
one network, one update.

The trunk and per-layer backprop reuse `engine.ActorCritic`'s routines so the
comparison against the original slice engine stays on the same building blocks.

    PPO   on-policy, clipped surrogate on the joint (summed-log-prob) ratio,
          GAE(lambda) advantages.
    DQN   Double DQN with a per-branch Q head, a shared bootstrap target
          (Branching Dueling Q-Network style), replay and a target network.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from engine import ActorCritic

Action = List[int]


# ==========================================================================
# Shared branching network
# ==========================================================================

class BranchingNet:
    """Shared 2-layer trunk with one linear head per action branch.

    `heads_softmax` decides whether `forward` returns probabilities (PPO actor)
    or raw scores (DQN Q-values). A separate 3-layer critic trunk mirrors
    `engine.ActorCritic` exactly so its backprop can be reused verbatim.
    """

    def __init__(self, seed: int, n_inputs: int, branch_sizes: Sequence[int],
                 hidden: int = 64):
        rng = np.random.default_rng(seed)
        h = int(hidden)
        self.n_inputs = int(n_inputs)
        self.branch_sizes = list(branch_sizes)
        self.hidden = h

        self.tw = [rng.normal(0, 0.12, (self.n_inputs, h)), rng.normal(0, 0.12, (h, h))]
        self.tb = [np.zeros(h), np.zeros(h)]
        self.hw = [rng.normal(0, 0.08, (h, k)) for k in self.branch_sizes]
        self.hb = [np.zeros(k) for k in self.branch_sizes]

        self.cw = [rng.normal(0, 0.12, (self.n_inputs, h)),
                   rng.normal(0, 0.12, (h, h)), rng.normal(0, 0.08, (h, 1))]
        self.cb = [np.zeros(h), np.zeros(h), np.zeros(1)]

    # -- forward ---------------------------------------------------------------

    def _trunk(self, x: np.ndarray):
        z1 = x @ self.tw[0] + self.tb[0]; h1 = np.maximum(z1, 0)
        z2 = h1 @ self.tw[1] + self.tb[1]; h2 = np.maximum(z2, 0)
        return (x, z1, h1, z2, h2)

    def heads(self, x: np.ndarray, softmax: bool) -> Tuple[List[np.ndarray], tuple]:
        cache = self._trunk(x)
        h2 = cache[-1]
        outputs = []
        for w, b in zip(self.hw, self.hb):
            out = h2 @ w + b
            if softmax:
                out = out - out.max(axis=1, keepdims=True)
                e = np.exp(out)
                out = e / e.sum(axis=1, keepdims=True)
            outputs.append(out)
        return outputs, cache

    def probs(self, x: np.ndarray) -> List[np.ndarray]:
        return self.heads(x, softmax=True)[0]

    def q(self, x: np.ndarray) -> List[np.ndarray]:
        return self.heads(x, softmax=False)[0]

    def values(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.cw[0] + self.cb[0]; h1 = np.maximum(z1, 0)
        z2 = h1 @ self.cw[1] + self.cb[1]; h2 = np.maximum(z2, 0)
        return (h2 @ self.cw[2] + self.cb[2])[:, 0]

    def _value_cache(self, x: np.ndarray):
        z1 = x @ self.cw[0] + self.cb[0]; h1 = np.maximum(z1, 0)
        z2 = h1 @ self.cw[1] + self.cb[1]; h2 = np.maximum(z2, 0)
        return (x, z1, h1, z2, h2)

    # -- backward -----------------------------------------------------------

    def apply_head_grads(self, cache, head_grads: List[np.ndarray], lr: float) -> None:
        """One SGD step: per-head gradients, then the shared trunk.

        `head_grads[k]` is dL/d(logits_k). Gradients are clipped to +/-2 as in
        `engine.ActorCritic._backprop`.
        """
        x, z1, h1, z2, h2 = cache
        gh2 = np.zeros_like(h2)
        for k, g in enumerate(head_grads):
            gw = h2.T @ g
            gb = g.sum(0)
            np.clip(gw, -2, 2, out=gw)
            np.clip(gb, -2, 2, out=gb)
            gh2 += g @ self.hw[k].T
            self.hw[k] -= lr * gw
            self.hb[k] -= lr * gb
        gz2 = gh2 * (z2 > 0)
        gw1 = h1.T @ gz2; gb1 = gz2.sum(0)
        gh1 = gz2 @ self.tw[1].T
        gz1 = gh1 * (z1 > 0)
        gw0 = x.T @ gz1; gb0 = gz1.sum(0)
        for w, b, gw, gb in ((self.tw[1], self.tb[1], gw1, gb1),
                             (self.tw[0], self.tb[0], gw0, gb0)):
            np.clip(gw, -2, 2, out=gw)
            np.clip(gb, -2, 2, out=gb)
            w -= lr * gw
            b -= lr * gb

    def fit_value(self, x: np.ndarray, returns: np.ndarray, lr: float) -> float:
        cache = self._value_cache(x)
        values = (cache[-1] @ self.cw[2] + self.cb[2])[:, 0]
        n = len(x)
        grad_v = (2 * (values - returns) / n)[:, None]
        for i, gb in ActorCritic._backprop(grad_v, cache, self.cw, lr):
            self.cb[i] -= lr * gb
        return float(np.mean((values - returns) ** 2))

    # -- checkpoint (for the DQN target network) --------------------------

    def copy_weights_from(self, other: "BranchingNet") -> None:
        self.tw = [w.copy() for w in other.tw]
        self.tb = [b.copy() for b in other.tb]
        self.hw = [w.copy() for w in other.hw]
        self.hb = [b.copy() for b in other.hb]


# ==========================================================================
# PPO
# ==========================================================================

class BranchingPPOAgent:
    name = "ppo"

    def __init__(self, obs_dim: int, branch_sizes: Sequence[int], seed: int = 3080,
                 gamma: float = 0.95, lam: float = 0.95, clip: float = 0.2,
                 lr: float = 0.012, epochs: int = 8):
        self.net = BranchingNet(seed, obs_dim, branch_sizes)
        self.branch_sizes = list(branch_sizes)
        self.rng = np.random.default_rng(seed)
        self.gamma, self.lam, self.clip, self.lr, self.epochs = gamma, lam, clip, lr, epochs

    def act(self, obs: np.ndarray, greedy: bool = False) -> Tuple[Action, float]:
        probs = self.net.probs(obs[None, :])
        action, logp = [], 0.0
        for p in probs:
            row = p[0]
            k = int(np.argmax(row)) if greedy else int(self.rng.choice(len(row), p=row))
            action.append(k)
            logp += float(np.log(max(row[k], 1e-12)))
        return action, logp

    def _gae(self, rewards, values, dones, last_value):
        n = len(rewards)
        adv = np.zeros(n)
        running = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            running = delta + self.gamma * self.lam * non_terminal * running
            adv[t] = running
            next_value = values[t]
        return adv, adv + values

    def update(self, batch: Dict[str, np.ndarray], last_value: float = 0.0) -> Dict:
        states = batch["states"]
        actions = batch["actions"]            # (n, n_branches)
        old_logp = batch["logp"]             # (n,)
        values = self.net.values(states)
        adv, returns = self._gae(batch["rewards"], values, batch["dones"], last_value)
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(states)

        for _ in range(self.epochs):
            probs, cache = self.net.heads(states, softmax=True)
            # Joint log-prob across branches, and the clipped-surrogate coeff.
            logp = np.zeros(n)
            chosen = []
            for k, p in enumerate(probs):
                pick = p[np.arange(n), actions[:, k]]
                chosen.append(pick)
                logp += np.log(np.maximum(pick, 1e-12))
            ratio = np.exp(logp - old_logp)
            active = ~(((adv_norm >= 0) & (ratio > 1 + self.clip))
                       | ((adv_norm < 0) & (ratio < 1 - self.clip)))
            coeff = -(adv_norm * ratio * active) / n

            head_grads = []
            for k, p in enumerate(probs):
                grad = -p
                grad[np.arange(n), actions[:, k]] += 1.0     # onehot - p
                grad *= coeff[:, None]
                head_grads.append(grad)
            self.net.apply_head_grads(cache, head_grads, self.lr)
            self.net.fit_value(states, returns, self.lr)

        return {"mean_advantage": float(adv.mean()), "mean_return": float(returns.mean())}


# ==========================================================================
# Double DQN with branching heads
# ==========================================================================

@dataclass
class ReplayBuffer:
    capacity: int
    obs_dim: int
    n_branches: int
    size: int = 0
    cursor: int = 0

    def __post_init__(self):
        self.states = np.zeros((self.capacity, self.obs_dim))
        self.next_states = np.zeros((self.capacity, self.obs_dim))
        self.actions = np.zeros((self.capacity, self.n_branches), dtype=int)
        self.rewards = np.zeros(self.capacity)
        self.dones = np.zeros(self.capacity)

    def add(self, state, action, reward, next_state, done) -> None:
        i = self.cursor
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.cursor = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=min(n, self.size))
        return {"states": self.states[idx], "actions": self.actions[idx],
                "rewards": self.rewards[idx], "next_states": self.next_states[idx],
                "dones": self.dones[idx]}


class BranchingDQNAgent:
    name = "dqn"

    def __init__(self, obs_dim: int, branch_sizes: Sequence[int], seed: int = 3080,
                 gamma: float = 0.95, lr: float = 0.008,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay_steps: int = 400, buffer_size: int = 20_000,
                 batch_size: int = 32, target_sync: int = 40, warmup: int = 32):
        self.branch_sizes = list(branch_sizes)
        self.online = BranchingNet(seed, obs_dim, branch_sizes)
        self.target = BranchingNet(seed, obs_dim, branch_sizes)
        self.target.copy_weights_from(self.online)
        self.buffer = ReplayBuffer(buffer_size, obs_dim, len(self.branch_sizes))
        self.rng = np.random.default_rng(seed)
        self.gamma, self.lr = gamma, lr
        self.epsilon_start, self.epsilon_end = epsilon_start, epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.batch_size, self.target_sync, self.warmup = batch_size, target_sync, warmup
        self.steps = 0

    @property
    def epsilon(self) -> float:
        t = min(1.0, self.steps / self.epsilon_decay_steps)
        return self.epsilon_start + t * (self.epsilon_end - self.epsilon_start)

    def act(self, obs: np.ndarray, greedy: bool = False) -> Tuple[Action, float]:
        if not greedy and self.rng.random() < self.epsilon:
            return [int(self.rng.integers(k)) for k in self.branch_sizes], 0.0
        q = self.online.q(obs[None, :])
        return [int(np.argmax(head[0])) for head in q], 1.0

    def observe(self, state, action, reward, next_state, done) -> None:
        self.buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        self.steps += 1
        if self.buffer.size < max(self.warmup, self.batch_size):
            return None
        batch = self.buffer.sample(self.batch_size, self.rng)
        n = len(batch["states"])

        online_next = self.online.q(batch["next_states"])
        target_next = self.target.q(batch["next_states"])
        # Double DQN: online picks, target scores; average the per-branch
        # bootstraps into one shared TD target (BDQN).
        bootstrap = np.zeros(n)
        for online_head, target_head in zip(online_next, target_next):
            best = np.argmax(online_head, axis=1)
            bootstrap += target_head[np.arange(n), best]
        bootstrap /= len(online_next)
        y = batch["rewards"] + self.gamma * bootstrap * (1 - batch["dones"])

        q_heads, cache = self.online.heads(batch["states"], softmax=False)
        head_grads = []
        loss = 0.0
        for k, head in enumerate(q_heads):
            pred = head[np.arange(n), batch["actions"][:, k]]
            error = pred - y
            loss += float(np.mean(error ** 2))
            grad = np.zeros_like(head)
            grad[np.arange(n), batch["actions"][:, k]] = np.clip(error, -1.0, 1.0) / n
            head_grads.append(grad)
        self.online.apply_head_grads(cache, head_grads, self.lr)

        if self.steps % self.target_sync == 0:
            self.target.copy_weights_from(self.online)
        return loss / len(q_heads)


# Back-compatible aliases.
PPOAgent = BranchingPPOAgent
DQNAgent = BranchingDQNAgent


# ==========================================================================
# Training loops
# ==========================================================================

def train_ppo(twin, episodes: int = 8, seed: int = 3080,
              agent: Optional[BranchingPPOAgent] = None) -> Tuple[BranchingPPOAgent, Dict]:
    agent = agent or BranchingPPOAgent(twin.obs_dim, twin.branch_sizes, seed)
    history: List[float] = []
    start = time.perf_counter()
    for episode in range(episodes):
        obs = twin.reset(seed + episode)
        states, actions, logps, rewards, dones = [], [], [], [], []
        done = False
        while not done:
            action, logp = agent.act(obs)
            states.append(obs)
            actions.append(action)
            logps.append(logp)
            obs, reward, done, _ = twin.step(action)
            rewards.append(reward)
            dones.append(float(done))
        last_value = float(agent.net.values(obs[None, :])[0])
        agent.update({
            "states": np.stack(states), "actions": np.array(actions, dtype=int),
            "logp": np.array(logps), "rewards": np.array(rewards),
            "dones": np.array(dones),
        }, last_value)
        history.append(float(np.mean(rewards)))
    return agent, {
        "algorithm": "ppo", "episodes": episodes,
        "control_steps": episodes * twin.cfg.episode_steps,
        "reward_history": history, "train_seconds": time.perf_counter() - start,
        "network": f"{twin.obs_dim} -> 64 -> 64 -> {twin.branch_sizes} (per-tier branches)",
    }


def train_dqn(twin, episodes: int = 8, seed: int = 3080,
              agent: Optional[BranchingDQNAgent] = None) -> Tuple[BranchingDQNAgent, Dict]:
    agent = agent or BranchingDQNAgent(twin.obs_dim, twin.branch_sizes, seed)
    history: List[float] = []
    losses: List[float] = []
    start = time.perf_counter()
    for episode in range(episodes):
        obs = twin.reset(seed + episode)
        rewards = []
        done = False
        while not done:
            action, _ = agent.act(obs)
            next_obs, reward, done, _ = twin.step(action)
            agent.observe(obs, action, reward, next_obs, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            rewards.append(reward)
        history.append(float(np.mean(rewards)))
    return agent, {
        "algorithm": "dqn", "episodes": episodes,
        "control_steps": episodes * twin.cfg.episode_steps,
        "reward_history": history, "final_epsilon": round(agent.epsilon, 4),
        "mean_td_loss": float(np.mean(losses)) if losses else None,
        "train_seconds": time.perf_counter() - start,
        "network": f"{twin.obs_dim} -> 64 -> 64 -> {twin.branch_sizes} (per-tier branches)",
    }
