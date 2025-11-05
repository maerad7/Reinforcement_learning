# -*- coding: utf-8 -*-
"""
PPO (PyTorch) for Pendulum-v1 — GAE(λ)
Mixed style: concise line comments + math/derivation notes

Key formulas (for reference):
- Clipped PPO objective:
    L_clip(θ) = E_t[ min( r_t(θ) * Â_t, clip(r_t(θ), 1-ε, 1+ε) * Â_t ) ]
  where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t).

- Value loss (critic):
    L_V = E_t[ (V_θv(s_t) - Ŷ_t)^2 ]

- Generalized Advantage Estimation (GAE):
    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    Â_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}

- TD target used to fit V(s_t):
    Ŷ_t = Â_t + V(s_t)

- Gaussian log-likelihood (for continuous action policy):
    log π(a|μ,σ) = -1/2 * [ (a-μ)^2 / σ^2 + log(2πσ^2) ]  (sum over action dims)
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Prefer gymnasium; fall back to gym if not available
try:
    import gymnasium as gym
except ImportError:
    import gym

# ---------------------------
# Global Hyperparameters
# ---------------------------
env_name        = 'Pendulum-v1'   # environment id
actor_lr        = 5e-4            # actor learning rate
critic_lr       = 1e-3            # critic learning rate
gamma           = 0.99            # discount factor γ
hidden_size     = 128             # hidden units
update_interval = 50              # steps per policy update
clip_ratio      = 0.1             # PPO clip ε
lmbda           = 0.95            # GAE λ
epochs          = 7               # epochs per update batch
max_episodes    = 500             # training episodes

# Save / Video settings
SAVE_DIR        = "./checkpoints"
VIDEO_DIR       = "./videos"
EVAL_EVERY_EPISODES = 20
EVAL_EPISODES       = 3
VIDEO_RECORD_IN_EVAL= True  # if you want video during eval()

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Utility Functions
# ---------------------------
def to_tensor(x, dtype=torch.float32):
    """NumPy → Tensor (device-safe)."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype, device=device)
    else:
        return torch.tensor(x, dtype=dtype, device=device)

def list_to_batch(lst):
    """Stack time-major list into a single batch array."""
    return np.concatenate(lst, axis=0) if len(lst) > 1 else lst[0]

def gaussian_log_prob(mu: torch.Tensor, std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Log π(a|μ,σ) for diagonal Gaussian (sum over dims)."""
    var = std.pow(2)
    # per-dim log prob, then sum across action dims (dim=1)
    log_probs = -0.5 * (((action - mu) ** 2) / (var + 1e-8) + torch.log(2 * torch.pi * var + 1e-8))
    return torch.sum(log_probs, dim=1, keepdim=True)

def make_env(record_video: bool = False, video_dir: str = VIDEO_DIR, video_prefix: str = "ppo-video"):
    """
    Create env; attach RecordVideo when requested.
    Note: Gymnasium requires render_mode='rgb_array' for RecordVideo.
    """
    try:
        env = gym.make(env_name, render_mode="rgb_array")
    except TypeError:
        # Old gym without render_mode kw
        env = gym.make(env_name)

    if record_video:
        try:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda ep: True,  # record all episodes
                name_prefix=video_prefix
            )
        except Exception as e:
            print(f"[WARN] RecordVideo attach failed: {e}. Video disabled.")
    return env


# ---------------------------
# Networks
# ---------------------------
class Actor(nn.Module):
    """Stochastic policy π_θ(a|s) ~ N(μ_θ(s), σ_θ(s))"""
    def __init__(self, state_size: int, action_size: int, action_bound: float, std_bound=(1e-2, 1.0)):
        super().__init__()
        self.action_bound = float(action_bound)  # env's action high (scalar for Pendulum)
        self.std_min, self.std_max = std_bound

        # shared trunk
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        # heads for μ and σ
        self.mu_head  = nn.Linear(hidden_size, action_size)
        self.std_head = nn.Linear(hidden_size, action_size)
        self.softplus = nn.Softplus()  # ensures σ > 0

    def forward(self, states: torch.Tensor):
        """Return μ(s), σ(s) for Gaussian policy."""
        z = self.net(states)
        mu = torch.tanh(self.mu_head(z)) * self.action_bound   # keep μ within action bounds
        std = self.softplus(self.std_head(z))                   # positive std
        std = torch.clamp(std, min=self.std_min, max=self.std_max)  # stabilize exploration scale
        return mu, std

    @torch.no_grad()
    def get_action(self, state_np: np.ndarray):
        """Sample action a ~ π(.|s) for training (stochastic)."""
        state = to_tensor(state_np.reshape(1, -1))
        mu, std = self.forward(state)
        action = torch.normal(mu, std)  # sample from N(μ, σ)
        action = torch.clamp(action, -self.action_bound, self.action_bound)
        logp = gaussian_log_prob(mu, std, action)  # log π_old(a|s)
        return logp.cpu().numpy(), action.cpu().numpy()

    @torch.no_grad()
    def act_deterministic(self, state_np: np.ndarray):
        """Use μ(s) for evaluation (deterministic policy)."""
        state = to_tensor(state_np.reshape(1, -1))
        mu, _ = self.forward(state)
        return torch.clamp(mu, -self.action_bound, self.action_bound).cpu().numpy()


class Critic(nn.Module):
    """State-value V_ψ(s) approximator."""
    def __init__(self, state_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, states: torch.Tensor):
        return self.net(states)


# ---------------------------
# PPO Agent
# ---------------------------
class PPOAgent:
    def __init__(self, gamma: float):
        # Env + basic specs
        self.env = make_env()
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = float(self.env.action_space.high[0])  # scalar magnitude for Pendulum
        self.std_bound = (1e-2, 1.0)

        self.gamma = gamma
        self.actor = Actor(self.state_size, self.action_size, self.action_bound, self.std_bound).to(device)
        self.critic = Critic(self.state_size).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mse = nn.MSELoss()

        self.best_eval_return = -float("inf")  # track best model by eval return

    # ---------- GAE & TD targets ----------
    def gae_target(self, rewards: np.ndarray, curr_Qs: np.ndarray, next_Q: float, done: bool):
        """
        Compute GAE advantages and TD targets.
        Vectorized over a single trajectory chunk (length T):
          δ_t   = r_t + γ * V(s_{t+1}) - V(s_t)
          Â_t  = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}
          Ŷ_t  = Â_t + V(s_t)
        """
        td_targets = np.zeros_like(rewards)  # V targets (Ŷ_t)
        gae = np.zeros_like(rewards)         # advantages (Â_t)
        gae_cum = 0.0
        R_to_go = 0.0 if done else next_Q  # bootstrap with V(s_{T}) if not done

        # reverse-time accumulation of TD residuals
        for k in reversed(range(len(rewards))):
            delta = rewards[k, 0] + gamma * R_to_go - curr_Qs[k, 0]
            gae_cum = gamma * lmbda * gae_cum + delta
            gae[k, 0] = gae_cum
            R_to_go = curr_Qs[k, 0]             # shift baseline for next step
            td_targets[k, 0] = gae[k, 0] + curr_Qs[k, 0]
        return gae, td_targets

    # ---------- PPO losses ----------
    def ppo_actor_loss(self, log_old_policy: torch.Tensor, states: torch.Tensor,
                       actions: torch.Tensor, gaes: torch.Tensor) -> torch.Tensor:
        """
        Clipped surrogate loss:
            L_clip = E[min(r * Â, clip(r, 1-ε,1+ε) * Â)]
        """
        mu, std = self.actor(states)
        log_new = gaussian_log_prob(mu, std, actions)           # log π_θ(a|s)
        ratio = torch.exp(log_new - log_old_policy.detach())    # r = π/π_old
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        surr = -torch.min(ratio * gaes.detach(), clipped * gaes.detach())  # maximize ⇒ negative for optimizer
        return surr.mean()

    def critic_loss(self, v_pred: torch.Tensor, td_targets: torch.Tensor) -> torch.Tensor:
        """MSE( V(s), Ŷ )."""
        return self.mse(v_pred, td_targets.detach())

    # ---------- Save/Load Best ----------
    def save_best(self):
        """Save best-performing actor/critic to checkpoints/best.pt"""
        path = os.path.join(SAVE_DIR, "best.pt")
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "best_eval_return": self.best_eval_return
        }, path)
        print(f"[BEST MODEL SAVED] → {path} (mean_return={self.best_eval_return:.2f})")

    def load_best(self, map_location=None):
        """Load best checkpoint (puts nets in eval mode)."""
        path = os.path.join(SAVE_DIR, "best.pt")
        ckpt = torch.load(path, map_location=map_location or device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.best_eval_return = ckpt.get("best_eval_return", self.best_eval_return)
        self.actor.eval()
        self.critic.eval()
        print(f"[BEST MODEL LOADED] ← {path} (best_eval_return={self.best_eval_return:.2f})")

    # ---------- Evaluate (optionally record video) ----------
    @torch.no_grad()
    def evaluate(self, num_episodes: int = 1, record_video: bool = False, video_prefix: str = "eval") -> float:
        """Run deterministic policy for num_episodes; optionally record video."""
        returns = []
        eval_env = make_env(record_video=record_video, video_dir=VIDEO_DIR, video_prefix=video_prefix)
        try:
            for _ in range(num_episodes):
                reset_out = eval_env.reset()
                state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                done = False
                ep_ret = 0.0
                while not done:
                    action = self.actor.act_deterministic(state)[0]  # (act_dim,)
                    step_out = eval_env.step(action)
                    if len(step_out) == 5:  # gymnasium: (obs, reward, terminated, truncated, info)
                        next_state, r, terminated, truncated, _ = step_out
                        done = terminated or truncated
                    else:  # old gym: (obs, reward, done, info)
                        next_state, r, done, _ = step_out
                    ep_ret += r
                    state = next_state
                returns.append(ep_ret)
        finally:
            try:
                eval_env.close()
            except Exception:
                pass
        return float(np.mean(returns)) if returns else 0.0

    def maybe_eval_and_save(self, episode: int):
        """Periodic evaluation; save only if mean return improved."""
        if (episode + 1) % EVAL_EVERY_EPISODES != 0:
            return
        mean_ret = self.evaluate(
            num_episodes=EVAL_EPISODES,
            record_video=False,
            video_prefix=f"eval_ep{episode+1}"
        )
        print(f"[EVAL] EP{episode+1} mean_return={mean_ret:.2f} best={self.best_eval_return:.2f}")
        if mean_ret > self.best_eval_return:
            self.best_eval_return = mean_ret
            self.save_best()

    # ---------- Training Loop ----------
    def train(self):
        for episode in range(max_episodes):
            reset_out = self.env.reset()
            state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            episode_reward = 0.0

            # Per-trajectory buffers
            states, actions, rewards, old_logps = [], [], [], []

            while not done:
                # 1) Sample action and log prob under current policy
                log_old, action = self.actor.get_action(state)

                # 2) Step environment
                step_out = self.env.step(action[0])
                if len(step_out) == 5:
                    next_state, r, terminated, truncated, _ = step_out
                    done = terminated or truncated
                else:
                    next_state, r, done, _ = step_out

                # 3) Push transition (shape to [1, ...] for batching later)
                s  = state.reshape(1, -1)
                a  = action.reshape(1, -1)
                rr = np.array([[r]], dtype=np.float32)
                nS = next_state.reshape(1, -1)
                log_old = log_old.reshape(1, 1)

                states.append(s)
                actions.append(a)
                # Reward shaping: rescale Pendulum rewards to ~[ -1, 1 ]
                rewards.append((rr + 8.0) / 8.0)
                old_logps.append(log_old)

                state = next_state
                episode_reward += r

                # 4) Policy update when chunk is ready or episode ends
                if len(states) >= update_interval or done:
                    states_b  = list_to_batch(states)
                    actions_b = list_to_batch(actions)
                    rewards_b = list_to_batch(rewards)
                    oldlog_b  = list_to_batch(old_logps)

                    # 4-1) Compute V(s_t) for batch and bootstrap V(s_T)
                    with torch.no_grad():
                        v_curr = self.critic(to_tensor(states_b))  # [T, 1]
                        v_next = self.critic(to_tensor(nS))        # [1, 1]
                        curr_Qs = v_curr.cpu().numpy()
                        next_Q  = float(v_next.cpu().numpy()[0, 0])

                    # 4-2) GAE advantages and TD targets
                    gaes_np, td_np = self.gae_target(rewards_b, curr_Qs, next_Q, done)

                    # 4-3) To tensors
                    states_t  = to_tensor(states_b)
                    actions_t = to_tensor(actions_b)
                    oldlog_t  = to_tensor(oldlog_b)
                    gaes_t    = to_tensor(gaes_np)
                    td_t      = to_tensor(td_np)

                    # 4-4) Advantage normalization (stabilizes scale)
                    gaes_t = (gaes_t - gaes_t.mean()) / (gaes_t.std() + 1e-8)

                    # 4-5) PPO update: multiple epochs over the same chunk
                    for _ in range(epochs):
                        # Actor step
                        self.actor_opt.zero_grad()
                        a_loss = self.ppo_actor_loss(oldlog_t, states_t, actions_t, gaes_t)
                        a_loss.backward()
                        self.actor_opt.step()

                        # Critic step
                        self.critic_opt.zero_grad()
                        v_pred = self.critic(states_t)
                        c_loss = self.critic_loss(v_pred, td_t)
                        c_loss.backward()
                        self.critic_opt.step()

                    # 4-6) Clear buffers for next chunk
                    states, actions, rewards, old_logps = [], [], [], []

            print(f"EP{episode+1} Reward={episode_reward:.2f}")
            self.maybe_eval_and_save(episode)


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # 1) Train and save ONLY the best checkpoint
    agent = PPOAgent(gamma=gamma)
    agent.train()
    print("Training finished. (best.pt saved when improved)")

    # 2) Load best and generate evaluation video(s)
    agent.load_best()
    mean_final = agent.evaluate(num_episodes=3, record_video=True, video_prefix="best_final")
    print(f"[BEST FINAL EVAL] mean_return={mean_final:.2f}")
    print(f"Video files are saved under: {VIDEO_DIR}/")
