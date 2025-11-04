# ===============================================================
# 🧠 A3C (Asynchronous Advantage Actor-Critic) with PyTorch
#      - Continuous Action (Pendulum-v1)
#      - Multi-threaded (CPU cores)
# ===============================================================

import os
import math
import time
import threading
from threading import Thread, Lock
from multiprocessing import cpu_count
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym.wrappers import RecordVideo
import pathlib
# ===============================================================
# 🔧 기본 설정값
# ===============================================================
actor_lr = 5e-4            # Actor 학습률
critic_lr = 1e-3           # Critic 학습률
gamma = 0.99               # 할인율 (discount factor)
hidden_size = 128          # 은닉층 뉴런 개수
update_interval = 50       # 몇 step마다 글로벌 업데이트할지
max_episodes = 500         # 전체 학습 episode 수
entropy_beta = 1e-3        # 엔트로피 항 가중치 (탐험성)
grad_clip = 5.0            # gradient clipping 한계
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 장치 선택
torch.set_default_dtype(torch.float64)  # float64 정밀도 사용 (TensorFlow 코드 호환)

GLOBAL_EP = 0              # 전역 episode 카운터
GLOBAL_EP_LOCK = Lock()    # 스레드 동기화용 Lock
PRINT_LOCK = Lock()        # 출력 동기화용 Lock (출력 꼬임 방지)

# ===============================================================
# 🧩 Gym API 버전 호환 헬퍼
# ===============================================================
def reset_env(env):
    """Gym 버전에 따라 reset() 반환값이 (obs, info)인 경우가 있으므로 호환 처리"""
    out = env.reset()
    if isinstance(out, tuple):
        s, info = out
        return s
    return out

def step_env(env, action) -> Tuple[np.ndarray, float, bool, dict]:
    """Gym 버전별 step() 반환값 호환"""
    out = env.step(action)
    if len(out) == 5:  # Gym v26+: (obs, reward, terminated, truncated, info)
        ns, r, term, trunc, info = out
        done = bool(term or trunc)
        return ns, r, done, info
    else:              # Old API: (obs, reward, done, info)
        ns, r, done, info = out
        return ns, r, bool(done), info


# ===============================================================
# 🧱 Actor (정책 네트워크)
# ===============================================================
class Actor(nn.Module):
    def __init__(self, state_size, action_size, action_bound):
        super().__init__()
        self.action_bound = float(action_bound)  # 환경의 액션 최대값

        # 두 개의 은닉층 (ReLU 활성화)
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # 평균(μ)과 로그표준편차(logσ)를 출력하는 두 개의 헤드
        self.mu_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)

        # 가중치 초기화 (He initialization)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """입력 상태 → μ, σ 출력"""
        h = self.net(x)
        mu = torch.tanh(self.mu_head(h)) * self.action_bound  # tanh → [-1,1] → 범위 스케일
        log_std = self.log_std_head(h)
        # log_std 클램핑: 너무 작은 std(폭발 방지)
        log_std = torch.clamp(log_std, math.log(1e-2), math.log(1.0))
        std = torch.exp(log_std)
        return mu, std

    def sample_action(self, state_tensor):
        """현재 정책에서 액션 샘플링"""
        with torch.no_grad():
            mu, std = self(state_tensor)
            dist = torch.distributions.Normal(mu, std)
            a = dist.sample()  # 정규분포에서 샘플
            logp = dist.log_prob(a).sum(dim=-1, keepdim=True)  # 로그 확률 (손실 계산용)
            # 환경의 액션 범위로 클리핑
            a = torch.clamp(a, -self.action_bound, self.action_bound)
        return a.cpu().numpy()[0], logp

    def log_prob_and_entropy(self, states, actions):
        """정책 로그확률과 엔트로피 계산"""
        mu, std = self(states)
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(actions).sum(dim=-1, keepdim=True)   # log π(a|s)
        ent = dist.entropy().sum(dim=-1, keepdim=True)            # 엔트로피(탐험성)
        return logp, ent


# ===============================================================
# 🧱 Critic (가치 함수)
# ===============================================================
class Critic(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """상태 가치 V(s) 반환"""
        return self.net(x)


# ===============================================================
# 🌐 Global A3C: 글로벌 네트워크 및 옵티마이저 관리
# ===============================================================
class GlobalA3C:
    def __init__(self, env_name: str):
        env = gym.make(env_name)
        self.env_name = env_name
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.action_bound = float(env.action_space.high[0])

        # 글로벌 Actor & Critic 생성
        self.actor = Actor(self.state_size, self.action_size, self.action_bound).to(device)
        self.critic = Critic(self.state_size).to(device)

        # 옵티마이저 각각 분리 (TensorFlow 버전과 동일)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 스레드 동기화를 위한 Lock
        self.update_lock = Lock()

    def apply_grads(self, actor_loss, critic_loss, actor_params, critic_params):
        """워커가 계산한 loss를 이용해 글로벌 파라미터 갱신"""
        with self.update_lock:
            # ---- Actor ----
            self.actor_opt.zero_grad()
            actor_loss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
            self.actor_opt.step()

            # ---- Critic ----
            self.critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
            self.critic_opt.step()

    def sync_to_local(self, local_actor: Actor, local_critic: Critic):
        """글로벌 → 로컬 네트워크로 파라미터 복사"""
        local_actor.load_state_dict(self.actor.state_dict())
        local_critic.load_state_dict(self.critic.state_dict())

        # 모델 저장용 함수 (선택)

    def save(self, actor_path: str, critic_path: str):
        os.makedirs("Video/continuous", exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join("Video/continuous", actor_path))
        torch.save(self.critic.state_dict(), os.path.join("Video/continuous", critic_path))

    def load(self, actor_path: str, critic_path: str, map_location=None):
        map_location = map_location or device
        self.actor.load_state_dict(torch.load(actor_path, map_location=map_location))
        self.critic.load_state_dict(torch.load(critic_path, map_location=map_location))
        self.actor.to(device).eval()
        self.critic.to(device).eval()

# ===============================================================
# 🧵 Worker: 병렬 학습 스레드
# ===============================================================
class Worker(Thread):
    def __init__(self, wid: int, env_name: str, global_agent: GlobalA3C):
        super().__init__(daemon=True)
        self.wid = wid
        self.name = f"w{wid}"
        self.env = gym.make(env_name)
        self.global_agent = global_agent

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.action_bound = float(self.env.action_space.high[0])

        # 로컬 네트워크 (초기엔 글로벌과 동일하게 시작)
        self.actor = Actor(self.state_size, self.action_size, self.action_bound).to(device)
        self.critic = Critic(self.state_size).to(device)
        self.sync_with_global()  # 글로벌 가중치로 초기화

        # 경험 버퍼
        self.buffer_s = []
        self.buffer_a = []
        self.buffer_r = []
        self.buffer_logp = []

    def sync_with_global(self):
        """글로벌 네트워크로부터 최신 파라미터 복사"""
        self.global_agent.sync_to_local(self.actor, self.critic)

    def compute_td_target(self, reward, next_state, done):
        """1-step TD 타깃 계산"""
        with torch.no_grad():
            if done:
                return torch.tensor([[reward]], dtype=torch.float64, device=device)
            ns = torch.from_numpy(next_state).to(device).unsqueeze(0).to(torch.float64)
            v_next = self.critic(ns)
            return reward + gamma * v_next

    def push_transition(self, s, a, r, logp):
        """버퍼에 transition 저장"""
        self.buffer_s.append(s)
        self.buffer_a.append(a)
        self.buffer_r.append(r)
        self.buffer_logp.append(logp)

    def clear_buffers(self):
        """버퍼 초기화"""
        self.buffer_s.clear()
        self.buffer_a.clear()
        self.buffer_r.clear()
        self.buffer_logp.clear()

    def run(self):
        """워커의 메인 루프"""
        global GLOBAL_EP

        while True:
            # ---- 종료 조건 체크 ----
            with GLOBAL_EP_LOCK:
                if GLOBAL_EP >= max_episodes:
                    break
                ep_idx = GLOBAL_EP + 1  # 현재 에피소드 번호

            # 환경 초기화
            s = reset_env(self.env)
            ep_ret = 0.0
            done = False
            self.clear_buffers()

            # ---- 한 에피소드 실행 ----
            while not done:
                # 현재 상태를 텐서로 변환
                st = torch.from_numpy(np.asarray(s)).to(device).unsqueeze(0).to(torch.float64)
                # 액션 샘플링
                action_np, logp_t = self.actor.sample_action(st)
                action_env = np.asarray(action_np, dtype=np.float64)

                # 환경 스텝
                ns, r, done, _ = step_env(self.env, action_env)
                ep_ret += r

                # 보상 스케일링 (TF 코드의 (r+8)/8)
                r_scaled = (r + 8.0) / 8.0

                # 버퍼 저장
                self.push_transition(
                    s=np.asarray(s, dtype=np.float64),
                    a=action_np,
                    r=r_scaled,
                    logp=logp_t.cpu().numpy()
                )

                s = ns

                # 일정 스텝마다 글로벌 네트워크 업데이트
                if len(self.buffer_s) >= update_interval or done:
                    self.update_global(ns, done)
                    self.sync_with_global()
                    self.clear_buffers()

            # ---- 에피소드 종료 후 ----
            with GLOBAL_EP_LOCK:
                GLOBAL_EP += 1
                ep_no = GLOBAL_EP

            with PRINT_LOCK:
                print(f"{self.name} | EP{ep_no} EpisodeReward={ep_ret:.2f}")

    def update_global(self, next_state, done):
        """로컬 네트워크에서 그라디언트를 계산하고 글로벌 네트워크에 적용"""
        # numpy → torch 변환
        states = torch.from_numpy(np.vstack(self.buffer_s)).to(device).to(torch.float64)
        actions = torch.from_numpy(np.vstack(self.buffer_a)).to(device).to(torch.float64)
        logps_old = torch.from_numpy(np.vstack(self.buffer_logp)).to(device).to(torch.float64)
        rewards = torch.from_numpy(np.vstack(self.buffer_r)).to(device).to(torch.float64)

        # ---- n-step TD 타깃 계산 ----
        with torch.no_grad():
            if done:
                v_next = torch.zeros((1, 1), dtype=torch.float64, device=device)
            else:
                ns_t = torch.from_numpy(np.asarray(next_state)).to(device).unsqueeze(0).to(torch.float64)
                v_next = self.critic(ns_t)

        # 리턴(누적 보상) 계산 (뒤에서부터 discount 누적)
        returns = []
        R = v_next.squeeze(0)
        for r in reversed(rewards):
            R = r + gamma * R
            returns.append(R)
        returns = returns[::-1]
        returns = torch.stack(returns).unsqueeze(-1)  # [T, 1] 형태

        # 현재 가치 V(s)
        values = self.critic(states)
        advantages = returns - values  # A(s,a) = R - V(s)

        # 정책 로그확률 및 엔트로피 계산
        logp, entropy = self.actor.log_prob_and_entropy(states, actions)

        # ---- 손실 정의 ----
        # (1) Actor: -E[logπ(a|s) * A] - β * Entropy
        actor_loss = -(logp * advantages.detach()).mean() - entropy_beta * entropy.mean()
        # (2) Critic: MSE(R, V)
        critic_loss = 0.5 * (returns.detach() - values).pow(2).mean()

        # ---- 글로벌 네트워크 갱신 ----
        self.global_agent.apply_grads(actor_loss, critic_loss, self.actor.parameters(), self.critic.parameters())


# ===============================================================
# 🧠 A3C Agent 클래스 (스레드 실행 제어)
# ===============================================================
class A3CAgent:
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.global_agent = GlobalA3C(env_name)
        self.num_workers = cpu_count()  # 사용 가능한 CPU 코어 수

    def train(self):
        """모든 워커 스레드 시작"""
        print(f"Training on {self.num_workers} cores (threads)")
        workers = [Worker(i, self.env_name, self.global_agent) for i in range(self.num_workers)]
        for w in workers:
            w.start()
        for w in workers:
            w.join()



@torch.no_grad()
def evaluate_and_record(env_name: str,
                        actor_path: str,
                        critic_path: str,
                        out_dir: str = "videos",
                        episodes: int = 3,
                        max_steps: int = 2000):
    """
    저장된 모델을 로드하여 평가하고, 각 에피소드를 영상으로 저장합니다.
    """
    # 비디오 폴더 준비
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 비디오 녹화 가능한 env 생성 (rgb_array 필수)
    base_env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(base_env, video_folder=out_dir, episode_trigger=lambda e: True)

    # 액션 범위/상태 크기 확인을 위해 임시 환경에서 정보 얻기
    tmp = gym.make(env_name)
    state_size = tmp.observation_space.shape[0]
    action_size = tmp.action_space.shape[0]
    action_bound = float(tmp.action_space.high[0])
    tmp.close()

    # 동일 아키텍처 모델 생성 후 로드
    actor = Actor(state_size, action_size, action_bound).to(device)
    critic = Critic(state_size).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    critic.load_state_dict(torch.load(critic_path, map_location=device))
    actor.eval(); critic.eval()

    def select_action_mu(state_np: np.ndarray) -> np.ndarray:
        """평가용: 평균 행동(μ) 사용 (deterministic)"""
        s = torch.from_numpy(state_np).to(device).unsqueeze(0).to(torch.float64)
        mu, std = actor(s)
        a = mu.clamp(-action_bound, action_bound)  # 안전 클램프
        return a.squeeze(0).cpu().numpy()

    # 에피소드 루프
    for ep in range(episodes):
        # reset은 gym/gymnasium 호환 처리
        out = env.reset()
        state = out[0] if isinstance(out, tuple) else out
        done = False
        total_r = 0.0

        for t in range(max_steps):
            action = select_action_mu(state)
            # step도 버전별 반환 형태를 감안해 안전 처리
            out = env.step(action)
            if len(out) == 5:
                next_state, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                next_state, reward, done, info = out
            total_r += reward
            state = next_state
            if done:
                break

        print(f"[EVAL] Episode {ep+1}/{episodes} return={total_r:.2f}")

    env.close()
# ===============================================================
# 🚀 실행 엔트리포인트
# ===============================================================
if __name__ == "__main__":
    env_name = "Pendulum-v1"  # 연속 제어 환경
    agent = A3CAgent(env_name)

    # --- (A) 학습 ---
    DO_TRAIN = False
    if DO_TRAIN:
        agent.train()
        # 모델 저장 (경로는 자유롭게 변경)
        actor_path = "a3c_actor.pth"
        critic_path = "a3c_critic.pth"
        agent.global_agent.save(actor_path, critic_path)
        print(f"[SAVE] Saved to {actor_path}, {critic_path}")

    # --- (B) 로드 후 비디오 평가/저장 ---
    DO_RECORD = True
    if DO_RECORD:
        # 학습 직후가 아니라면, 새 인스턴스로 불러와도 됩니다:
        # loader = GlobalA3C(env_name)
        # loader.load("a3c_actor.pth", "a3c_critic.pth")

        evaluate_and_record(
            env_name=env_name,
            actor_path="Video/continuous/a3c_actor.pth",
            critic_path="Video/continuous/a3c_critic.pth",
            out_dir="videos",  # 비디오 저장 폴더
            episodes=3,  # 저장할 에피소드 수
            max_steps=2000  # 에피소드 최대 스텝
        )
        print("[VIDEO] Saved evaluation videos under ./videos")