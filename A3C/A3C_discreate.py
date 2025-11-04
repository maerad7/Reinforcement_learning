# ================================================================
# 🧠 PyTorch A3C (CartPole-v1) - Best Score Auto Save (Train + Eval)
#   - dtype=float32 통일
#   - 멀티스레드 락 동기화
#   - 훈련/평가 최고 점수 시 자동 저장
#   - 수동 저장/로드 + 비디오 녹화(RecordVideo)
# ================================================================

import os, time
from typing import Tuple
from threading import Thread, Lock
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

# ================================================================
# Global 설정
# ================================================================
torch.set_default_dtype(torch.float32)                # 모든 tensor 기본 dtype을 float32로 고정해 혼용 문제 방지
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # CUDA 사용 가능하면 GPU, 아니면 CPU

actor_lr = 5e-4                                       # Actor(정책) 네트워크 학습률
critic_lr = 1e-3                                      # Critic(가치) 네트워크 학습률
gamma = 0.99                                          # 보상 할인율 γ
hidden_size = 128                                     # MLP 은닉차원
update_interval = 50                                  # 로컬 버퍼 길이: 이 길이에 도달하거나 에피소드 종료 시 글로벌 업데이트
max_episodes = 500                                    # 총 학습 에피소드 수
entropy_beta = 1e-2                                   # 정책 엔트로피 보너스 계수(탐험 유도)
EVAL_EVERY = 20                                       # N 에피소드마다 평가 수행

GLOBAL_EP = 0                                         # 현재까지 완료된 에피소드 카운터(전역)
GLOBAL_EP_LOCK = Lock()                               # GLOBAL_EP 증가를 원자적으로 보호
PRINT_LOCK = Lock()                                   # 콘솔 출력이 스레드 간 섞이지 않도록 보호
UPDATE_LOCK = Lock()                                  # 글로벌 네트워크 파라미터 업데이트 임계영역 보호

# ================================================================
# Gym Helper 함수
# ================================================================
def reset_env(env):
    out = env.reset()                                 # Gymnasium은 (obs, info) 튜플 반환
    return out[0] if isinstance(out, tuple) else out  # 관측치(obs)만 사용

def step_env(env, action) -> Tuple[np.ndarray, float, bool, dict]:
    out = env.step(action)                            # Gymnasium은 (obs, reward, terminated, truncated, info)
    if len(out) == 5:                                 # 최신 Gymnasium 포맷
        ns, r, term, trunc, info = out
        return ns, float(r), bool(term or trunc), info  # term 또는 trunc가 True면 done
    else:                                             # (구버전 호환) (obs, reward, done, info)
        ns, r, done, info = out
        return ns, float(r), bool(done), info

# ================================================================
# Actor 정의
# ================================================================
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)      # 입력: 상태 벡터 → 은닉
        self.fc2 = nn.Linear(hidden_size, hidden_size)     # 은닉 → 은닉
        self.policy_head = nn.Linear(hidden_size, action_size)  # 은닉 → 각 행동의 로짓
        self.softmax = nn.Softmax(dim=-1)                  # 로짓 → 확률
        self.opt = optim.Adam(self.parameters(), lr=actor_lr)   # 정책 최적화기
        self.entropy_beta = entropy_beta                    # 엔트로피 가중치 저장

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))                         # ReLU 비선형성
        x = torch.relu(self.fc2(x))
        logits = self.policy_head(x)                        # 행동별 로짓
        return self.softmax(logits)                         # 정규화된 행동 확률 π(a|s)

    def compute_loss(self, probs, actions, advantages):
        dist = torch.distributions.Categorical(probs)       # 이산 행동 공간용 범주분포
        log_probs = dist.log_prob(actions.squeeze(-1))      # 선택한 행동의 log π(a|s)
        entropy = dist.entropy().mean()                     # 평균 엔트로피(탐험성 척도)
        policy_loss = -(log_probs * advantages.squeeze(-1)).mean()  # A3C 정책손실 = -E[logπ * A]
        return policy_loss - self.entropy_beta * entropy    # 엔트로피 보너스로 탐험성 유지

    def train_step(self, states, actions, advantages):
        probs = self.forward(states)                        # 미니배치 상태 → 행동확률
        loss = self.compute_loss(probs, actions, advantages)# 정책 손실 계산
        self.opt.zero_grad()                                # 그래디언트 초기화
        loss.backward()                                     # 역전파
        self.opt.step()                                     # 파라미터 갱신
        return float(loss.item())

# ================================================================
# Critic 정의
# ================================================================
class Critic(nn.Module):
    def __init__(self, state_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)       # 상태 → 은닉
        self.fc2 = nn.Linear(hidden_size, hidden_size)      # 은닉 → 은닉
        self.v_head = nn.Linear(hidden_size, 1)             # 은닉 → 스칼라 V(s)
        self.opt = optim.Adam(self.parameters(), lr=critic_lr)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.v_head(x)                               # 상태가치 V(s)

    def train_step(self, states, td_targets):
        values = self.forward(states)                       # V(s) 예측
        loss = torch.mean((td_targets - values) ** 2)       # TD-타깃과의 MSE
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

# ================================================================
# A3CAgent (글로벌 네트워크 + Best 저장 로직)
# ================================================================
class A3CAgent:
    def __init__(self, env_name: str, gamma: float):
        self.env_name = env_name
        self.gamma = gamma

        # 환경 정보 확인(상태/행동 차원 파악용 더미 env)
        tmp = gym.make(env_name)
        self.state_size = tmp.observation_space.shape[0]    # CartPole: 4차원
        self.action_size = tmp.action_space.n               # CartPole: 2개(좌/우)
        tmp.close()

        # 전역 공유 네트워크(Actor, Critic)
        self.global_actor = Actor(self.state_size, self.action_size).to(device).float()
        self.global_critic = Critic(self.state_size).to(device).float()

        # 워커 수: CPU 코어 수 기반(필요시 상한 제한 가능)
        self.num_workers = cpu_count()

        # 최고점 저장 관리 변수
        self.best_score = float("-inf")                     # 현재까지 최고 평균/리턴
        self.best_lock = Lock()                             # 동시 접근 보호
        self.best_actor_path = "a3c_cartpole_actor_best.pth"
        self.best_critic_path = "a3c_cartpole_critic_best.pth"

    # -----------------------------
    # 통합 베스트 저장 함수
    # -----------------------------
    def save_best(self, score: float, tag: str = "train"):
        """훈련/평가에서 얻은 score가 최고치면 글로벌 네트워크 가중치 저장"""
        with self.best_lock:                                # 다중 스레드 보호
            if score > self.best_score:                     # 최고 기록 갱신 시
                self.best_score = score
                torch.save(self.global_actor.state_dict(), self.best_actor_path)
                torch.save(self.global_critic.state_dict(), self.best_critic_path)
                with PRINT_LOCK:
                    print(f"[BEST-{tag.upper()}] New best {score:.2f} saved "
                          f"({self.best_actor_path}, {self.best_critic_path})")

    # -----------------------------
    # 학습 실행 (워커 병렬)
    # -----------------------------
    def train(self):
        print(f"Training on {self.num_workers} cores")      # 사용 코어 수 안내
        # 워커 스레드 생성
        workers = [Worker(i, self.env_name, self.gamma, self) for i in range(self.num_workers)]
        for w in workers: w.start()                         # 각 워커 시작(daemon=True)
        for w in workers: w.join()                          # 모든 워커 종료까지 대기

    def save(self, actor_path, critic_path):
        """마지막 모델 수동 저장"""
        torch.save(self.global_actor.state_dict(), actor_path)
        torch.save(self.global_critic.state_dict(), critic_path)

    def load(self, actor_path, critic_path, map_location=None):
        """저장된 모델 로드 + 평가모드 전환"""
        map_location = map_location or device
        self.global_actor.load_state_dict(torch.load(actor_path, map_location=map_location))
        self.global_critic.load_state_dict(torch.load(critic_path, map_location=map_location))
        self.global_actor.eval()
        self.global_critic.eval()

    # -----------------------------
    # 정책 평가 (평균 리턴 계산)
    # -----------------------------
    @torch.no_grad()
    def evaluate_policy(self, episodes=5, max_steps=500, seed=42):
        env = gym.make(self.env_name, max_episode_steps=max_steps)  # 평가용 env
        actor = Actor(self.state_size, self.action_size).to(device).float()
        actor.load_state_dict(self.global_actor.state_dict())       # 글로벌 정책 스냅샷
        actor.eval()

        def greedy_action(s_np):
            """가장 확률이 높은 행동 선택(탐욕)"""
            s = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
            return int(torch.argmax(actor(s), dim=-1).item())

        returns = []                                               # 에피소드별 총보상 저장
        for ep in range(episodes):
            s, _ = env.reset(seed=seed + ep)                       # 시드 고정으로 재현성 확보
            done, ep_ret = False, 0.0
            for _ in range(max_steps):
                a = greedy_action(s)                               # 탐욕 실행
                s, r, term, trunc, _ = env.step(a)
                done = term or trunc
                ep_ret += float(r)
                if done: break
            returns.append(ep_ret)
        env.close()

        avg = float(np.mean(returns))                              # 평균 리턴
        with PRINT_LOCK:
            print(f"[EVAL] avg_return={avg:.2f}")
        self.save_best(avg, tag="eval")                            # 평가 기준 베스트 저장
        return avg

# ================================================================
# Worker 클래스 (로컬 네트워크 + 글로벌 갱신)
# ================================================================
class Worker(Thread):
    def __init__(self, wid, env_name, gamma, agent_ref):
        super().__init__(daemon=True)                              # 메인 종료 시 자동 종료
        self.wid = wid
        self.name = f"w{wid}"                                      # 로그 식별용 이름
        self.env = gym.make(env_name)                              # 각 워커 독립 환경
        self.gamma = gamma
        self.agent = agent_ref

        # 글로벌 네트워크 핸들
        self.global_actor = agent_ref.global_actor
        self.global_critic = agent_ref.global_critic

        # 로컬 네트워크 생성(초기엔 글로벌 파라미터로 동기화)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.actor = Actor(self.state_size, self.action_size).to(device).float()
        self.critic = Critic(self.state_size).to(device).float()
        self.sync_with_global()                                     # 초기 동기화

    def sync_with_global(self):
        """글로벌 → 로컬 파라미터 복사(동기화)"""
        self.actor.load_state_dict(self.global_actor.state_dict())
        self.critic.load_state_dict(self.global_critic.state_dict())

    def get_action(self, state_np):
        """현재 정책에 따라 확률적으로 행동 샘플"""
        s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        probs = self.actor(s)                                       # π(a|s)
        dist = torch.distributions.Categorical(probs)
        return int(dist.sample().item())                            # 샘플링으로 탐험 반영

    def n_step_td_target(self, rewards_np, next_v, done):
        """
        n-step TD 타깃 계산.
        - rewards_np: shape (n, 1), 경로에서 모은 보상들
        - next_v: 마지막 다음 상태의 V(s_{t+n}), 종료면 0
        - done: 종료 여부
        """
        td_targets = np.zeros_like(rewards_np, dtype=np.float32)
        R_to_go = 0.0 if done else float(next_v)                    # 종료면 bootstrap 없음
        for k in reversed(range(len(rewards_np))):                  # 뒤에서부터 누적 할인합
            R_to_go = float(rewards_np[k, 0]) + self.gamma * R_to_go
            td_targets[k, 0] = R_to_go
        return td_targets

    def run(self):
        """워커의 메인 루프: 환경 실행 → 로컬 버퍼 적재 → 주기적 글로벌 업데이트"""
        global GLOBAL_EP
        while True:
            # 에피소드 인덱스 읽기/종료 검사(임계영역)
            with GLOBAL_EP_LOCK:
                if GLOBAL_EP >= max_episodes:
                    break
                ep_idx = GLOBAL_EP + 1                              # (필요시 로깅용)

            state = reset_env(self.env)
            done, ep_return = False, 0.0
            states, actions, rewards = [], [], []                   # 롤아웃 버퍼

            while not done:
                action = self.get_action(state)                     # 정책에 따른 행동 선택
                next_state, reward, done, _ = step_env(self.env, action)
                ep_return += reward
                states.append(state)                                # s_t
                actions.append([action])                            # a_t (열벡터 형태)
                rewards.append([reward])                            # r_{t+1} (열벡터)
                state = next_state

                # 배치가 차거나 종료되면 글로벌 업데이트
                if len(states) >= update_interval or done:
                    states_t  = torch.tensor(np.vstack(states), dtype=torch.float32, device=device)
                    actions_t = torch.tensor(np.vstack(actions), dtype=torch.int64, device=device)
                    rewards_np = np.vstack(rewards).astype(np.float32)

                    with torch.no_grad():
                        curr_Vs = self.critic(states_t).detach().cpu().numpy()  # 현재 로컬 Critic의 V(s_t)
                        next_v = 0.0
                        if not done:
                            ns_t = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                            next_v = float(self.critic(ns_t).item())             # bootstrap V(s_{t+n})

                    td_targets_np = self.n_step_td_target(rewards_np, next_v, done)  # n-step 타깃
                    td_targets_t  = torch.tensor(td_targets_np, dtype=torch.float32, device=device)
                    advantages_t  = td_targets_t - torch.tensor(curr_Vs, dtype=torch.float32, device=device)  # A = G_t - V(s)

                    # 글로벌 네트워크 업데이트(임계영역 보호)
                    with UPDATE_LOCK:
                        self.global_actor.train_step(states_t, actions_t, advantages_t)
                        self.global_critic.train_step(states_t, td_targets_t)

                    self.sync_with_global()                            # 업데이트 후 로컬 재동기화
                    states, actions, rewards = [], [], []              # 버퍼 초기화

            # --- 에피소드 종료 시점: 최고 리턴 저장 시도 ---
            self.agent.save_best(ep_return, tag="train")               # 훈련 리턴 기준

            # 전역 에피소드 카운터 증가(임계영역)
            with GLOBAL_EP_LOCK:
                GLOBAL_EP += 1
                ep_no = GLOBAL_EP

            # --- 주기적 평가 (워커 0만 수행해서 중복 방지) ---
            if (ep_no % EVAL_EVERY == 0) and (self.wid == 0):
                with UPDATE_LOCK:                                      # 평가 직전/중 파라미터 고정
                    self.agent.evaluate_policy(episodes=5, max_steps=500, seed=1234)

            # 에피소드 로그 출력(스레드 안전)
            with PRINT_LOCK:
                print(f"{self.name} | EP{ep_no} Return={ep_return:.2f}")

# ================================================================
# 평가 + 비디오 저장
# ================================================================
@torch.no_grad()
def evaluate_and_record_discrete(env_name, actor_path, out_dir="videos",
                                 episodes=5, max_steps=500, seed=42, greedy=True):
    """
    저장된 Actor 가중치로 에피소드 실행하며 비디오 저장.
    - greedy=True: argmax 정책으로 실행(성능 확인용)
    - greedy=False: 확률 샘플링으로 실행(다양성 확인용)
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")                           # 결과 폴더 타임스탬프
    video_dir = os.path.join(out_dir, stamp)
    os.makedirs(video_dir, exist_ok=True)

    # RecordVideo 래퍼: 매 에피소드 비디오 저장(episode_trigger=lambda e: True)
    env = RecordVideo(
        gym.make(env_name, render_mode="rgb_array", max_episode_steps=max_steps),
        video_folder=video_dir, episode_trigger=lambda e: True
    )

    # Actor 구조 생성을 위해 상태/행동 차원 확인
    tmp = gym.make(env_name)
    state_size = tmp.observation_space.shape[0]
    action_size = tmp.action_space.n
    tmp.close()

    # 저장된 가중치 로드
    actor = Actor(state_size, action_size).to(device).float()
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    def select_action(state_np):
        """greedy(Argmax) 또는 stochastic(Categorical.sample) 선택"""
        s = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        probs = actor(s)
        return int(torch.argmax(probs, dim=-1)) if greedy else int(torch.distributions.Categorical(probs).sample().item())

    # 에피소드 실행 + 비디오 기록
    for ep in range(episodes):
        s, _ = env.reset(seed=seed + ep)
        done, ep_ret = False, 0.0
        for t in range(max_steps):
            a = select_action(s)
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_ret += float(r)
            if done: break
        print(f"[EVAL] Ep {ep+1}/{episodes} return={ep_ret:.2f}")

    env.close()
    print(f"[VIDEO] Saved under: {video_dir}")                        # 비디오 저장 경로 출력

# ================================================================
# Main 실행
# ================================================================
if __name__ == "__main__":
    env_name = "CartPole-v1"
    agent = A3CAgent(env_name, gamma)                                 # 글로벌 네트워크/관리자 초기화

    DO_TRAIN = True
    if DO_TRAIN:
        agent.train()                                                 # 병렬 A3C 학습 시작
        agent.save("a3c_actor_last.pth", "a3c_critic_last.pth")       # 마지막 스냅샷 저장
        print(f"[SAVE-LAST] Last models saved.")
        print(f"[BEST] best_score={agent.best_score:.2f}")            # 세션 최고 리턴 기록 표시

    DO_RECORD = True
    if DO_RECORD:
        # 최고 성능 모델(훈련/평가 기준)을 사용해 비디오 녹화
        evaluate_and_record_discrete(env_name, actor_path=agent.best_actor_path,
                                     out_dir="videos", episodes=3, max_steps=500)
