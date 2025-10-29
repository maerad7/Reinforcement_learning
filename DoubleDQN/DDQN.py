# ===== PyTorch DQN with Save/Load =====
import random
import numpy as np
from collections import deque

# Gymnasium이 설치돼 있으면 import, 아니면 gym 사용
try:
    import gymnasium as gym
    NEW_GYM = True
except ImportError:
    import gym
    NEW_GYM = False

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ------------------ Gym 호환 유틸 ------------------
def safe_reset(env, seed=None):
    # Gymnasium: (obs, info) 반환 → obs만 추출
    if NEW_GYM:
        obs, info = env.reset(seed=seed)
        return obs
    else:  # 구버전 gym: obs만 반환
        return env.reset()

def safe_step(env, action):
    out = env.step(action)
    if NEW_GYM:
        # Gymnasium: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
        return obs, reward, done
    else:
        # 구버전 gym: (obs, reward, done, info)
        obs, reward, done, info = out
        return obs, reward, done


# ------------------ Q-Network ------------------
class Network(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        # 입력 → 은닉1
        self.fc1 = nn.Linear(state_size, hidden_size)
        # 은닉1 → 은닉2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 은닉2 → 출력(Q값, 행동 차원만큼)
        self.out = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        # ReLU 활성화 적용
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 각 행동별 Q값 반환
        return self.out(x)


# ------------------ DQN Agent ------------------
class DQNAgent:
    def __init__(self, env, batch_size, target_update,
                 hidden_size=128, lr=1e-3, gamma=0.99,
                 memory_size=2000, device=None):
        self.env = env
        # 상태 공간 차원 수
        self.state_size = int(np.prod(env.observation_space.shape))
        # 행동 공간 차원 수
        self.action_size = env.action_space.n

        # 학습 하이퍼파라미터
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma

        # 디바이스 선택 (GPU > CPU)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # policy_net: 학습 대상 네트워크
        self.policy_net = Network(self.state_size, self.action_size, hidden_size).to(device)
        # target_net: 타깃 네트워크 (주기적으로만 동기화)
        self.target_net = Network(self.state_size, self.action_size, hidden_size).to(device)
        self._target_hard_update()  # 초기 동기화

        # Adam 옵티마이저
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # 리플레이 메모리
        self.memory = deque(maxlen=memory_size)
        self.update_cnt = 0  # 학습 스텝 카운터

    # ε-greedy 행동 선택
    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            # 탐험: 랜덤 행동
            return np.random.randint(self.action_size)
        # 탐욕: argmax Q(s,a)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)[0]
            return int(torch.argmax(q).item())

    # 메모리에 transition 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 타깃 네트워크 ← 정책 네트워크 동기화
    def _target_hard_update(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    # Q 업데이트(학습 스텝)
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        # 리플레이 메모리에서 미니배치 샘플링
        batch = random.sample(self.memory, self.batch_size)
        states      = np.vstack([b[0] for b in batch]).astype(np.float32)
        actions     = np.array([b[1] for b in batch], dtype=np.int64)
        rewards     = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.vstack([b[3] for b in batch]).astype(np.float32)
        dones       = np.array([b[4] for b in batch], dtype=np.float32)

        # 텐서 변환
        states_t      = torch.tensor(states, device=self.device)
        actions_t     = torch.tensor(actions, device=self.device)
        rewards_t     = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t       = torch.tensor(dones, device=self.device)

        # 현재 Q(s,a) 추출
        q_all = self.policy_net(states_t)
        q_sa  = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN 타깃 계산
        with torch.no_grad():
            # online net으로 argmax 행동 선택
            next_q_online = self.policy_net(next_states_t)
            next_actions  = torch.argmax(next_q_online, dim=1)
            # target net에서 해당 행동 Q값 평가
            next_q_target = self.target_net(next_states_t)
            next_q_max    = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # r + γQ(s',a*) (종단 상태면 부트스트랩 X)
            targets = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        # Huber Loss 계산 (TD 오차 최소화)
        loss = F.smooth_l1_loss(q_sa, targets)

        # 역전파 & 가중치 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # 일정 주기마다 타깃 네트워크 동기화
        self.update_cnt += 1
        if self.update_cnt % self.target_update == 0:
            self._target_hard_update()


# ------------------ 메인 ------------------
if __name__ == "__main__":
    env_name = "CartPole-v0"
    # 학습용 환경 (렌더링 켜둔 버전)
    train_env = gym.make(env_name, render_mode="human")

    target_update = 100
    hidden_size   = 128
    max_episodes  = 600
    batch_size    = 64

    # ε-greedy 파라미터
    epsilon      = 1.0
    max_epsilon  = 1.0
    min_epsilon  = 0.01
    decay_rate   = 0.005

    # 에이전트 초기화
    agent = DQNAgent(
        env=train_env,
        batch_size=batch_size,
        target_update=target_update,
        hidden_size=hidden_size,
        lr=1e-3,
        gamma=0.99,
        memory_size=2000,
    )

    scores = []

    # ---- 학습 루프 ----
    for episode in range(max_episodes):
        state = safe_reset(agent.env)
        done = False
        ep_ret = 0
        while not done:
            # ε-greedy로 행동 선택
            action = agent.get_action(state, epsilon)
            # 환경 한 스텝
            next_state, reward, done = safe_step(agent.env, action)
            # transition 저장
            agent.append_sample(state, action, reward, next_state, done)
            # 상태 갱신
            state = next_state
            ep_ret += reward
            # 일정 이상 메모리 쌓이면 학습
            if len(agent.memory) >= agent.batch_size:
                agent.train_step()
        scores.append(ep_ret)
        print(f"Episode {episode+1}: reward={ep_ret:.1f}, eps={epsilon:.3f}")

        # ε 감소
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * (episode+1))

    print("✅ Training complete")

    # ---- 모델 저장 ----
    torch.save(agent.policy_net.state_dict(), "dqn_cartpole.pth")
    print("✅ 모델 저장 완료: dqn_cartpole.pth")

    # ---- 모델 로드 ----
    loaded_net = Network(agent.state_size, agent.action_size, hidden_size).to(agent.device)
    loaded_net.load_state_dict(torch.load("dqn_cartpole.pth"))
    loaded_net.eval()
    print("✅ 모델 로드 완료")

    # ---- 테스트 (렌더링) ----
    if NEW_GYM:
        test_env = gym.make(env_name, render_mode="human")
    else:
        test_env = gym.make(env_name)

    for i in range(5):
        state = safe_reset(test_env)
        done = False
        ep_ret = 0
        while not done:
            if not NEW_GYM:
                test_env.render()  # 구버전 gym일 경우 직접 render()
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
                # 로드한 네트워크로 행동 선택
                action = int(torch.argmax(loaded_net(s)[0]).item())
            state, reward, done = safe_step(test_env, action)
            ep_ret += reward
        print(f"[TEST {i+1}] return={ep_ret:.1f}")

    test_env.close()
