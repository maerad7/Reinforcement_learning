# ========================== DQN with PolicyNet + TargetNet (PyTorch) ==========================
import random                 # 리플레이 버퍼 샘플링 등에 사용할 표준 랜덤
import gym                    # 강화학습 환경 라이브러리 (CartPole 등)
import numpy as np            # 수치 계산용 라이브러리
from collections import deque # 고정 길이 큐(리플레이 버퍼) 구현을 위해 사용

import torch                  # PyTorch 메인 패키지
import torch.nn as nn         # 신경망 레이어 모듈
import torch.optim as optim   # 최적화 알고리즘(Adam 등)
import torch.nn.functional as F  # 활성함수/손실 함수 등 유틸

# -------------------------- 하이퍼파라미터 --------------------------
hidden_size   = 128           # 은닉층 뉴런 수
max_episodes  = 200           # 학습 에피소드 수
batch_size    = 64            # 미니배치 크기
gamma         = 0.99          # 할인율 (미래 보상 가중치)
lr            = 1e-3          # 학습률
buffer_size   = 2000          # 리플레이 버퍼 최대 크기
warmup_steps  = 1000          # 학습 시작 전 최소 저장해야 할 샘플 수
target_sync_every = 10        # target net 파라미터 동기화 주기(에피소드 단위)

# 입실론 탐험 파라미터(ε-greedy)
epsilon       = 1.0           # 초기 ε
max_epsilon   = 1.0           # 최대 ε
min_epsilon   = 0.01          # 최소 ε
decay_rate    = 0.005         # ε 지수 감쇠율(에피소드 단위)

# 디바이스 설정 (GPU가 있으면 CUDA 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- Gym/Gymnasium 호환 유틸 --------------------------
def safe_reset(env, seed=None):
    """환경 초기화 시 Gym/Gymnasium의 반환 차이를 흡수"""
    try:
        # gymnasium: reset(seed=...) 지원, (obs, info) 반환
        out = env.reset(seed=seed) if seed is not None else env.reset()
        if isinstance(out, tuple):       # (obs, info) 형태면 obs만 사용
            obs, _ = out
        else:                             # 구버전 gym: obs만 반환
            obs = out
        return obs
    except TypeError:
        # 일부 구버전 호환
        out = env.reset()
        if isinstance(out, tuple):
            obs, _ = out
        else:
            obs = out
        return obs

def safe_step(env, action):
    """환경 step 시 Gym/Gymnasium의 반환 차이를 흡수"""
    out = env.step(action)
    if len(out) == 5:                     # gymnasium: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, _ = out
        done = terminated or truncated    # 종료 플래그 통합
    else:                                 # 구버전 gym: (obs, reward, done, info)
        obs, reward, done, _ = out
    return obs, reward, done

# -------------------------- Q-네트워크 정의 --------------------------
class Network(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()                                      # 부모 생성자 호출
        self.fc1 = nn.Linear(state_size, hidden_size)           # 입력 -> 은닉1
        self.fc2 = nn.Linear(hidden_size, hidden_size)          # 은닉1 -> 은닉2
        self.out = nn.Linear(hidden_size, action_size)          # 은닉2 -> Q값(행동 수)

    def forward(self, x):
        x = F.relu(self.fc1(x))                                 # ReLU 활성화
        x = F.relu(self.fc2(x))                                 # ReLU 활성화
        q = self.out(x)                                         # 각 행동에 대한 Q(s, a)
        return q

# -------------------------- DQN 에이전트 --------------------------
class DQNAgent:
    def __init__(self, env: gym.Env):
        self.env = env                                          # 환경 핸들 저장
        self.state_size  = int(np.prod(env.observation_space.shape))  # 상태 차원(1D로 펼침)
        self.action_size = env.action_space.n                   # 행동 개수

        # policy(online) 네트워크: 학습 대상
        self.policy_net = Network(self.state_size, self.action_size).to(device)
        # target 네트워크: 일정 주기로 policy_net 파라미터를 복사받아 타깃 계산에만 사용
        self.target_net = Network(self.state_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # 초기 동기화
        self.target_net.eval()                                   # 학습(gradient) 비활성

        # 최적화 알고리즘(Adam)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 리플레이 버퍼(순환 큐)
        self.memory = deque(maxlen=buffer_size)

        # 스텝/에피소드 카운터
        self.total_steps = 0

    def get_action(self, state: np.ndarray, epsilon: float) -> int:
        """ε-greedy 행동 선택: 확률 ε로 랜덤, 아니면 argmax Q"""
        if np.random.rand() <= epsilon:                          # 탐험
            return np.random.randint(self.action_size)
        with torch.no_grad():                                    # 추론 모드(그래프 비저장)
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # [1, state_size]
            q = self.policy_net(s)[0]                            # Q(s, :)
            return int(torch.argmax(q).item())                   # argmax로 행동 선택

    def append(self, state, action, reward, next_state, done):
        """경험을 리플레이 버퍼에 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """리플레이 버퍼에서 미니배치 샘플링 후, 1 스텝 학습 수행"""
        if len(self.memory) < batch_size:                        # 미니배치 불가 시 스킵
            return

        # 미니배치 무작위 샘플링
        batch = random.sample(self.memory, batch_size)

        # 배치 구성 성분을 각 배열로 분리
        states      = np.vstack([b[0] for b in batch]).astype(np.float32)  # [B, state_size]
        actions     = np.array([b[1] for b in batch], dtype=np.int64)      # [B]
        rewards     = np.array([b[2] for b in batch], dtype=np.float32)    # [B]
        next_states = np.vstack([b[3] for b in batch]).astype(np.float32)  # [B, state_size]
        dones       = np.array([b[4] for b in batch], dtype=np.float32)    # [B]

        # 넘파이 -> 토치 텐서 변환
        states_t      = torch.tensor(states, device=device)                 # [B, S]
        actions_t     = torch.tensor(actions, device=device)                # [B]
        rewards_t     = torch.tensor(rewards, device=device)                # [B]
        next_states_t = torch.tensor(next_states, device=device)            # [B, S]
        dones_t       = torch.tensor(dones, device=device)                  # [B]

        # ----------------- 현재 Q(s,a) 계산 (policy_net) -----------------
        q_all = self.policy_net(states_t)                                    # [B, A]
        q_sa  = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)          # 선택한 a의 Q(s,a)

        # ----------------- 타깃 값 계산 (target_net) -----------------
        with torch.no_grad():                                                # 타깃은 미분 제외
            next_q_all = self.target_net(next_states_t)                      # Q_target(s', :)
            next_q_max = torch.max(next_q_all, dim=1).values                 # max_a' Q_target(s', a')
            target = rewards_t + gamma * next_q_max * (1.0 - dones_t)        # done이면 부트스트랩 항 제거

        # ----------------- 손실 계산 및 역전파 -----------------
        loss = F.smooth_l1_loss(q_sa, target)                                # Huber Loss(안정적)

        self.optimizer.zero_grad()                                           # 기울기 초기화
        loss.backward()                                                      # 역전파
        self.optimizer.step()                                                # 가중치 업데이트

    def sync_target(self):
        """policy_net → target_net 파라미터 하드 업데이트(복사)"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

# -------------------------- 메인 학습 루프 --------------------------
if __name__ == "__main__":
    env_name = "CartPole-v1"                                # 환경 이름
    env = gym.make(env_name)                                # 환경 생성

    agent = DQNAgent(env)                                   # 에이전트 생성

    scores = []                                             # 에피소드 총 보상 기록
    epsilon_curr = epsilon                                  # 현재 ε

    for episode in range(1, max_episodes + 1):              # 1 ~ max_episodes
        state = safe_reset(env)                              # 환경 초기화(관찰 obs)
        done = False                                         # 종료 플래그
        ep_reward = 0.0                                      # 에피소드 보상 누적

        while not done:                                      # 종료까지 반복
            action = agent.get_action(state, epsilon_curr)   # ε-greedy 행동 선택
            next_state, reward, done = safe_step(env, action)# 환경 반응 수집

            agent.append(state, action, reward, next_state, done) # 경험 버퍼 저장
            agent.total_steps += 1                           # 전체 스텝 카운트 +1

            # 버퍼가 어느 정도 쌓인 후부터 학습 시작(초기 과도한 과적합/불안정 방지)
            if len(agent.memory) >= max(batch_size, warmup_steps):
                agent.train_step()                           # 1 스텝 학습 수행

            state = next_state                               # 상태 갱신
            ep_reward += reward                              # 보상 누적

        scores.append(ep_reward)                             # 에피소드 보상 기록
        print(f"[Ep {episode:03d}] reward={ep_reward:.1f}  eps={epsilon_curr:.3f}  buf={len(agent.memory)}")

        # 일정 주기마다 target 네트워크 동기화(하드 업데이트)
        if episode % target_sync_every == 0:
            agent.sync_target()                              # policy → target 파라미터 복사

        # ε 지수 감쇠(탐험 비율 감소)
        epsilon_curr = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    env.close()                                              # 환경 자원 해제
# =============================================================================================
