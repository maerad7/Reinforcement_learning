import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.ao.quantization import no_observer_set  # (사용하지 않음) 필요 없으면 지워도 됨

# CartPole 환경 생성
env = gym.make("CartPole-v1")

# --- Matplotlib 설정 (노트북/인터랙티브 여부 확인) ---
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()  # interactive 모드 on (플롯을 매 스텝 갱신)

# --- 학습 디바이스 선택(GPU/Metal/MPS/CPU) ---
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# ----- 리플레이 버퍼에 담길 Transition 구조 정의 -----
# state, action, next_state, reward 4개 필드를 갖는 튜플 클래스
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# ----- 리플레이 메모리 구현 -----
class ReplayMemory(object):
    def __init__(self, capacity):
        # deque를 고정 크기(capacity)로 사용(가득 차면 가장 오래된 샘플부터 자동 삭제)
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # (state, action, next_state, reward)를 Transition으로 감싸 append
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # 무작위 미니배치 샘플링
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # 현재 버퍼에 담긴 transition 수
        return len(self.memory)

# ----- DQN 모델 정의(간단한 MLP 2개 은닉층) -----
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)  # 최종 출력 차원 = 행동 수 (각 행동의 Q값)

    def forward(self, x):
        x = F.relu(self.layer1(x))  # 1층 ReLU
        x = F.relu(self.layer2(x))  # 2층 ReLU
        return self.layer3(x)       # 선형 출력(Q값 벡터)

# ----- 하이퍼파라미터 -----
BATCH_SIZE = 128      # 미니배치 크기
GAMMA = 0.99          # 할인율
EPS_START = 0.9       # ε 시작값(탐험 확률)
EPS_END = 0.01        # ε 최저값
EPS_DECAY = 2500      # ε 지수감쇠 스케일(클수록 천천히 감소)
TAU = 0.005           # 소프트 타깃 업데이트 계수
LR = 3e-4             # 학습률

# ----- 환경 차원/행동 수 파악 및 네트워크 초기화 -----
n_actions = env.action_space.n
state, info = env.reset()              # 초기 리셋
n_observations = len(state)            # 관측 벡터 길이(=입력 차원)

policy_net = DQN(n_observations, n_actions).to(device)  # 현재 Q-네트워크
target_net = DQN(n_observations, n_actions).to(device)  # 타깃 Q-네트워크
target_net.load_state_dict(policy_net.state_dict())     # 초기에는 동일하게 설정

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)           # 리플레이 버퍼(최대 10k transition)

steps_done = 0  # ε 스케줄 계산에 사용할 스텝 카운터

# ----- ε-greedy 정책으로 행동 선택 -----
def select_action(state):
    global steps_done

    # [0,1) 난수로 탐험/탐욕 결정
    sample = random.random()

    # ε 지수감쇠 스케줄: 초기엔 큼 → 점차 EPS_END로 수렴
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        # 탐욕(Exploitation): Q값이 최대인 행동 선택
        with torch.no_grad():
            # policy_net(state) → (1, n_actions) Q벡터
            # .max(1).indices → argmax 행동 인덱스 → (1,1)로 뷰 변환(배치 차원 유지)
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # 탐험(Exploration): action_space에서 무작위 샘플
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# ----- 에피소드 길이 기록 및 러닝 평균을 그리는 유틸 -----
episode_durations = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # 최근 100 에피소드 이동평균
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# ----- DQN 업데이트(리플레이 배치 샘플 → 손실 역전파) -----
def optimize_model():
    # 버퍼가 아직 작으면 건너뜀
    if len(memory) < BATCH_SIZE:
        return

    # 미니배치 샘플링 및 열 방향 전치
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # next_state가 None이 아닌(=terminal 아닌) 샘플 마스크
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool
    )

    # all-terminal 배치면 torch.cat이 실패하므로 분기
    if non_final_mask.any():
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    else:
        non_final_next_states = None

    # (B, obs), (B,1), (B,), (B,)
    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action).long()
    reward_batch = torch.cat(batch.reward).float()

    # 현재 Q(s,a): policy_net로 모든 행동 Q를 구한 뒤,
    # gather로 실제 수행한 action의 Q만 (행별 인덱스) 추출 → (B,)
    state_action_values = policy_net(state_batch).gather(1, action_batch).squeeze(1)

    # 다음 상태의 최댓값 Q_target(s',·) (terminal이면 0)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():  # 타깃 계산은 그래프 추적 X
        if non_final_next_states is not None:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # DQN 타깃: r + γ * max_a Q_target(s', a)
    expected_state_action_values = reward_batch + GAMMA * next_state_values

    # Huber Loss(SmoothL1Loss)로 TD오차 최소화
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # 경사 초기화 → 역전파 → 그라드 클리핑 → 옵티마이저 스텝
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# ----- 에피소드 수(디바이스에 따라 다르게) -----
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

# ----- 메인 학습 루프 -----
for i_episode in range(num_episodes):
    # 에피소드 시작: 환경 리셋
    obs, info = env.reset()
    # 관측을 (1, obs_dim) 텐서로 변환
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    for t in count():  # 무한 루프 → 내부에서 done이면 break
        # ε-greedy로 행동 선택
        action = select_action(state)
        # env.render()
        # 환경 한 스텝 전진
        obs, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated  # 둘 중 하나라도 True면 에피소드 종료

        # 종료면 다음 상태 없음(None), 아니면 텐서 변환
        if done:
            next_state = None
        else:
            next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # 리플레이 버퍼에 transition 저장
        memory.push(state, action, next_state, reward)

        # 다음 반복을 위한 현재 상태 갱신
        state = next_state

        # 1 스텝의 미니배치 학습 수행
        optimize_model()

        # --- 소프트 타깃 업데이트(Polyak averaging) ---
        with torch.no_grad():
            for p, tp in zip(policy_net.parameters(), target_net.parameters()):
                tp.data.copy_(TAU * p.data + (1.0 - TAU) * tp.data)

        # 에피소드 종료 처리(기록/플롯/탈출)
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

torch.save(policy_net.state_dict(), "dqn_cartpole2.pth")
print("✅ 모델 저장 완료: dqn_cartpole.pth")

# 네트워크 초기화 (동일한 구조여야 함)
loaded_net = DQN(n_observations, n_actions).to(device)
loaded_net.load_state_dict(torch.load("dqn_cartpole2.pth"))
loaded_net.eval()   # 평가 모드 (Dropout/BatchNorm 비활성화)
print("✅ 모델 로드 완료")


print('Complete')
plot_durations(show_result=True)
# 렌더링 환경 생성
env = gym.make("CartPole-v1", render_mode="human")

for i in range(5):  # 5 에피소드 실행
    obs, info = env.reset()
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    while not done:
        env.render()
        with torch.no_grad():
            action = loaded_net(state).max(1).indices.view(1, 1)
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        if not done:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

env.close()



plt.ioff()
plt.show()
