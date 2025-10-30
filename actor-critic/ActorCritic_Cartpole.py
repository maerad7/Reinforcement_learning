import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from pathlib import Path

# ----- Utils -----
def to_tensor(x, device):
    # numpy array → torch tensor 변환 및 device(cpu/gpu) 이동
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.float().to(device)


# ----- Actor / Critic -----
class Actor(nn.Module):
    # 정책 네트워크 (확률분포 π(a|s) 출력)
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()
        # 입력: 상태, 출력: 행동 확률
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_size), nn.Softmax(dim=-1)  # 행동 확률 분포
        )
    def forward(self, x):
        return self.net(x)  # 행동 확률 반환


class CriticV(nn.Module):
    # 가치 네트워크 (상태 가치 V(s) 추정)
    def __init__(self, state_size, hidden_size=32):
        super().__init__()
        # 입력: 상태, 출력: 스칼라 값 V(s)
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x)


# ----- Agent -----
class A2CAgent:
    def __init__(self, env_id="CartPole-v1", gamma=0.99, hidden_size=32,
                 actor_lr=7e-3, critic_lr=7e-3, entropy_coef=0.01,
                 device=None, seed=2000):
        # 디바이스 자동 선택 (CUDA 있으면 GPU, 없으면 CPU)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # 환경 초기화 (Gymnasium, 영상 녹화를 위해 render_mode="rgb_array")
        self.env_id = env_id
        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env.reset(seed=seed)

        # 상태/행동 공간 크기 가져오기
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        # Actor-Critic 네트워크 정의
        self.actor = Actor(self.state_size, self.action_size, hidden_size).to(self.device)
        self.critic = CriticV(self.state_size, hidden_size).to(self.device)

        # 최적화 알고리즘 정의 (Adam)
        self.a_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.c_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        # 상태 → 행동 선택 (확률적 sampling)
        s = to_tensor(state, self.device).unsqueeze(0)   # 배치 차원 추가
        probs = self.actor(s)                            # 행동 확률 얻기
        dist = Categorical(probs)                        # Categorical 분포 생성
        return dist.sample().item()                      # 확률적으로 행동 선택

    def train_step(self, state, action, reward, next_state, done):
        # 학습 1 step: (s,a,r,s')로 Actor와 Critic 업데이트
        s = to_tensor(state, self.device).unsqueeze(0)
        ns = to_tensor(next_state, self.device).unsqueeze(0)

        # 현재 상태 가치
        V_s = self.critic(s)
        with torch.no_grad():
            # 다음 상태 가치
            V_ns = self.critic(ns)
            # TD 타깃 (done이면 V_ns 제외)
            td_target = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
            if not done:
                td_target += self.gamma * V_ns
            # TD 오차 = Advantage
            advantage = td_target - V_s

        # Critic 손실 (MSE)
        critic_loss = nn.functional.mse_loss(V_s, td_target)

        # Actor 손실 (policy gradient) + 엔트로피 보너스
        probs = self.actor(s)
        dist = Categorical(probs)
        log_prob = dist.log_prob(torch.tensor([action], device=self.device))
        entropy = dist.entropy().mean()
        actor_loss = -(log_prob * advantage.detach()).mean() - self.entropy_coef * entropy

        # 역전파
        self.a_opt.zero_grad(); self.c_opt.zero_grad()
        actor_loss.backward(); critic_loss.backward()
        self.a_opt.step(); self.c_opt.step()
        return actor_loss.item(), critic_loss.item()

    def save(self, path="checkpoints"):
        # 모델 저장
        Path(path).mkdir(exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def load(self, path="checkpoints_best"):
        # 모델 로드
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth"), map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth"), map_location=self.device))
        print("[LOAD] 모델 가중치 불러옴")

    def train(self, max_episodes=300):
        # 메인 학습 루프
        rewards = []
        best_reward = -float("inf")   # 지금까지 최고 보상

        for ep in range(max_episodes):
            state, _ = self.env.reset()
            done, truncated = False, False
            ep_reward = 0
            while not (done or truncated):
                action = self.select_action(state)   # 행동 선택
                next_state, reward, done, truncated, _ = self.env.step(action)
                # 네트워크 업데이트
                self.train_step(state, action, reward, next_state, done or truncated)
                state = next_state
                ep_reward += reward
            rewards.append(ep_reward)
            # 최고 보상 갱신 → 모델 저장
            if ep_reward > best_reward:
                best_reward = ep_reward
                self.save(path="checkpoints_best")
                print(f"[BEST] Episode {ep + 1}, New Best Reward: {best_reward}")

            if (ep+1) % 10 == 0:
                print(f"Episode {ep+1}, Avg Reward: {np.mean(rewards[-10:]):.2f}")
        self.save()
        print("[SAVE] 모델 저장 완료")
        print(f"[INFO] Best reward = {best_reward}, model saved in checkpoints_best/")

    def record_video(self, episodes=3, video_dir="videos"):
        # 학습된 모델로 비디오 녹화
        self.load()
        eval_env = gym.make(self.env_id, render_mode="rgb_array")
        eval_env = RecordVideo(eval_env, video_folder=video_dir, name_prefix="A2C")
        for ep in range(episodes):
            state, _ = eval_env.reset()
            done, truncated = False, False
            ep_reward = 0
            while not (done or truncated):
                with torch.no_grad():
                    s = to_tensor(state, self.device).unsqueeze(0)
                    probs = self.actor(s)
                    action = torch.argmax(probs, dim=-1).item()  # 테스트 시엔 greedy
                state, reward, done, truncated, _ = eval_env.step(action)
                ep_reward += reward
            print(f"[VIDEO] Episode {ep+1}, Reward: {ep_reward}")
        eval_env.close()
        print(f"[VIDEO] 저장 위치: {video_dir}/")


# ===== 실행 예시 =====
agent = A2CAgent()           # 에이전트 생성
agent.train(max_episodes=5000) # 학습
agent.record_video(episodes=2) # 학습된 모델로 비디오 저장
