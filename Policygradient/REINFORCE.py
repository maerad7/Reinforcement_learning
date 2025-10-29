import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# ----- Policy Network 정의 -----
class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNet, self).__init__()
        # 입력 → 은닉1
        self.fc1 = nn.Linear(state_size, hidden_size)
        # 은닉1 → 은닉2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 은닉2 → 행동 확률 (출력)
        self.policy_head = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        # 첫 번째 은닉층 (ReLU 활성화)
        x = torch.relu(self.fc1(x))
        # 두 번째 은닉층 (ReLU 활성화)
        x = torch.relu(self.fc2(x))
        # 행동 확률 분포 (Softmax)
        logits = self.policy_head(x)
        return torch.softmax(logits, dim=-1)

# ----- REINFORCE Agent -----
class REINFORCEAgent:
    def __init__(self, env, lr=7e-3, gamma=0.99, hidden_size=64, device="cpu"):
        self.env = env
        self.gamma = gamma
        self.device = device

        # 상태와 행동 차원 파악
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        # 정책 신경망 생성
        self.policy = PolicyNet(self.state_size, self.action_size, hidden_size).to(device)
        # 옵티마이저 (Adam)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def get_action(self, state):
        # 상태를 텐서로 변환하고 배치 차원 추가
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # 정책 네트워크로 확률 분포 얻기
        probs = self.policy(state)
        # Categorical 분포로 모델링
        dist = Categorical(probs)
        # 확률적으로 행동 샘플링
        action = dist.sample()
        # 행동과 로그확률 반환
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards):
        """할인된 리턴 G_t 계산"""
        R = 0
        returns = []
        # 뒤에서부터 할인 보상 누적
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def train_step(self, log_probs, returns):
        # 리턴을 텐서로 변환
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # 안정화를 위해 리턴 정규화
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        # 각 시점별 정책 손실 = -logπ(a|s) * G_t
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)
        # 전체 손실 = 합계
        loss = torch.stack(policy_loss).sum()

        # 역전파 학습
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

        # ----- 모델 저장 -----

    def save(self, filepath="reinforce_model.pth"):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"모델 저장 완료: {filepath}")

        # ----- 모델 불러오기 -----

    def load(self, filepath="reinforce_model.pth"):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"모델 불러오기 완료: {filepath}")
# ----- 학습 루프 -----
if __name__ == "__main__":
    # CartPole 환경 생성
    env = gym.make("CartPole-v0")
    seed = 12345
    env.reset(seed=seed)
    torch.manual_seed(seed)

    # 에이전트 생성
    agent = REINFORCEAgent(env, lr=7e-3, gamma=0.99, hidden_size=64)
    max_episodes = 300
    scores = []

    for episode in range(max_episodes):
        # 환경 리셋
        state, _ = env.reset()
        log_probs = []   # 로그확률 저장
        rewards = []     # 보상 저장
        done = False
        episode_reward = 0

        while not done:
            # 행동 선택
            action, log_prob = agent.get_action(state)
            # 환경 한 스텝 실행
            next_state, reward, done, _, _ = env.step(action)

            # 로그확률과 보상 기록
            log_probs.append(log_prob)
            rewards.append(reward)

            # 상태 업데이트
            state = next_state
            episode_reward += reward

        # 에피소드가 끝난 후 리턴 계산
        returns = agent.compute_returns(rewards)
        # 학습 (정책 파라미터 업데이트)
        loss = agent.train_step(log_probs, returns)

        # 성과 기록 및 출력
        scores.append(episode_reward)
        print(f"Episode {episode+1}, Reward: {episode_reward}, Loss: {loss:.4f}")

        # 50 에피소드마다 모델 저장
        if (episode + 1) % 50 == 0:
            agent.save(f"reinforce_model_ep{episode + 1}.pth")

        # 학습 끝나고 최종 모델 저장
    agent.save("reinforce_model_final.pth")