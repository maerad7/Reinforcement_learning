from re import A
import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit
from agent import Agent

steps = 1000
epsilon = 0.1

bandit = Bandit()
agent = Agent(epsilon = epsilon)
total_reward = 0
total_rewards = [] # 보상 합
rates = []

for step in range(steps):
    action = agent.get_action() # 행도 선택
    reward = bandit.paly(action) # 보상 획득
    agent.update(action, reward) # 행동 가치 업데이트
    total_reward += reward

    total_rewards.append(total_reward) # 현재까지의 보상합 저장
    rates.append(total_reward/ (step + 1)) #현재 까지의 승률 저장

print(total_reward)

plt.ylabel('Total Reward')
plt.xlabel('Steps')
plt.plot(total_rewards)
plt.show()

plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(rates)
plt.show()
