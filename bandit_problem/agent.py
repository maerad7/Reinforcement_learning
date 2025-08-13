import numpy as np

class Agent:
    def __init__(self, epsilon, action_size = 10):
        self.epsilon = epsilon # 탐색 확률
        self.Qs = np.zeros(action_size) # 각 행동의 가치
        self.Ns = np.zeros(action_size) # 각 행동의 횟수

    def update(self, action, reward):
        self.Ns[action] +=1 # 행동 횟수 증가
        self.Qs[action] += (reward - self.Qs[action]) / self.Ns[action] #    행동 가치 업데이트

    def get_action(self): # 행동 ㅅ헌택( 입실론 - 탐욕 정책)
        if np.random.rand() < self.epsilon: 
            return np.random.randint(0, len(self.Qs)) # 랜덤 행동
        return np.argmax(self.Qs) # 탐욕 행동 선택택

