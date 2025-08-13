import numpy as np


# 정상 문제란 보상의 확률 분포가 변하지 않는 문제
# 비정상 문제란 보상의 확률 분포가 변하도록 설정된 문제
# 비정상 문제에서는 시간이 흐르면 환경이 변하기 때문에 과거 데이터 부상의 중요도는 낮아 져야됨

class NonSatatBandit:
    def __init__(self, arms= 10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self ,arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms) # 노이즈 추가
        if rate > np.random.rand():
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, alpha, epsilon, actions = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)


