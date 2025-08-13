import numpy as np

# slot machine problem
class Bandit:
    def __init__(self, arms = 10):
        self.rates = np.random.rand(arms)
        
    def paly(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0

bandit = Bandit()

for i in range(3):
    print(bandit.paly(0))

# 증분 구현
Q = 0
# 0번째 슬롯 머신에만 집중하여 해당 슬로멋신의 가치를 추정
for n in range(1,11):
    reward = bandit.paly(0)
    Q = Q + (1/n) * (reward - Q)
    print(Q)

# 10번 연속으로 플레이하고 보상을 받을 때 마다 슬롯머신의 가치를 추정
bandit = Bandit()
Qs = np.zeros(10)
Ns = np.zeros(10)

for n in range(1,11):
    action = np.random.randint(0,10)
    reward = bandit.paly(action)
    Ns[action] = Ns[action] + 1
    Qs[action] = Qs[action] + (1/Ns[action]) * (reward - Qs[action])
    print(Qs)
