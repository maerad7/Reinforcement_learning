from collections import defaultdict, deque
import numpy as np
from sympy.abc import alpha

from TD.SARSA import episodes
from gridworld import GridWorld

def greedy_probs(Q, state, epsilon=0, action_size=4):
    """
    ε-탐욕(ε-greedy) 정책 분포를 생성한다.
    - 핵심 아이디어: 확률 (1-ε)로 현재 Q값이 최대인 행동을 선택하고,
                    나머지 ε 확률은 모든 행동에 균등 분배하여 탐험(exploration)한다.
    - 입력
      Q: defaultdict 형태의 Q테이블. 키는 (state, action) 튜플, 값은 실수 Q값.
      state: 현재 상태(해시 가능해야 함).
      epsilon: 탐험 비율 0~1. 0이면 완전 탐욕, 1이면 완전 무작위.
      action_size: 행동의 개수. GridWorld 가정상 {0,1,2,3} 4방향.
    - 출력
      action_probs: {action: prob} 형태의 확률 분포 딕셔너리.
    """
    qs = [Q[(state, action)] for action in range(action_size)]   # 현재 상태에서 가능한 모든 행동의 Q값 목록
    max_action = np.argmax(qs)                                   # Q값이 가장 큰(동률이면 첫 번째) 행동 인덱스

    base_prob = epsilon / action_size                            # 탐험 확률 ε를 모든 행동에 균등 분배한 기본 확률
    action_probs = {action: base_prob for action in range(action_size)}  # 모든 행동에 기본 확률로 초기화
    action_probs[max_action] += (1 - epsilon)                    # 최선 행동에 (1-ε) 만큼 추가하여 합이 1이 되도록 함
    return action_probs

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9                # 할인율 (미래 보상에 대한 중요도)
        self.alpha = 0.8                # 학습률 (가중치 업데이트 비율)
        self.epsilon = 0.1              # 탐험 비율 (ε-greedy 정책에서 무작위 탐험 확률)
        self.action_size = 4            # 가능한 행동 개수 (GridWorld에서 4방향)

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}  # 초기 상태: 모든 행동 균등 확률
        self.pi = defaultdict(lambda: random_actions)          # 정책 π: 상태별로 행동 분포를 저장

        self.Q = defaultdict(lambda: 0)                       # Q 테이블: (state, action) 쌍 → Q값, 기본값 0
        self.b = defaultdict(lambda : random_actions)          # 행동 정책 b (탐험 포함 ε-greedy)

    def get_action(self, state):
        """
        현재 정책 π에 따라 확률적으로 행동을 샘플링한다.
        """
        action_probs = self.pi[state]                # 현재 상태에서의 행동 확률 분포
        actions = list(action_probs.keys())          # 가능한 행동 리스트
        probs = list(action_probs.values())          # 각 행동의 확률 리스트
        return np.random.choice(actions, p=probs)    # 확률에 따라 무작위로 행동 선택

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning 업데이트 수행.
        - 현재 (s,a) 쌍의 Q값을 보상과 다음 상태의 최대 Q값을 사용해 갱신.
        """
        if done:  # 종료 상태라면 다음 Q값 없음
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]  # 다음 상태에서 가능한 모든 Q값
            next_q_max = max(next_qs)                                           # 그 중 최댓값 (greedy)

        target = reward + self.gamma * next_q_max                               # Q-learning 타깃
        self.Q[state,action] += (target - self.Q[state,action]) * alpha         # Q값 업데이트 (TD error 반영)

        self.pi[state] = greedy_probs(self.Q, state, epsilon=0)                 # 정책 π: 항상 greedy로 업데이트
        self.b[state] = greedy_probs(self.Q, state, self.epsilon)               # 행동 정책 b: ε-greedy로 업데이트

# 환경 및 학습 루프
env = GridWorld()
agent = QLearningAgent()
episodes = 10000
for episode in range(episodes):
    state = env.reset()   # 에피소드 시작 시 환경 초기화

    while True:
        action = agent.get_action(state)                 # 정책에 따라 행동 선택
        next_state, reward, done = env.step(action)      # 환경에 행동을 적용하고 결과 관찰

        agent.update(state, action, reward ,next_state, done)  # Q값 업데이트

        if done:    # 목표에 도달하거나 실패 시 에피소드 종료
            break

        state = next_state   # 다음 상태로 진행

env.render_q(agent.Q)  # 학습된 Q테이블 시각화
