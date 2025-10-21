from collections import defaultdict     # 기본값을 자동으로 지정할 수 있는 딕셔너리 자료구조
import numpy as np                      # 수치 계산 라이브러리

from montecarlo_method.gridworld import GridWorld   # GridWorld 환경 불러오기 (사용자가 만든 환경 클래스)


# ε-greedy 정책에 따라 행동 확률을 계산하는 함수
def greedy_probs(Q, state, epsilon=0, action_size=4):
    qs = [Q[(state, action)] for action in range(action_size)]   # 현재 상태에서 각 행동(action)에 대한 Q값 리스트
    max_action = np.argmax(qs)                                  # Q값이 가장 큰 행동 선택

    base_prob = epsilon / action_size                           # 탐험 확률을 행동 개수로 나눈 기본 확률
    action_probs = {action: base_prob for action in range(action_size)}  # 모든 행동에 대해 기본 확률 할당
    action_probs[max_action] += (1 - epsilon)                   # 최적 행동에 (1-ε) 확률 추가
    return action_probs                                         # 행동 확률 분포 반환


# 몬테카를로 학습을 수행하는 에이전트 클래스
class McAgent:
    def __init__(self):
        self.gamma = 0.9                      # 감가율(미래 보상에 대한 현재 가치)
        self.alpha = 0.1                      # 학습률 (Q 업데이트 속도)
        self.action_size = 4                  # 가능한 행동 개수 (상, 하, 좌, 우)
        self.epsilon = 0.1                    # 탐험 확률 (ε-greedy)

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}   # 초기 정책은 모든 행동을 균일하게 선택
        self.pi = defaultdict(lambda: random_actions)           # 상태별 정책(π): 기본은 균등 분포
        self.Q = defaultdict(lambda: 0)                         # Q함수(상태-행동 가치): 기본값 0
        self.memory = []                                        # 한 에피소드의 (상태, 행동, 보상) 기록 저장


    def get_action(self, state):
        action_probs = self.pi[state]                           # 현재 상태에 대한 행동 확률 불러오기
        actions = list(action_probs.keys())                     # 행동 리스트
        probs = list(action_probs.values())                     # 행동 확률 리스트
        return np.random.choice(actions, p=probs)               # 확률 분포에 따라 행동 선택


    def add(self, state, action, reward):
        data = (state, action, reward)                          # (상태, 행동, 보상) 튜플 생성
        self.memory.append(data)                                # memory에 저장 (에피소드 궤적 기록)


    def reset(self):
        self.memory.clear()                                     # 에피소드가 끝나면 memory 초기화


    def update(self):
        G = 0                                                   # 반환값(Return) 초기화
        for data in reversed(self.memory):                      # 에피소드 궤적을 뒤에서부터(역순) 순회
            state, action, reward = data
            key = (state, action)
            G = self.gamma * G + reward                         # 반환값 G 업데이트 (할인 보상 합)
            self.Q[key] += (G - self.Q[key]) * self.alpha       # Q 함수 업데이트 (증분 방식)
            self.pi[state] = greedy_probs(self.Q, state, epsilon=self.epsilon)  # 정책 개선(ε-greedy)


# 실행 부분
if __name__ == "__main__":
    env = GridWorld()                          # GridWorld 환경 생성
    agent = McAgent()                          # Monte Carlo 에이전트 생성

    episodes = 10000                           # 학습할 에피소드 수
    for episode in range(episodes):
        state = env.reset()                    # 환경 초기화 및 시작 상태 획득
        agent.reset()                          # 에피소드 메모리 초기화

        while True:
            action = agent.get_action(state)   # 정책에 따라 행동 선택
            next_state, reward, done = env.step(action)   # 환경에서 한 스텝 실행

            agent.add(state, action, reward)   # (상태, 행동, 보상) 기록 저장
            if done:                           # 에피소드 종료 조건 도달 시
                agent.update()                 # Q 함수와 정책 업데이트
                break                          # 에피소드 종료

            state = next_state                 # 상태 갱신 후 다음 루프 진행

    env.render_q(agent.Q)                      # 최종 학습된 Q값 시각화/출력
