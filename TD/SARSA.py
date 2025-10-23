from collections import defaultdict, deque
import numpy as np
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


class SarsaAgent:
    """
    온-폴리시(온정책) TD 제어 알고리즘인 SARSA(λ=0 버전)의 핵심 로직을 구현한 에이전트.
    - 업데이트 식: Q(S, A) ← Q(S, A) + α [R + γ Q(S', A') - Q(S, A)]
    - 이 구현은 메모리 deque를 길이 2로 유지하며,
      직전 전이와 현재 전이를 함께 보면서 Q(S,A)를 SARSA 방식으로 업데이트한다.
    """
    def __init__(self):
        # 하이퍼파라미터 설정
        self.gamma = 0.9       # 감가율 γ: 미래 보상의 현재가치 가중치 (0~1)
        self.alpha = 0.8       # 학습률 α: TD 오차를 Q에 반영하는 비율 (0~1)
        self.epsilon = 0.1     # 탐험률 ε: 정책이 무작위로 행동을 시도할 확률
        self.action_size = 4   # 행동 개수(상하좌우 등)

        # 초기 정책 π를 균등분포로 세팅 (모든 상태에서 모든 행동 확률 0.25)
        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)  # 상태 미등록 시 균등정책을 기본으로 사용

        # Q 테이블: (state, action) → 값. 미등록 키 접근 시 0으로 초기화
        self.Q = defaultdict(lambda: 0)

        # 최근 2개 전이만 보관하는 버퍼
        # 각 전이 항목: (state, action, reward, done)
        # 예: t-1 시점 전이와 t 시점 전이를 함께 보며 SARSA 업데이트 수행
        self.memory = deque(maxlen=2)

    def reset(self):
        """에피소드 시작 시 메모리(전이 버퍼) 초기화."""
        self.memory.clear()

    def get_action(self, state):
        """
        현재 정책 π에 따라 확률적으로 행동을 샘플링한다(정책 실행).
        - self.pi[state]는 {action: prob} 형식의 분포.
        """
        action_probs = self.pi[state]                # 현재 상태에서의 행동 확률 분포
        actions = list(action_probs.keys())          # 행동 인덱스 목록
        probs = list(action_probs.values())          # 각 행동에 대한 확률
        return np.random.choice(actions, p=probs)    # 확률 분포에 따라 1개 행동 샘플링

    def update(self, state, action, reward, done):
        """
        SARSA 업데이트를 수행한다.
        호출 프로토콜(루프 외부에서 보장):
          1) 매 스텝마다 (s, a, r, done)을 먼저 push
          2) 메모리 길이가 2가 되면 이전 전이(t-1)를 현재 전이(t)의 (s', a')와 함께 사용해 업데이트
          3) 에피소드가 끝나면 (terminal_state, None, None, None)을 한 번 더 push해서
             마지막 전이도 다음-상태/행동이 없는 형태로 업데이트되도록 한다.
        """
        # 1) 이번 스텝의 전이를 버퍼에 추가
        self.memory.append((state, action, reward, done))

        # 2) 버퍼가 2개 차야 (이전 전이, 현재 전이) 쌍으로 업데이트 가능
        if len(self.memory) < 2:
            return

        # 3) t-1 시점 전이(업데이트 대상)와 t 시점 전이(부트스트랩용 다음 상태/행동) 분리
        state, action, reward, done = self.memory[0]   # 업데이트할 (S, A, R, done)
        next_state, next_action, _, _ = self.memory[1]                     # 부트스트랩용 (S', A')

        # 4) SARSA의 다음 상태-행동 가치 Q(S', A') 계산
        #    터미널이면 부트스트랩을 끊어 0으로 둔다.
        next_q = 0 if done else self.Q[(next_state, next_action)]

        # 5) TD 타깃: R + γ Q(S', A')
        target = reward + self.gamma * next_q

        # 6) TD 오차를 이용한 Q 업데이트: Q ← Q + α (target - Q)
        self.Q[(state, action)] += (target - self.Q[(state, action)]) * self.alpha

        # 7) 정책 개선(ε-탐욕화): 최신 Q를 반영해 해당 상태의 행동분포를 갱신
        #    → 온-폴리시이므로, 학습과 동일한 정책으로 행동을 계속 샘플링
        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


# --------- 학습 환경 및 루프 ---------
env = GridWorld()        # GridWorld 환경 인스턴스. reset(), step(a), render_q(Q) 등을 제공해야 함
agent = SarsaAgent()     # SARSA 에이전트 생성

episodes = 10000         # 총 에피소드 수
for episode in range(episodes):
    state = env.reset()  # 환경 초기화 및 시작 상태 반환
    agent.reset()        # 에이전트 내부 버퍼 초기화(전이 deque 비우기)

    while True:
        # 1) 현재 정책에 따라 행동 선택(ε-탐욕 정책)
        action = agent.get_action(state)

        # 2) 환경 상호작용: S --a--> (S', R, done)
        next_state, reward, done = env.step(action)

        # 3) (S, A, R, done)을 버퍼에 추가하고, 가능하면 직전 전이를 SARSA로 업데이트
        agent.update(state, action, reward, done)

        # 4) 에피소드 종료 처리
        if done:
            # 마지막 전이(S_T-1, A_T-1, R_T, done=True)의 업데이트를 위해
            # 더미 전이(S_T, None, None, None)를 한 번 더 넣어
            # "다음 상태-행동"이 없는 형태(Q=0 부트스트랩)로 마무리 업데이트되도록 한다.
            agent.update(next_state, None, None, None)
            break

        # 5) 종료가 아니면 다음 루프를 위해 상태 갱신
        state = next_state

# 학습이 끝난 뒤, 환경이 제공하는 유틸로 Q테이블 시각화/출력
env.render_q(agent.Q)
