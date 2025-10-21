from collections import defaultdict
from math import gamma   # gamma 함수 (사용자 코드에서는 아래에서 같은 이름 변수 gamma를 쓰기 때문에 주의 필요)

from policy_iter import greedy_policy   # 정책 반복 코드에서 정의한 탐욕 정책 함수 import
from gridworld import GridWorld         # GridWorld 환경 불러오기


# --------------------------
# 가치 반복법의 1스텝 업데이트
# --------------------------
def value_iter_onestep(V, env, gamma):
    for state in env.states():                  # 환경의 모든 상태에 대해 반복
        if state == env.goal_state:             # 목표 상태는 가치 0으로 고정
            V[state] = 0
            continue

        action_values = []                      # 가능한 행동들의 가치 저장 리스트
        for action in env.actions():            # 모든 행동에 대해
            next_state = env.next_state(state,action)   # 행동 수행 후의 다음 상태
            r = env.reward(state, action, next_state)   # 보상 계산
            value = r + gamma * V[next_state]           # 벨만 최적 방정식 Q(s,a) = r + γ V(s')
            action_values.append(value)                 # 행동 가치 리스트에 추가

        V[state] = max(action_values)           # 상태 가치 = 가능한 행동 가치 중 최대값

    return V                                    # 업데이트된 가치 함수 반환


# --------------------------
# 전체 가치 반복 (Value Iteration)
# --------------------------
def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)                     # 현재 가치 함수를 화면에 시각화

        old_V = V.copy()                        # 기존 가치 함수 복사
        V = value_iter_onestep(V, env, gamma)   # 한 번의 가치 반복 수행

        delta = 0                               # 가치 변화량 초기화
        for state in V.keys():
            t = abs(V[state] - old_V[state])    # 상태별 가치 변화량
            if delta < t:
                dalta = t                       # (오타: delta 대신 dalta) 최대 변화량 갱신

        if delta < threshold:                   # 변화량이 임계값보다 작으면 수렴했다고 판단
            break

    return V                                    # 최종 가치 함수 반환


# --------------------------
# 실행부
# --------------------------
if __name__=="__main__":
    V = defaultdict(lambda:0)                   # 초기 가치 함수: 모든 상태 0
    env = GridWorld()                           # GridWorld 환경 초기화
    gamma = 0.9                                 # 할인율 설정

    V = value_iter(V, env, gamma)               # 가치 반복 수행
    pi = greedy_policy(V, env, gamma)           # 최적 가치 함수 기반으로 탐욕 정책 도출
    env.render_v(V, pi)                         # 최종 가치 함수와 정책 시각화
