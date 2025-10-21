from collections import defaultdict
from gridworld import GridWorld   # GridWorld 환경을 불러옴 (사용자가 만든 gridworld.py 안에 정의된 클래스)

# --------------------------
# 가장 큰 값을 가지는 key를 반환하는 함수
# --------------------------
def argmax(d):
    max_value = max(d.values())   # 딕셔너리 d에서 최대 value 찾기
    max_key = 0
    for key, value in d.items():
        if value == max_value:    # value가 최대값이면
            max_key = key         # 해당 key를 저장
    return max_key                # 최대값을 가진 key 반환

# --------------------------
# 가치 함수 V를 기반으로 하는 탐욕 정책(greedy policy) 생성
# --------------------------
def greedy_policy(V, env, gamma):
    pi = {}   # 정책 π를 저장할 딕셔너리

    for state in env.states():   # 환경의 모든 상태에 대해
        action_values = {}       # 각 행동의 가치를 저장할 딕셔너리

        for action in env.actions():  # 가능한 모든 행동에 대해
            next_state = env.next_state(state, action)       # 해당 행동을 했을 때의 다음 상태
            r = env.reward(state, action, next_state)        # 보상
            value = r + gamma * V[next_state]                # 행동 가치 Q(s,a) = r + γ * V(s')
            action_values[action] = value                    # 행동-가치 저장

        max_action = argmax(action_values)                   # 최대 가치 행동 선택
        action_probs = {0:0, 1:0, 2:0, 3:0}                  # 행동 확률 초기화
        action_probs[max_action] = 1.0                       # 최적 행동 확률만 1로 설정 (탐욕 정책)
        pi[state] = action_probs                             # 해당 상태의 정책 저장

    return pi

# --------------------------
# 정책 평가 1스텝 (주어진 정책 π로 가치함수 V를 업데이트)
# --------------------------
def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:   # 목표 상태의 가치는 0
            V[state] = 0
            continue

        action_probs = pi[state]  # 현재 상태에서의 정책 (행동별 확률)
        new_V = 0

        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)       # 다음 상태
            r = env.reward(state, action, next_state)        # 보상
            # 벨만 기대 방정식: V(s) = Σ_a π(a|s) [ r + γ V(s') ]
            new_V += action_prob * (r + gamma * V[next_state])

        V[state] = new_V   # 상태 가치 업데이트
    return V

# --------------------------
# 정책 평가 (반복해서 수렴할 때까지 가치 함수 업데이트)
# --------------------------
def policy_eval(pi, V, env, gamma, threshold=0.001):
    while True:
        old_V = V.copy()                             # 이전 가치 함수 복사
        V = eval_onestep(pi, V, env, gamma)          # 정책 평가 한 스텝 수행

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])         # 상태 가치 변화량
            if delta < t:
                delta = t                            # 가장 큰 변화량 기록

        if delta < threshold:                        # 충분히 수렴하면 반복 종료
            break
    return V

# --------------------------
# 정책 반복 (policy iteration)
# --------------------------
def policy_iter(env, gamma, threshold=0.001, is_render=False):
    # 초기 정책: 모든 행동에 대해 균등 확률(0.25씩)
    pi = defaultdict(lambda : {0:0.25, 1:0.25, 2:0.25, 3:0.25})
    # 초기 가치 함수: 0으로 초기화
    V = defaultdict(lambda : 0)

    while True:
        # (1) 정책 평가: 현재 정책 π에 대해 가치 함수 V 계산
        V = policy_eval(pi, V, env, gamma, threshold)
        # (2) 정책 개선: 가치 함수에 기반하여 새로운 탐욕 정책 생성
        new_pi = greedy_policy(V, env, gamma)

        if is_render:
            env.render_v(V, pi)  # 환경 시각화 (가치 함수 및 정책 출력)

        if new_pi == pi:         # 정책이 변하지 않으면 최적 정책 도달
            break

        pi = new_pi              # 새 정책으로 업데이트 후 반복

    return pi

# --------------------------
# 실행부
# --------------------------
if __name__ == '__main__':
    env = GridWorld()                   # GridWorld 환경 초기화
    gamma = 0.9                         # 할인율 설정
    pi = policy_iter(env, gamma, is_render=True)  # 정책 반복 실행
