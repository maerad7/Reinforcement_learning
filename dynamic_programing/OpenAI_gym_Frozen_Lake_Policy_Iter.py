import gym
import numpy as np
# 1. 정책 초기화
# - 시작은 임의 정책 (예: 모든 상태에서 행동 무작위 선택)으로 시작한다.
# - policy[s] = {a: 확률} 형태로 저장 가능.

# 2. 정책 평가 (Policy Evaluation)
# - 현재 정책 π에 대해 가치함수 V(s)를 근사적으로 구한다.
# - 벨만 기대 방정식: V(s) = Σ_a π(a|s) Σ_s' P(s'|s,a)[ R(s,a,s') + γ V(s') ]
# - 실제 구현에서는 반복 계산(벨만 기대 방정식의 고정점 근사)을 사용.

# 3. 정책 개선 (Policy Improvement)
# - 구한 V(s)를 기준으로 각 상태에서 Q(s,a)를 계산한다:
#   Q(s,a) = Σ_s' P(s'|s,a)[ R(s,a,s') + γ V(s') ]
# - Q(s,a)가 가장 큰 행동 a*를 선택해 새로운 정책 π'(s)=argmax_a Q(s,a) 로 갱신.

# 4. 정책 안정성 검사
# - 새로운 정책과 이전 정책이 동일하면 종료 (정책 수렴).
# - 다르면 다시 2단계(정책 평가)로 돌아가 반복.

# 5. 종료
# - 수렴된 정책 π*와 그에 대응하는 최적 가치 함수 V*를 반환.

env = gym.make('FrozenLake-v1')

# env.render()

state_size  = env.observation_space.n
action_size = env.action_space.n

def compute_value_function(policy, gamma=1.0):

    # initialize value table with zeros
    value_table = np.zeros(state_size)

    # set the threshold
    threshold = 1e-10

    while True:

        # copy the value table to the updated_value_table
        updated_value_table = np.copy(value_table)

        # for each state in the environment, select the action according to the policy and compute the value table
        for state in range(state_size):
            action = policy[state]

            # build the value table with the selected action
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                        for trans_prob, next_state, reward_prob, _ in env.P[state][action]])

        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break

    return value_table

def extract_policy(value_table, gamma = 1.0):

    # Initialize the policy with zeros
    policy = np.zeros(state_size)


    for state in range(state_size):

        # initialize the Q table for a state
        Q_table = np.zeros(action_size)

        # compute Q value for all ations in the state
        for action in range(action_size):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

        # Select the action which has maximum Q value as an optimal action of the state
        policy[state] = np.argmax(Q_table)

    return policy

def policy_iteration(env,gamma = 1.0):

    # Initialize policy with zeros
    old_policy = np.zeros(state_size)
    no_of_iterations = 200000

    for i in range(no_of_iterations):

        # compute the value function
        new_value_function = compute_value_function(old_policy, gamma)

        # Extract new policy from the computed value function
        new_policy = extract_policy(new_value_function, gamma)

        # Then we check whether we have reached convergence i.e whether we found the optimal
        # policy by comparing old_policy and new policy if it same we will break the iteration
        # else we update old_policy with new_policy

        if (np.all(old_policy == new_policy)):
            print ('Policy-Iteration converged at step %d.' %(i+1))
            break
        old_policy = new_policy

    return new_policy

print (policy_iteration(env))