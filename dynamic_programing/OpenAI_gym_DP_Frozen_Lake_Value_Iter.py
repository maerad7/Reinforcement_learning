import random
import gym
import numpy as np

# 1. 가치 함수 초기화
# - 모든 상태 s에 대해 V(s) = 0 으로 초기화 (혹은 임의의 값).
# - 종료 상태는 V(s) = 0 으로 두는 경우가 많음.

# 2. 반복 수행 (벨만 최적 방정식 기반 업데이트)
# - 모든 상태 s에 대해 다음 값을 계산:
#   V_new(s) = max_a Σ_s' P(s'|s,a) [ R(s,a,s') + γ V(s') ]
# - 즉, 모든 행동 a를 고려해 그 행동의 기대 보상을 계산하고,
#   그 중 최대값을 새로운 가치로 반영.
# - 여기서 γ는 할인율(discount factor).

# 3. 수렴 조건 확인
# - |V_new(s) - V(s)| 의 최대값(delta)이 임계값 θ보다 작으면 수렴.
# - 충분히 작은 변화만 남으면 루프 종료.

# 4. 정책 추출 (Policy Extraction)
# - 수렴된 V*를 이용해 최적 정책 π*를 얻는다:
#   π*(s) = argmax_a Σ_s' P(s'|s,a) [ R(s,a,s') + γ V(s') ]
# - 즉, 각 상태에서 Q(s,a)를 계산해 가장 큰 행동을 선택.

# 5. 결과 반환
# - 최적 가치 함수 V*와 최적 정책 π*를 반환한다.

class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            gamma (float): discount factor
        """

        # 2.3 CREATING THE Q-TABLE
        self.env = env

        self.state_size  = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        self.gamma = 0.9    # discount rate

    def one_step_lookahead(self, env, state, V, discount_factor):
        action_values = np.zeros(self.action_size)
        for action in range(self.action_size):
            for probability, next_state, reward, done in self.env.P[state][action]:
                action_values[action] += probability * (reward + discount_factor * V[next_state])
        return action_values

    def value_iteration(self, env, discount_factor=1.0, theta=1e-9, max_iterations=1e9):
        # Number of evaluation iterations
        evaluation_iterations = 1
        # Initialize state-value function with zeros for each env state
        V = np.zeros(self.state_size)
        for i in range(int(max_iterations)):
            # Initialize a change of value function as zero
            delta = 0
            # Iterate though each state
            for state in range(self.state_size):

                # Do a one-step lookahead to calculate state-action values
                action_value = self.one_step_lookahead(self.env, state, V, discount_factor)

                # Select best action to perform based on the highest state-action value
                best_action_value = np.max(action_value)

                # Calculate the absolute change of value function
                delta = max(delta, np.abs(V[state] - best_action_value))

                # Update the value function for current state
                V[state] = best_action_value
            evaluation_iterations += 1

            # Terminate if value change is insignificant
            if delta < theta:
                print(f'Value-iteration converged at iteration#{i}.')
                break

        # Create a deterministic policy using the optimal value function
        policy = np.zeros([self.state_size, self.action_size])
        for state in range(self.state_size):

            # One step lookahead to find the best action for this state
            action_value = self.one_step_lookahead(self.env, state, V, discount_factor)
            # Select best action based on the highest state-action value
            best_action = np.argmax(action_value)

            # Update the policy to perform a better action at a current state
            policy[state, best_action] = 1.0

        return policy, V

'''
Value iteration
'''

# 2.2 CREATING THE ENVIRONMENT
env_name = "FrozenLake-v1"
env = gym.make(env_name)
# env.seed(777)     # reproducible, general Policy gradient has high variance

# 2.4 INITIALIZING THE Q-PARAMETERS
max_episodes = 10000  # Set total number of episodes to train agent on.

max_iterations = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate
render = False                # display the game environment


# train
agent = DQNAgent(
    env,
#     memory_size,
#     batch_size,
#     epsilon_decay,
)

if __name__ == "__main__":
    # Search for an optimal policy using policy iteration
    policy, V = agent.value_iteration(env.env)
    # Apply best policy to the real env

    wins = 0
    episode_reward = 0

    for episode in range(max_episodes):
        state = agent.env.reset()
        done = False  # has the enviroment finished?

        if render: env.render()

        # 2.7 EACH TIME STEP
        while not done:
            # Select best action to perform in a current state
            action = np.argmax(policy[state])
            # Perform an action an observe how env acted in response
            next_state, reward, done, _ = agent.env.step(action)

            if render: env.render()
            # Our new state is state
            state = next_state

            # Summarize total reward
            episode_reward += reward
            # Calculate number of wins over episodes
            if done and reward == 1.0:
                wins += 1
    average_reward = episode_reward / max_episodes

    print(f'Value Iteration : number of wins over {max_episodes} episodes = {wins}')
    print(f'Value Iteration : average reward over {max_episodes} episodes = {average_reward} \n\n')