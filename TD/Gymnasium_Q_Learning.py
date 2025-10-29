import gymnasium as gym
import numpy as np

class Q_learning:
    def __init__(self, env, gamma=0.8, alpha=0.1, eps=0.1, render=False, max_episode=1000):

        self.state_dim = env.observation_space.n
        self.action_dim = env.action_space.n

        self.env = env

        self.nrow = 4
        self.ncol = 4

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

        self.max_episode = max_episode

        self.render = render

        self.q = np.zeros([self.state_dim,self.action_dim])

        self.seed =777

    def action(self, s):
        if np.random.random() < self.eps:
            action = np.random.randint(low=0, high=self.action_dim - 1)
        else:
            action = np.argmax(self.q[s, :])

        return action

    def run(self):

        self.success = 0

        for episode in range(self.max_episode):
            observation, _ = self.env.reset()
            done = False

            episode_reward = 0
            local_step = 0

            while not done:
                action = self.action(observation)
                next_observation, reward, done, _ ,_ = self.env.step(action)

                if reward == 0:
                    reward = -0.001

                if done and next_observation != 15:
                    reward = -1

                if local_step == 100:
                    done = True
                    reward = -1

                if observation == next_observation:
                    reward = -1

                self.q[observation,action] = self.q[observation, action] + self.alpha*(reward + self.gamma*np.max(self.q[observation,:])- self.q[observation, action])

                observation = next_observation

                episode_reward += reward
                local_step += 1

            if observation == 15:
                self.success +=1

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, fps=2)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.action(state)
            next_observation, reward, done, _, _ = self.env.step(action)

            state = next_observation
            score += reward

        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env

env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)  # define the environment.


q_config = {
    'env': env,
    'gamma': 0.8,
    'alpha': 0.1,
    'eps': 0.1,
    'render': True,
    'max_episode': 10000
}

q_learning = Q_learning(**q_config)
q_learning.run()
video_folder="videos/Q-learning"
q_learning.test(video_folder=video_folder)