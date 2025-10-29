import gymnasium as gym   # 강화학습 환경 라이브러리 (OpenAI Gym 후속)
import numpy as np        # 수치 연산 라이브러리

class SARSA:
    def __init__(self, env, gamma=0.9, alpha=0.1, eps=0.1, render=False, max_episode=1000):

        self.state_dim = env.observation_space.n  # 상태 공간 크기 (FrozenLake는 16칸)
        self.action_dim = env.action_space.n      # 행동 공간 크기 (FrozenLake는 4가지 방향)

        self.env = env                            # 환경 저장

        # FrozenLake는 4x4 격자판
        self.nrow = 4
        self.ncol = 4

        # 하이퍼파라미터들
        self.alpha = alpha    # 학습률
        self.gamma = gamma    # 할인율
        self.eps = eps        # 입실론 (탐험 확률)
        self.max_episode = max_episode  # 최대 학습 에피소드 수
        self.render = render  # 렌더링 여부

        # Q 테이블 초기화 (상태 × 행동)
        self.q = np.zeros([self.state_dim, self.action_dim])

        # 랜덤 시드
        self.seed = 777


    def action(self, s):
        # epsilon-greedy 정책
        if np.random.random() < self.eps:                    # 확률 ε로 무작위 행동
            action = np.random.randint(low=0, high=self.action_dim - 1)
        else:                                                # 확률 (1-ε)로 최적 행동
            action = np.argmax(self.q[s,:])                  # 상태 s에서 Q값이 최대인 행동 선택
        return action

    def run(self):

        self.success = 0  # 성공 횟수 카운트 (목표 도착 횟수)

        # 여러 에피소드 실행
        for episode in range(self.max_episode):
            observation, _ = self.env.reset()  # 환경 초기화 (state 반환)
            action = self.action(observation)  # 첫 행동 선택

            done = False  # 에피소드 종료 여부
            episode_reward = 0  # 에피소드 누적 보상
            local_step = 0  # 스텝 카운트

            while not done:  # 종료 전까지 반복
                next_observation, reward, done, _, _ = self.env.step(action)  # 행동 실행

                # 보상 shaping: 원래 FrozenLake는 보상이 sparse하므로 학습을 돕기 위해 추가
                if reward == 0:
                    reward = -0.001  # 움직이지만 도착 못하면 작은 패널티
                if done and next_observation != 15:
                    reward = -1  # 구멍에 빠지면 큰 패널티
                if local_step == 100:
                    done = True  # 무한 루프 방지
                    reward = -1
                if observation == next_observation:
                    reward = -1  # 같은 상태에 머물면 패널티

                next_action = self.action(next_observation)  # 다음 행동 선택

                # SARSA 업데이트 식
                self.q[observation, action] = self.q[observation, action] + \
                                              self.alpha * (
                                                          reward + self.gamma * self.q[next_observation, next_action] -
                                                          self.q[observation, action])

                # 상태, 행동 업데이트
                observation = next_observation
                action = next_action

                # 보상과 스텝 누적
                episode_reward += reward
                local_step += 1

            # 목표 상태(15번 칸)에 도착한 경우 성공 카운트 증가
            if observation == 15:
                self.success += 1

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # 비디오 기록을 위한 wrapper 적용
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, fps=2)

        state, _ = self.env.reset(seed=self.seed)  # 테스트 환경 초기화
        action = self.action(state)  # 첫 행동 선택
        done = False
        score = 0  # 총 점수 (보상 합)

        while not done:  # 에피소드 종료 전까지 반복
            action = self.action(state)  # 행동 선택
            next_observation, reward, done, _, _ = self.env.step(action)  # 환경에서 실행

            next_action = self.action(next_observation)  # 다음 행동 선택
            action = next_action
            state = next_observation
            score += reward  # 보상 누적

        print("score: ", score)  # 최종 점수 출력
        self.env.close()  # 비디오 저장 종료

        # 원래 환경으로 복원
        self.env = naive_env


# FrozenLake 환경 생성 (4x4, 미끄러짐 제거)
env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)

# SARSA 설정값 정의
sarsa_config = {
    'env': env,
    'gamma': 0.8,     # 할인율
    'alpha': 0.1,     # 학습률
    'eps': 0.1,       # 입실론 (탐험 확률)
    'render': True,   # 렌더링 여부
    'max_episode': 100  # 학습 에피소드 수
}

sarsa = SARSA(**sarsa_config)      # SARSA 객체 생성
sarsa.run()                        # 학습 실행

video_folder="videos/Sarsa"          # 비디오 저장 폴더 경로
sarsa.test(video_folder=video_folder)  # 학습된 정책 테스트 + 비디오 저장
