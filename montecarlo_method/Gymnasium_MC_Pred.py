"""## Configuration for Colab"""   # 주피터/Colab 환경용 설정 섹션 표시

import gymnasium as gym           # Gymnasium 라이브러리 불러오기 (강화학습 환경)
import numpy as np                # 수치 계산용 Numpy 불러오기
# from utils import JupyterRender  # 주피터 렌더링 유틸 (현재는 주석 처리)
# ===============================================================
# 몬테카를로(Monte Carlo) 방법: 파이썬 주석으로 한 번에 이해하기
# ---------------------------------------------------------------
# [핵심 아이디어]
# - 환경의 전이확률 P(s'|s,a)나 보상함수 R(s,a,s')를 '모르더라도',
#   에피소드를 '샘플링'해서 얻은 실제 리턴(Return)을 표본 평균으로 사용해
#   가치함수(V)나 행동가치함수(Q)를 추정한다. (model-free)
# - 에피소드(시작~종료)를 완주한 뒤, 뒤에서부터 누적 보상 G_t를 계산하여
#   해당 상태(또는 상태-행동)에 대한 추정치를 업데이트한다.
#
# [필요 전제]
# - 에피소드가 '반드시 종료'하는 환경(episodic)에서 깔끔하게 동작한다.
# - 충분히 많은 샘플(무한히 많은 에피소드)과 모든 상태-행동 방문 가정하에 수렴.
#
# [리턴(Return) 정의]
#   G_t = R_{t+1} + γ R_{t+2} + γ^2 R_{t+3} + ... = Σ_{k=0}^{T-t-1} γ^k R_{t+1+k}
#   (여기서 γ는 감가율 0≤γ≤1, T는 에피소드 종료 시점)
#
# [세 가지 대표 태스크]
# 1) MC Prediction(정책 평가): 주어진 정책 π에 대해 V^π(s) 또는 Q^π(s,a) 추정
# 2) MC Control(정책 개선): ε-탐욕 등으로 정책을 개선하며 Q*와 최적 정책 π* 학습
# 3) Off-policy MC: 행동정책 b로 수집하고 목표정책 π를 평가(중요도 샘플링)
#
# 아래 코드는(의사코드에 가깝게) 핵심 함수들을 '주석 중심'으로 구현해 설명한다.
# ===============================================================
# Monte Carlo Prediction 클래스 정의
class MC_prediction:
    def __init__(self, env, pi, gamma=0.9, max_episode=100, render=False):
        self.state_dim = env.observation_space.n  # 상태 공간 크기 (FrozenLake의 상태 개수)
        self.action_dim = env.action_space.n      # 행동 공간 크기 (FrozenLake의 행동 개수)

        self.env = env                            # 환경 객체 저장

        # 환경 크기 (FrozenLake 4x4 환경 기준)
        self.nrow = 4
        self.ncol = 4

        self.gamma = gamma                        # 감가율 (discount factor)
        self.max_episode = max_episode            # 최대 에피소드 수
        self.render = render                      # 렌더링 여부 플래그
        self.pi = pi                              # 정책 (state별 선택 행동)

        self.v = np.zeros([self.state_dim])       # 상태 가치 함수 V(s) 초기화
        self.returns = [[] for _ in range(self.state_dim)]  # 상태별 리턴 기록

        self.seed = 777                           # 시드값 (재현성)

        # 정책 유효성 검사 (정책 길이 = 상태 개수, 행동 범위 확인)
        assert len(self.pi) == self.state_dim
        for i in range(self.state_dim):
            assert self.pi[i] >= 0 and self.pi[i] < self.action_dim

    def run(self):
        # Monte Carlo Prediction 실행 (정책 평가)

        for episode in range(self.max_episode):       # 지정된 에피소드 수만큼 반복
            observation, _ = self.env.reset()         # 환경 초기화 → 시작 상태

            done = False
            local_step = 0
            trajectory = []                           # (s,a,r) 저장할 trajectory

            while not done:                           # 에피소드 종료까지 반복
                action = self.pi[observation]         # 정책에 따라 행동 선택
                next_observation, reward, done, _, _ = self.env.step(action)  # 행동 실행

                # --- 보상 shaping (추가 규칙) ---
                if reward == 0:
                    reward = -0.001                   # 빈 칸에 머무르면 약간의 패널티
                if done and next_observation != 15:   # 종료인데 목표(G) 아닌 경우 → 구멍
                    reward = -1                       # 큰 패널티
                if local_step == 100:                 # 너무 오래 진행되면 종료 처리
                    done = True
                    reward = -1
                if observation == next_observation:   # 제자리 행동(무의미) → 패널티
                    reward = -1

                # trajectory에 (상태, 행동, 보상) 저장
                trajectory.append({'s': observation, 'a': action, 'r': reward})

                observation = next_observation        # 상태 업데이트
                local_step += 1                       # 스텝 수 증가

            # trajectory를 역순으로 뒤집기 (마지막 시점부터 리턴 G 계산하기 위함)
            trajectory.reverse()
            G = 0
            traj_states = list(map(lambda x: x['s'], trajectory))  # 상태들 리스트

            # 역순으로 리턴 누적
            for i in range(len(trajectory)):
                G = self.gamma * G + trajectory[i]['r']  # G_t = r + γG_{t+1}

                # First-Visit MC: 첫 방문한 상태만 업데이트
                if trajectory[i]['s'] not in traj_states[i+1:]:
                    self.returns[trajectory[i]['s']].append(G)
                    self.v[trajectory[i]['s']] = sum(self.returns[trajectory[i]['s']]) / len(self.returns[trajectory[i]['s']])

    def test(self, video_folder: str) -> None:
        """주어진 정책(pi)으로 환경 실행 + 비디오 녹화"""
        self.is_test = True

        naive_env = self.env                                     # 원본 env 저장
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder,  fps=2)  # 비디오 녹화 wrapper

        state, _ = self.env.reset(seed=self.seed)                # 시드 고정 후 초기화
        done = False
        score = 0

        while not done:
            action = self.pi[state]                              # 정책에 따른 행동
            next_observation, reward, done, _, _ = self.env.step(action)
            state = next_observation
            score += reward                                      # 보상 누적

        print("score: ", score)                                  # 최종 점수 출력
        self.env.close()                                         # 비디오 파일 닫기
        self.env = naive_env                                     # 원래 env로 되돌리기


# 환경 생성 (FrozenLake 4x4, 미끄럽지 않게 설정)
env = gym.make("FrozenLake-v1", render_mode='rgb_array', is_slippery=False)

# 미리 정의한 정책 (상태마다 특정 행동을 하도록 지정된 결정적 정책)
policy = np.array([1, 2, 1, 0, 1, 0, 1, 0, 2, 1, 1, 0, 0, 2, 2, 0], dtype=int)

# Monte Carlo Prediction 설정값
mc_config = {
    'env': env,
    'pi': policy,
    'gamma': 0.9,
    'render': True,
    'max_episode': 10
}

# MC Prediction 객체 생성
mc = MC_prediction(**mc_config)
mc.run()  # Monte Carlo 정책 평가 실행


"""## Test"""

# test: 지정된 정책으로 실제 환경 플레이 + 비디오 저장
video_folder="videos/per"
mc.test(video_folder=video_folder)


"""## Render"""

import base64
import glob
import io
import os
from IPython.display import HTML, display

def ipython_show_video(path: str) -> None:
    """지정된 mp4 비디오 파일을 주피터 노트북에서 바로 재생"""
    if not os.path.isfile(path):
        raise NameError("Cannot access: {}".format(path))

    video = io.open(path, "r+b").read()                     # 비디오 파일 읽기
    encoded = base64.b64encode(video)                       # base64로 인코딩

    # HTML video 태그로 표시
    display(
        HTML(
            data="""
        <video width="320" height="240" alt="test" controls>
        <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
        </video>
        """.format(
                encoded.decode("ascii")
            )
        )
    )

def show_latest_video(video_folder: str) -> str:
    """폴더에서 가장 최근 mp4 파일 찾아 재생"""
    list_of_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    latest_file = max(list_of_files, key=os.path.getctime)  # 최신 파일 찾기
    ipython_show_video(latest_file)                         # 주피터에 표시
    return latest_file

latest_file = show_latest_video(video_folder=video_folder)  # 최근 비디오 실행
print("Played:", latest_file)
