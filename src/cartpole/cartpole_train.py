import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Policy 신경망
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),  # 입력: state 4개
            nn.ReLU(),
            nn.Linear(128, 2),  # 출력: 행동 2개 (좌/우)
            nn.Softmax(dim=-1)  # 확률로 변환
        )

    def forward(self, x):
        return self.net(x)

# 2. 에피소드 하나 실행: (state, action, reward) 기록
def run_episode(env, policy):
    obs, _ = env.reset()
    states, actions, rewards = [], [], []

    while True:
        # state → tensor로 변환
        state_t = torch.tensor(obs, dtype=torch.float32)

        # policy한테 확률 물어보고 샘플링
        probs = policy(state_t)
        action = torch.multinomial(probs, 1).item()

        # 환경에 action 던지기
        obs, reward, terminated, truncated, _ = env.step(action)

        states.append(state_t)
        actions.append(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    return states, actions, rewards

# 3. 리턴 계산: 나중 reward일수록 조금 할인 (0.99 감가)
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    # 정규화 (학습 안정화)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns

# 4. 학습 루프
env = gym.make("CartPole-v1")
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

best_reward = 0  # 역대 최고 점수 기록용

print("학습을 시작합니다!")

for episode in range(1, 1001):  # 1부터 시작하게 변경
    states, actions, rewards = run_episode(env, policy)
    returns = compute_returns(rewards)

    # REINFORCE: 좋은 행동의 확률을 높여라
    loss = 0
    for state, action, G in zip(states, actions, returns):
        probs = policy(state)
        log_prob = torch.log(probs[action])
        loss -= log_prob * G  # 좋은 행동 → loss 낮춤

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = sum(rewards)

    # 10판마다 로그 출력해서 지루하지 않게 개선
    if episode % 10 == 0:
        print(f"에피소드 {episode:4d} | 이번 판 스텝수: {total_reward:.0f} (역대최고: {best_reward:.0f})")

    # 역대 최고 점수를 깨거나 500점 만점을 찍으면 그 즉시 실시간 저장
    if total_reward > best_reward:
        best_reward = total_reward
        torch.save(policy.state_dict(), "cartpole_model.pth")
        print(f"기록 갱신! {total_reward:.0f}점 달성으로 'cartpole_model.pth' 저장 완료!")

        if total_reward >= 500:
            print("만점 달성! 이제 Ctrl+C로 끄고 test.py 돌리셔도 됩니다!")

env.close()
