import gymnasium as gym
import torch
import torch.nn as nn
import time

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )
    def forward(self, x):
        return self.net(x)

# 환경 생성 (화면 보기 모드)
env = gym.make("CartPole-v1", render_mode="human")

# 뇌세포 로드
policy = PolicyNet()
policy.load_state_dict(torch.load("cartpole_model.pth"))
policy.eval()  # 평가 모드 전환

print("달인의 묘기를 감상하십시오.. (10판 테스트)")

for episode in range(10):
    obs, _ = env.reset()
    total_reward = 0

    while True:
        state_t = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            probs = policy(state_t)

        # 주사위 압수! 무조건 가장 높은 확률의 액션만 선택
        action = torch.argmax(probs).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        time.sleep(0.01)

        if terminated or truncated:
            break

    print(f"테스트 에피소드 {episode+1} 결과: {total_reward:.0f}점!")

env.close()
