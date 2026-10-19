import gymnasium as gym
from stable_baselines3 import PPO

# 1. 환경 소환
env = gym.make("Humanoid-v4", render_mode="human")

# 2. 저장된 뇌세포(.zip)를 그대로 로드
model = PPO.load("humanoid_model_v1", env=env)

print("🍶 취권 마스터의 전성기 무대를 감상하십시오.. (종료는 터미널에서 Ctrl+C)")

# 3. 무한 루프로 춤 감상하기
obs, _ = env.reset()

# 렌더러 내부 MuJoCo 카메라 접근
mujoco_env = env.unwrapped
cam = mujoco_env.mujoco_renderer.viewer.cam
cam.distance = 4.0
cam.azimuth = 90
cam.elevation = -20

while True:
    # 저장된 가중치 매트릭스에서 최선의 action(관절 토크) 받아오기
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # 카메라가 휴머노이드 루트 따라오게
    cam.lookat[:] = mujoco_env.data.qpos[:3]

    if terminated or truncated:
        obs, _ = env.reset()
