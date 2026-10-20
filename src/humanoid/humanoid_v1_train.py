from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. 환경 소환
env = make_vec_env("Humanoid-v4", n_envs=8)  # render_mode="human"

# 2. 중간 저장(체크포인트) 콜백 설정
checkpoint_callback = CheckpointCallback(
    save_freq=200000 // 8,
    save_path="./logs/",
    name_prefix="humanoid_ppo_chkpt",
)

# 3. 2족 보행 PPO 알고리즘 생성
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,     # 한 번에 수집하는 스텝 수
    batch_size=256,   # 디폴트(64)보다 크게 해서 안정적으로 학습
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,     # 너무 헛짓거리 못하게 탐색 다소 제한
)

# 4. 학습 개시
model.learn(total_timesteps=5000000, callback=checkpoint_callback)

# 5. 전성기 가중치 저장
model.save("humanoid_model_v1")
print("휴머노이드 뇌세포 저장 완료!")

env.close()
