import argparse
import glob
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv


# 1. 설정
ON_KAGGLE = os.path.exists("/kaggle/working")
BASE_DIR = "/kaggle/working" if ON_KAGGLE else "."


@dataclass
class Config:
    env_id: str = "Humanoid-v4"

    num_envs: int = 8  # CPU 코어 수에 맞춰 자동 조정
    total_timesteps: int = 10_000_000
    n_steps: int = 2048
    batch_size: int = 512
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    learning_rate: float = 3e-4

    net_arch: tuple = (256, 256)
    ortho_init: bool = True
    log_std_init: float = -0.5

    ckpt_dir: str = os.path.join(BASE_DIR, "checkpoints")
    tb_dir: str = os.path.join(BASE_DIR, "tb_logs")
    save_every: int = 500_000
    eval_every: int = 500_000
    eval_episodes: int = 5

    seed: int = 42


# 2. 콜백
class TrainLogCallback(BaseCallback):
    """주기적으로 ep_rew_mean 등을 stdout에 출력"""

    def __init__(self, log_every: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self._rollout_count = 0

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.log_every != 0:
            return
        stats = self.logger.name_to_value
        print(
            f"[{self.num_timesteps:>12,}]"
            f"  ep_rew={stats.get('rollout/ep_rew_mean', float('nan')):8.1f}"
            f"  ep_len={stats.get('rollout/ep_len_mean', float('nan')):6.0f}"
            f"  fps={stats.get('time/fps', 0):5.0f}"
            f"  kl={stats.get('train/approx_kl', float('nan')):.4f}",
            flush=True,
        )

    def _on_step(self) -> bool:
        return True


class TimeLimitCallback(BaseCallback):
    """지정한 시간(시간 단위)이 지나면 학습을 중단 (이후 main에서 저장)"""

    def __init__(self, max_hours: float, verbose: int = 1):
        super().__init__(verbose)
        self.deadline = time.time() + max_hours * 3600
        self.max_hours = max_hours

    def _on_step(self) -> bool:
        if time.time() >= self.deadline:
            print(f"\n[시간 제한] {self.max_hours}시간 경과. 학습을 중단하고 저장합니다.", flush=True)
            return False
        return True


class SaveVecNormOnBest(BaseCallback):
    """EvalCallback이 best 모델을 저장할 때 VecNormalize 통계도 함께 저장"""

    def __init__(self, venv: VecNormalize, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.venv = venv
        self.save_path = save_path

    def _on_step(self) -> bool:
        self.venv.save(self.save_path)
        return True


# 3. 체크포인트 유틸
def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """rl_model_{steps}_steps.zip 중 스텝 수가 가장 큰 파일 (문자열 정렬이 아닌 숫자 정렬)"""
    best, best_step = None, -1
    for f in glob.glob(os.path.join(ckpt_dir, "rl_model_*_steps.zip")):
        m = re.search(r"rl_model_(\d+)_steps\.zip$", os.path.basename(f))
        if m and int(m.group(1)) > best_step:
            best, best_step = f, int(m.group(1))
    return best


def vecnorm_path_for(ckpt_path: str) -> str:
    """rl_model_1000000_steps.zip -> rl_model_vecnormalize_1000000_steps.pkl"""
    d = os.path.dirname(ckpt_path)
    m = re.search(r"rl_model_(\d+)_steps\.zip$", os.path.basename(ckpt_path))
    step = m.group(1) if m else ""
    return os.path.join(d, f"rl_model_vecnormalize_{step}_steps.pkl")


def resolve_env_id(env_id: str) -> str:
    """Humanoid-v4가 등록 안 되어 있으면 v5로 대체"""
    try:
        gym.spec(env_id)
        return env_id
    except Exception:
        alt = env_id.rsplit("-v", 1)[0] + "-v5"
        print(f"[주의] {env_id} 를 찾을 수 없어 {alt} 로 대체합니다.")
        return alt


# 4. 메인
def main(cfg: Config, resume: bool, ckpt_path: Optional[str], device_arg: str, max_hours: Optional[float]):
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    print(f"[device] {device}")

    # num_envs를 CPU 코어 수에 맞춤
    n_cpu = os.cpu_count() or 4
    if cfg.num_envs > n_cpu:
        print(f"[CPU] 코어 {n_cpu}개 → num_envs {cfg.num_envs} → {n_cpu}")
        cfg.num_envs = n_cpu
    print(f"[config] num_envs={cfg.num_envs}, total_timesteps={cfg.total_timesteps:,}")

    cfg.env_id = resolve_env_id(cfg.env_id)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    vec_cls = SubprocVecEnv if cfg.num_envs > 1 else DummyVecEnv

    # ── 이어받기 대상 확인 ──
    load_path = None
    if resume:
        load_path = ckpt_path or find_latest_checkpoint(cfg.ckpt_dir)
        if not (load_path and os.path.exists(load_path)):
            print("[경고] 체크포인트 없음. 처음부터 시작합니다.")
            load_path = None
            resume = False

    # ── 학습 환경 (make_vec_env가 Monitor를 자동 적용) ──
    raw_env = make_vec_env(
        cfg.env_id, n_envs=cfg.num_envs, seed=cfg.seed, vec_env_cls=vec_cls
    )

    if resume:
        vn_path = vecnorm_path_for(load_path)
        if os.path.exists(vn_path):
            print(f"[resume] VecNormalize 불러오기: {vn_path}")
            train_env = VecNormalize.load(vn_path, raw_env)
            train_env.training = True
            train_env.norm_reward = True
        else:
            print("[주의] VecNormalize 통계 파일이 없습니다. 통계가 초기화되어 성능이 흔들릴 수 있습니다.")
            train_env = VecNormalize(raw_env, norm_obs=True, norm_reward=True,
                                     clip_obs=10.0, clip_reward=10.0, gamma=cfg.gamma)
    else:
        train_env = VecNormalize(raw_env, norm_obs=True, norm_reward=True,
                                 clip_obs=10.0, clip_reward=10.0, gamma=cfg.gamma)

    # ── 평가 환경 (obs 정규화 통계는 EvalCallback이 train_env와 동기화) ──
    eval_env = VecNormalize(
        make_vec_env(cfg.env_id, n_envs=1, seed=cfg.seed + 100),
        training=False, norm_reward=False, clip_obs=10.0, gamma=cfg.gamma,
    )

    # ── 모델 ──
    if resume:
        print(f"[resume] 모델 불러오기: {load_path}")
        model = PPO.load(load_path, env=train_env, device=device)
        print(f"[resume] {model.num_timesteps:,} 스텝에서 재개")
    else:
        policy_kwargs = dict(
            net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)),
            ortho_init=cfg.ortho_init,
            log_std_init=cfg.log_std_init,
        )
        model = PPO(
            "MlpPolicy", train_env,
            n_steps=cfg.n_steps, batch_size=cfg.batch_size, n_epochs=cfg.n_epochs,
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef, vf_coef=cfg.vf_coef, max_grad_norm=cfg.max_grad_norm,
            target_kl=cfg.target_kl, learning_rate=cfg.learning_rate,
            policy_kwargs=policy_kwargs, device=device, seed=cfg.seed,
            verbose=1, tensorboard_log=cfg.tb_dir,
        )

    remaining = cfg.total_timesteps - model.num_timesteps
    if remaining <= 0:
        print("[완료] 이미 목표 스텝에 도달했습니다.")
        train_env.close(); eval_env.close()
        return
    print(f"Humanoid PPO (SB3) | 남은 스텝: {remaining:,}", flush=True)

    # ── 콜백 (모델/환경 확정 후 생성) ──
    best_dir = os.path.join(cfg.ckpt_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(cfg.save_every // cfg.num_envs, 1),
        save_path=cfg.ckpt_dir,
        name_prefix="rl_model",
        save_vecnormalize=True,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=os.path.join(cfg.ckpt_dir, "eval_logs"),
        eval_freq=max(cfg.eval_every // cfg.num_envs, 1),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
        callback_on_new_best=SaveVecNormOnBest(train_env, os.path.join(best_dir, "vecnormalize.pkl")),
        verbose=1,
    )
    cbs = [checkpoint_cb, eval_cb, TrainLogCallback(log_every=10)]
    if max_hours:
        cbs.append(TimeLimitCallback(max_hours))
    callbacks = CallbackList(cbs)

    # ── 학습 ──
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n[중단] 사용자 인터럽트. 현재 상태를 저장합니다.")

    # ── 최종 저장 (resume 가능한 형식으로도 저장) ──
    n = model.num_timesteps
    model.save(os.path.join(cfg.ckpt_dir, "final_model"))
    train_env.save(os.path.join(cfg.ckpt_dir, "final_vecnormalize.pkl"))
    model.save(os.path.join(cfg.ckpt_dir, f"rl_model_{n}_steps"))
    train_env.save(os.path.join(cfg.ckpt_dir, f"rl_model_vecnormalize_{n}_steps.pkl"))
    print(f"\n[완료] {n:,} 스텝 | 저장 위치: {cfg.ckpt_dir}")

    train_env.close()
    eval_env.close()


# 5. 엔트리포인트
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 이어받기")
    parser.add_argument("--ckpt", type=str, default=None, help="특정 체크포인트 경로 (.zip)")
    parser.add_argument("--envs", type=int, default=None, help="num_envs override")
    parser.add_argument("--steps", type=int, default=None, help="total_timesteps override (이어받기 시 누적 목표)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max_hours", type=float, default=None, help="이 시간이 지나면 저장 후 종료")
    args = parser.parse_args()

    cfg = Config()
    if args.envs:
        cfg.num_envs = args.envs
    if args.steps:
        cfg.total_timesteps = args.steps

    main(cfg, resume=args.resume, ckpt_path=args.ckpt, device_arg=args.device, max_hours=args.max_hours)
