import argparse
import glob
import os
import sys

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="Humanoid-v4")
    p.add_argument("--dir", type=str, default=".",
                   help="모델/통계 파일이 들어있는 폴더 (하위 폴더까지 자동 검색)")
    p.add_argument("--model", type=str, default="final",
                   help="'final' | 'best' | .zip 파일 경로")
    p.add_argument("--vecnorm", type=str, default=None,
                   help="VecNormalize .pkl 경로 (생략 시 자동 검색)")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true",
                   help="확률적 행동 사용 (기본은 deterministic)")
    p.add_argument("--render", action="store_true",
                   help="MuJoCo 창으로 실시간 렌더링 (로컬 PC용)")
    p.add_argument("--video", type=str, default=None,
                   help="지정한 폴더에 mp4 영상 저장 (headless 환경용)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--fps", type=int, default=30,
                   help="저장 영상 fps (기본 30 = 약 0.45배속, 실시간은 약 67)")
    return p.parse_args()


# 1. 파일 탐색
def find_file(root: str, name: str):
    """root 이하에서 name 파일을 재귀 검색 (가장 얕은 경로 우선)"""
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    hits.sort(key=lambda x: (x.count(os.sep), x))
    return hits[0] if hits else None


def resolve_model(root: str, model_arg: str) -> str:
    if os.path.isfile(model_arg):
        return model_arg
    table = {"final": "final_model", "best": "best_model"}
    if model_arg in table:
        # 확장자 무관하게 검색
        stem = table[model_arg]
        hits = [
            p for p in glob.glob(os.path.join(root, "**", stem + "*"), recursive=True)
            if os.path.isfile(p) and not p.endswith(".pkl")
        ]
        hits.sort(key=lambda x: (not x.endswith(".zip"), x.count(os.sep), x))
        if hits:
            return hits[0]

    print(f"[오류] 모델 파일을 찾을 수 없습니다: {model_arg} (검색 폴더: {root})")
    print("[참고] 검색 폴더 안의 파일 목록:")
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True))[:50]:
        if os.path.isfile(p):
            print("   ", p)
    sys.exit(1)


def resolve_vecnorm(root: str, vn_arg):
    if vn_arg:
        if os.path.isfile(vn_arg):
            return vn_arg
        sys.exit(f"[오류] VecNormalize 파일이 없습니다: {vn_arg}")

    for name in ("final_vecnormalize.pkl", "vecnormalize.pkl"):
        path = find_file(root, name)
        if path:
            return path

    # 체크포인트용 vecnormalize_{step}_steps.pkl 중 가장 큰 step
    cands = glob.glob(os.path.join(root, "**", "vecnormalize_*_steps.pkl"), recursive=True)
    if cands:
        def step_of(f):
            s = os.path.basename(f).replace("vecnormalize_", "").replace("_steps.pkl", "")
            return int(s) if s.isdigit() else -1
        return max(cands, key=step_of)
    return None


# 2. 환경 생성
def make_env(args, vn_path):
    if args.render and args.video:
        sys.exit("[오류] --render 와 --video 는 동시에 사용할 수 없습니다.")

    render_mode = "human" if args.render else ("rgb_array" if args.video else None)

    def _make():
        env = gym.make(args.env, render_mode=render_mode)
        return Monitor(env)

    venv = DummyVecEnv([_make])
    venv.seed(args.seed)

    if vn_path:
        venv = VecNormalize.load(vn_path, venv)
        venv.training = False     # 통계 갱신 중지
        venv.norm_reward = False  # 보상은 원본 값으로 확인
    else:
        print("[경고] VecNormalize 파일 없음 → 정규화 없이 실행 (성능이 크게 떨어질 수 있음)")
    return venv


# 3. 메인
def main():
    args = parse_args()

    if args.video:
        # 화면 없는 서버에서 오프스크린 렌더링
        os.environ.setdefault("MUJOCO_GL", "egl")

    model_path = resolve_model(args.dir, args.model)
    vn_path = resolve_vecnorm(args.dir, args.vecnorm)
    print(f"[model]   {model_path}")
    print(f"[vecnorm] {vn_path}")

    if "best_model" in os.path.basename(model_path) and vn_path:
        print("[참고] best_model은 학습 도중 시점의 정규화 통계로 저장된 모델입니다. "
              "final 통계로 실행하면 미세한 차이가 있을 수 있습니다.")

    env = make_env(args, vn_path)
    model = PPO.load(model_path, env=env, device=args.device)

    deterministic = not args.stochastic
    rewards, lengths = [], []

    # 영상 저장 준비 (gymnasium RecordVideo 대신 직접 저장)
    writer = None
    if args.video:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio
        os.makedirs(args.video, exist_ok=True)

    def open_writer(idx):
        path = os.path.join(args.video, f"humanoid-episode-{idx}.mp4")
        return imageio.get_writer(path, fps=args.fps)

    obs = env.reset()
    ep_rew, ep_len = 0.0, 0
    if args.video:
        writer = open_writer(0)
        writer.append_data(env.get_images()[0])

    while len(rewards) < args.episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        ep_rew += float(reward[0])
        ep_len += 1

        if done[0]:
            rewards.append(ep_rew)
            lengths.append(ep_len)
            print(f"  episode {len(rewards):>2}/{args.episodes}"
                  f"  reward={ep_rew:9.1f}  length={ep_len:5d}")
            ep_rew, ep_len = 0.0, 0
            if writer is not None:
                writer.close()
                writer = None
                if len(rewards) < args.episodes:
                    writer = open_writer(len(rewards))
                    writer.append_data(env.get_images()[0])
        elif writer is not None:
            writer.append_data(env.get_images()[0])

    if writer is not None:
        writer.close()
    env.close()

    print("\n===== 결과 =====")
    print(f"reward 평균 : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"length 평균 : {np.mean(lengths):.0f}  (최대 1000)")
    if args.video:
        print(f"영상 저장   : {os.path.abspath(args.video)}")


if __name__ == "__main__":
    main()
