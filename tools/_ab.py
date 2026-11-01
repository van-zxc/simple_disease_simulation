"""同进程交替 A/B（历史工具，已被两进程的 tools/ab_iter.py 取代，勿用于验收）。

局限：两侧都要 import `disease_sim` 包，同进程里 sys.modules 会让第二侧复用第一侧
的模块——只有当对比的改动都在单个入口文件里（不涉及包）时才近似可用。要测包内部
改动请用 tools/ab_iter.py（每侧独立进程）；`.bak` 也不能直接加载，需先复制成 .py。

用法：python tools/_ab.py <profile> <population> <frames_per_round> <rounds> [warm]
环境变量：AB_BEFORE（默认 _orig_sim.py）、AB_AFTER（默认 simple_disease_simulation.py）。
每轮翻转两侧顺序以抵消热/时钟漂移；和所有性能数字一样只读同会话相对变化（±3% 噪声）。
"""
import os
import pathlib
import statistics
import sys
import time

# 复用 golden_check 的加载器与 patch_matplotlib，保证环境与其它工具一致。
sys.path.insert(0, str(pathlib.Path("tools").resolve()))
import golden_check as gc  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
BEFORE = pathlib.Path(os.environ.get("AB_BEFORE") or (ROOT / "_orig_sim.py"))
AFTER = pathlib.Path(os.environ.get("AB_AFTER") or (ROOT / "simple_disease_simulation.py"))

profile = sys.argv[1] if len(sys.argv) > 1 else "covid19"
pop = sys.argv[2] if len(sys.argv) > 2 else "10000"
frames = int(sys.argv[3]) if len(sys.argv) > 3 else 120
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
warm = 20  # 20 帧足以越过 kernel 首启；可用第 5 个位置参数覆盖。
if len(sys.argv) > 5:
    warm = int(sys.argv[5])

env = {"DISEASE_PROFILE": profile, "DISEASE_POPULATION": pop}
# 第二次 load 会命中第一次在 sys.modules 里留下的 disease_sim——这正是本工具只能做
# "单文件入口"对比的原因（见模块 docstring），也是它被 ab_iter.py 取代的原因。
mod_a, _, _ = gc.load_module(BEFORE, dict(env))
mod_b, _, _ = gc.load_module(AFTER, dict(env))
import torch  # noqa: E402

# 预热两边（含 CUDA kernel 首次启动），再从同一帧号交替测量
for mod in (mod_a, mod_b):
    for f in range(warm):
        mod.update(f)
torch.cuda.synchronize()

res_a, res_b = [], []
for r in range(rounds):
    # 每轮翻转 A/B 顺序：热漂移（时钟升频/降频）对两边的偏置方向相反，取中位数后近似抵消。
    order = (("a", mod_a), ("b", mod_b)) if r % 2 == 0 else (("b", mod_b), ("a", mod_a))
    for tag, mod in order:
        # 每轮接续上一轮末尾的帧号（状态继续演化）；同一轮内两侧跑同一段帧号，可比。
        start = warm + r * frames
        t0 = time.perf_counter()
        for f in range(start, start + frames):
            mod.update(f)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000 / frames
        (res_a if tag == "a" else res_b).append(dt)

ma, mb = statistics.median(res_a), statistics.median(res_b)
print(f"profile={profile} N={pop} frames/round={frames} rounds={rounds}")
print(f"before ({BEFORE.name}): {[f'{v:.2f}' for v in res_a]}  median={ma:.3f} ms/frame")
print(f"after  ({AFTER.name}) : {[f'{v:.2f}' for v in res_b]}  median={mb:.3f} ms/frame")
print(f"change: {(mb / ma - 1) * 100:+.1f}%  ({(mb - ma):+.3f} ms/frame)")

# 结构/统计一致性抽样：两边跑到同一帧号后的 counts 与标题（非逐位验收，仅作参考）
def _counts(mod):
    st = getattr(mod, "status")
    return [int(x) for x in torch.bincount(st, minlength=8).cpu().tolist()]


print("counts before:", _counts(mod_a))
print("counts after :", _counts(mod_b))
print("title before :", mod_a.title_obj.get_text().splitlines()[0])
print("title after  :", mod_b.title_obj.get_text().splitlines()[0])
