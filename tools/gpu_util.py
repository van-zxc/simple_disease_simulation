"""纯 update 循环（无绘图）跑一段，供外部采样器量"单进程能吃满多少 GPU"。

用法：python tools/gpu_util.py [population] [frames]   # 默认 1000000 / 600
启动后先 warm 10 帧，再打印 START_SAMPLING，跑完打印 DONE_SAMPLING 与每帧 ms；
外部（如 `nvidia-smi --query-gpu=... -l 1`）只在这两个标记之间采样才有意义。
每帧 sync 是"单世界延迟"口径；绝对 util 要 ≥10 s 窗口才稳（SCALING.md §6.6）。
"""
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path("tools").resolve()))
import golden_check as gc  # noqa: E402

import torch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTRY = ROOT / "simple_disease_simulation.py"

# 默认取 1M（显存约 1 GB）与 600 帧（约 10–15 s），让外部采样有足够长的窗口。
pop = sys.argv[1] if len(sys.argv) > 1 else "1000000"
frames = int(sys.argv[2]) if len(sys.argv) > 2 else 600

mod, _, _ = gc.load_module(ENTRY, {"DISEASE_PROFILE": "covid19", "DISEASE_POPULATION": pop})
# warm 10 帧：越过 kernel 首启与显存分配，否则采样窗口里会混进一次偏慢的帧。
for f in range(10):
    mod.update(f)
torch.cuda.synchronize()
# 标记行供外部采样器对齐窗口：START 之前的加载/预热 util 不该算进均值。
print("START_SAMPLING", flush=True)
t0 = time.perf_counter()
# 每帧 sync：量的是"一帧端到端延迟"（CPU 派发 + GPU 执行），不是吞吐；
# 不加 sync 只量到入队耗时，util 看似吃满其实队列里在排队。
for f in range(10, 10 + frames):
    mod.update(f)
    torch.cuda.synchronize()
dt = (time.perf_counter() - t0) * 1000 / frames
print(f"DONE_SAMPLING per_frame={dt:.2f}ms frames={frames}", flush=True)
