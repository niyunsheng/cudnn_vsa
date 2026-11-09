# cudnn_vsa

[English](README.md) | [简体中文](README.zh-CN.md)

<!-- Keep README.md and README.zh-CN.md synchronized, including examples and benchmark data. -->

用于 DiT/MMDiT 训练的打包多模态注意力：
**cuDNN 块稀疏细粒度注意力（fine attention）+ FA4 变长粗粒度注意力（coarse attention）**。

## 安装

安装支持 CUDA 的 PyTorch 后，在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
```

## 使用

输入为相互独立、内存连续的 BF16 张量 `q`、`k`、`v`、`gate`，形状均为
`[T, H, 128]`。输出保持输入的 token 顺序。

```python
import torch
from vsa import vsa_attention
from vsa_layout import Segment, build_vsa_layout

# One 720p 5s video and two independent 720p images, each with text.
H, D = 2, 128
layout = build_vsa_layout(
    samples=[
        [Segment("video", (31, 45, 80), sparsity=0.9), Segment("text", 2048)],
        [Segment("video", (1, 45, 80), sparsity=None), Segment("text", 2048)],
        [Segment("video", (1, 45, 80), sparsity=None), Segment("text", 2048)],
    ],
    head_num=H,
    dense_query_full=False,
    device="cuda",
)
q, k, v, gate = [
    torch.randn(layout.tokens, H, D, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
    for _ in range(4)
]
out = vsa_attention(q, k, v, gate, layout)
out.backward(torch.randn_like(out))
```

每个内层列表是独立样本，attention 不跨样本。输入按声明的样本/segment 顺序排列，
视频内部使用 FHW raster 顺序。每个样本至少包含一个视频：`(F, H, W)` 表示
其 token 网格，`F=1` 表示图像。文本、音频和关键帧 segment 使用 token 数，
且 `sparsity=None`；也可添加更多视频/图像 segment。
几何形状和设置相同时，各层复用 layout。

将此目录加入 `PYTHONPATH`，或者把 `vsa.py`、`vsa_core.py`、
`vsa_layout.py` 和 `vsa_kernels.py` 复制到项目中并调整 import。

## 输入顺序

`layout.input_order` 显式声明 token 顺序：

| `input_order` | 输入 token | Slot 转换 |
| --- | --- | --- |
| `"raster"`（默认） | 原始样本/segment 顺序，视频按 FHW 顺序排列 | 重排为 block 并添加 padding |
| `"block"` | 按下述 block 顺序排列的有效 token，不含 padding | 仅添加 padding；本地所有 slot 都有效时跳过转换 |

### Raster 输入

默认的 `input_order="raster"` 接受原始 FHW 序列，如上面的示例所示。
`vsa_attention` 在内部进行重排和 padding，然后将输出恢复为相同的 raster 顺序。
无需预处理。

### Block 输入

Block 输入只包含按 block 排列的有效 token，不含 padding。
对于 raster 数据，在**整个层堆栈前重排一次**，后续各层保持此顺序：

```python
from vsa import raster_to_block_wo_padding

layout = build_vsa_layout(samples, head_num=H, input_order="block", device="cuda")
# [T_global, C] -> [T_global, C]: reorder rows, without padding.
tokens_block = raster_to_block_wo_padding(tokens_global, vsa_layout=layout)
# [T_global, 3] -> [T_global, 3]: preserve each (f, h, w) coordinate.
position_ids_block = raster_to_block_wo_padding(position_ids_global, vsa_layout=layout)
```

`input_order` 由调用方声明，VSA 不会检测张量的实际行顺序。
位置、target 和 mask 应使用相同的行映射。
3D RoPE 的 position IDs 形状为 `[T_global, 3]`；重排保持每行原始的 `(f, h, w)` 坐标不变。

## 注意力语义

`sparsity` 表示视频 segment 的 **KV block 丢弃比例**。
默认情况下，每个 query block 和 head 都独立地从**每个**视频 KV region 选择 top-k block，
包括跨视频注意力：

```text
k = max(1, ceil((1 - sparsity) * video_region_blocks))
```

预算由 KV segment 决定；选中的 block 由 Q/K 决定。
`sparsity=None` 将 region 设为稠密：它的 KV block 始终可见，仅进行 padding。
数值稀疏度使用 4×4×4 视频 tile；`0` 保留这些 tile 的全部 block。

[FastH3](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/fastvideo/attention/backends/video_sparse_attn_h3.py#L27-L31)
增加了 query 侧的例外规则：稠密 query 可以看到**全部** KV block。
`dense_query_full=True` 启用该例外；默认值 `False` 对每个 query 都应用视频 KV 稀疏度。
稠密 query 来自 `sparsity=None` 的 region。

下面给出一个样本/head 在稀疏度为 0.5 时的 fine mask 示例。
两个坐标轴都遵循内部 block 顺序：**先放 None-sparsity segment，再放数值稀疏度视频 segment**。
`■` 表示可见，`□` 表示被 mask。Top-k 选择仅用于示意。

**`dense_query_full=False`（默认）**

```text
           Dense KV    Video KV
           0 1         2 3 4 5
Dense Q0   ■ ■         □ ■ ■ □
Dense Q1   ■ ■         ■ □ □ ■
Video Q2   ■ ■         ■ □ ■ □
Video Q3   ■ ■         □ ■ □ ■
Video Q4   ■ ■         ■ ■ □ □
Video Q5   ■ ■         □ □ ■ ■
```

**`dense_query_full=True`（FastH3 稠密 query 规则）**

```text
           Dense KV    Video KV
           0 1         2 3 4 5
Dense Q0   ■ ■         ■ ■ ■ ■
Dense Q1   ■ ■         ■ ■ ■ ■
Video Q2   ■ ■         ■ □ ■ □
Video Q3   ■ ■         □ ■ □ ■
Video Q4   ■ ■         ■ ■ □ □
Video Q5   ■ ■         □ □ ■ ■
```

对于含数值稀疏度的样本，coarse attention 覆盖**所有 segment** 的 BF16 池化 block，
为每个 token 生成 `fine + gate * coarse[its_block]`。
如果样本的所有 segment 都是 `sparsity=None`，则通过 **BSA 执行完整注意力**，
不使用 coarse 分支。前面独立图像的处理方式就是如此；
VSA 样本内部的稠密图像仍会参与 coarse。

## Ring 上下文并行

将已有的 Ring 进程组传给同一个 layout 构建函数和 attention API：

```python
layout = build_vsa_layout(
    samples, head_num=H, input_order="block", device="cuda",
    ring_cp_group=group, ring_partition="cyclic",  # Or "contiguous".
)
# T_ring_local = layout.tokens: Ring-local valid token count; C is hidden size.
# Global [T_global, C] -> Ring-local [T_ring_local, C], reordered without padding.
tokens_local = raster_to_block_wo_padding(tokens_global, vsa_layout=layout)
# Global [T_global, 3] -> Ring-local [T_ring_local, 3], with the same row mapping.
position_ids_local = raster_to_block_wo_padding(position_ids_global, vsa_layout=layout)
# Project tokens_local into Q/K/V/gate, each [T_ring_local, H, D].
out = vsa_attention(q, k, v, gate, layout)  # [T_ring_local, H, D]
```

这里的 `H` 是 VSA 收到的 head 数。
`ring_cp_group=None` 创建完整的 `VSALayout`（`layout.ring is None`），
无需初始化分布式环境，也支持 CPU。传入成员进程组会创建本地 Ring layout，
rank 和 size 从进程组获取。非成员句柄会被拒绝；即使组大小为 1 也要求 block 输入。

进程组只创建一次，在不同 micro-batch 的 layout 之间共享；每个 layout 在层间复用。
组内所有成员必须使用相同的全局样本形状、head 数和分配策略。
前反向共用同一个 `ring_cp_group`。

### Block 归属

Ring 模式调用 `vsa_attention` 时，Q/K/V/gate 必须符合 layout 定义的本地 block 顺序：
仅包含本 rank 拥有的 block 的有效 token，按全局 block ID 递增排列，不含 padding。
`raster_to_block_wo_padding` 融合分片选择与重排，不物化完整的全局 block 中间张量，也不通信。

`ring_partition` 独立于 `input_order` 选择归属方式，结果记录在
`layout.ring.partition` 中：

| `ring_partition` | 全局 block 归属 |
| --- | --- |
| `"cyclic"`（默认） | Block `b` 属于 rank `b % R`，全局编号跨样本连续 |
| `"contiguous"` | 每个样本给每个 rank 至多一个连续 block 区间 |

连续分配从 rank 0 开始。对每个样本的 N 个 block，计算
`base, extra = divmod(N, R)`：从当前起始 rank 循环访问各 rank，前 `extra` 个
各获得 `base + 1` 个连续 block，其余各获得 `base` 个。
随后起点前进 `extra` 并对 R 取模，使下一个样本从上一个样本分得较少的第一个
rank 开始。`extra=0` 时起点不变。

两种策略均保证每个样本及累计的 block 数相差不超过一个，
但不保证有效 token 数或动态 top-k 工作量均衡。循环分配将相邻 block 分散到不同 rank；
连续分配让每个样本的本地区间便于切片。两者使用相同的 attention kernel。
修改策略时必须重建 layout 并重新分片输入。

整个 pack 至少需要 R 个 block，使每个 Ring rank 都有有效数据。
单个样本的 token/block 数无需被 R 整除，也可以不出现在部分 rank 上。
内部每个 rank 分配 `C = ceil(global_blocks / R)` 个 block，每个 block 有
64 个 slot；不完整和 dummy slot 不参与数学计算。

### 示例：三个 rank 上的两个样本

`R=3`，样本 A 包含文本 block `b0` 和视频 block `b1..b4`（`sparsity=0.5`）。
样本 B 有两个 block：`b5` 存放稠密图像 token，`b6` 存放文本 token。
全局 block ID 跨样本连续编号。

| 策略 | 样本 / 数量 | Rank 0 | Rank 1 | Rank 2 |
| --- | --- | --- | --- | --- |
| `contiguous` | A | `b0, b1` | `b2, b3` | `b4` |
| `contiguous` | B | `b6` | — | `b5` |
| `contiguous` | 累计 block 数 | 3 | 2 | 2 |
| `cyclic` | A | `b0, b3` | `b1, b4` | `b2` |
| `cyclic` | B | `b6` | — | `b5` |
| `cyclic` | 累计 block 数 | 3 | 2 | 2 |

`contiguous` 下，A 从 rank 0 开始，向 rank 0/1/2 分配 `[2, 2, 1]` 个 block。
B 从分得较少的 rank 2 开始，依次给 rank 2、0 各一个 block；下一个样本从 rank 1 开始。

### 路由与 top-k

1. 按有效 token 计算本地 block 均值，all-gather K 均值并恢复全局 block 顺序，
   使本地 Q 可以跨所有 Ring rank 选择 KV。
2. 默认每个 query block/head 用 FP32 `Q_mean @ K_mean.T` 打分，
   在同一样本的每个视频 region 内独立选 top-k。
   每个 region 保留 `max(1, ceil((1 - sparsity) * region_blocks))` 个 block。
3. 将选中的全局 block ID 映射到所属 rank 和本地 block ID，供 fine attention 使用。
   稠密 KV 始终保留，padding 和其他样本被排除；离散选择不参与求导。

样本 A 有四个视频 block，`sparsity=0.5` 时 `keep=2`。
若 top-k 选中 `b2, b4`，加上文本 block `b0`，最终得到 `[b0, b2, b4]`。

### 执行流程

| 阶段 | 操作 |
| --- | --- |
| Block 汇总 | 计算本地 Q/K/V 均值，all-gather **FP32 K + BF16 V**。 |
| Fine attention | Q 保持本地，整个 K/V 分片沿 Ring 轮转；计算选中的 block，并在全部选中 key 上用统一的 softmax 归一化合并各步结果。 |
| Coarse attention | 本地 Q 均值关注同一样本的全部 K/V 均值，随后合并 `fine + gate * coarse[query_block]`；完全稠密的样本仅使用 fine。 |
| 反向 | 反向遍历 Ring，以 FP32 累加 dQ/dK/dV，并通过 FP32 SUM reduce-scatter 归约池化梯度。 |

`vsa_attention` 内部完成结果合并，输出为按本地 block 顺序排列的有效 token。
P2P 调用方式（`ring_cp_batch_p2p`）和反向计算分桶大小（`ring_cp_bwd_bucket_size`）
的定义见 [`build_vsa_layout`](vsa_layout.py)。

### Ring 通信

在并行初始化时创建一次普通优先级 NCCL 进程组，然后在不同 layout 和层之间复用：

```python
import torch.distributed as dist

# Call on all WORLD ranks in the same group-creation order, before building layouts.
group = dist.new_group(ranks=ring_ranks, backend="nccl")
layout = build_vsa_layout(
    samples, head_num=H, input_order="block", device="cuda",
    ring_cp_group=group,
)
out = vsa_attention(q, k, v, gate, layout)
```

同一个进程组承担前反向 P2P 和池化 collective。
VSA 在 fine attention 之前将 P2P 入队，以重叠通信和计算；
实际重叠程度取决于 kernel 的资源占用，需要通过测量确认。

### 设计原因

- **完整 block、紧凑输入**：保持 pooling/路由单元完整，避免投影和 MLP 处理 attention padding。
- **汇总均值、轮转 K/V**：支持全局路由和 coarse attention，无需每个 rank 保存完整的 token 级 K/V。
- **统一缓冲区大小**：按 Ring 组内最多的 block 数补齐内部缓冲区，保证 P2P 形状一致，支持 CUDA Graph 复用。

Fine attention 仍传输完整的带 padding K/V 分片，因此稀疏度减少计算量，不减少 Ring 传输量。
两种归属策略应在实际工作负载中比较。

## 无 CP、Ulysses CP 和混合 CP 示例

下面的示例使用两个带残差连接的 attention 层，并执行一次优化器更新。
三种模式使用同一个 VSA API。整个进程 world 是一个上下文并行组；
`H` 必须能被 Ulysses 大小 `U` 整除，world size 必须等于 `U * R`。

| 模式 | 层堆栈之前的输入准备 | 传入 VSA 的 Q/K/V/gate |
| --- | --- | --- |
| 无 CP（`U=1, R=1`） | 保持 raster 顺序 | 全部 token 和全部 head |
| 仅 Ulysses（`U>1, R=1`） | 将 raster 序列切成 U 个 token 分片 | All-to-All 后的全部 token 和 H/U 个 head |
| Ring + Ulysses（`U>1, R>1`） | 一次性选择并重排 Ring 拥有的完整 block，再将每个紧凑 Ring 序列切成 U 个 token 分片 | All-to-All 后本 Ring 分片的有效 token 和 H/U 个 head |

Ulysses All-to-All 在 VSA 执行前恢复 token 序列：无 Ring 时为 raster 顺序，
有 Ring 时为紧凑 block 顺序。
示例中的 `[start:start + A]` 是 Ulysses 的 token 切片，与 `ring_partition` 无关，
切片可以跨越 block 边界。
为使 Ulysses 消息大小一致，只在序列尾部添加**传输 padding**；
进入 attention 前将其移除，计算 loss 时也排除这些行。
这与 VSA 内部每个 block 的 64-slot padding 不同。

### 无 CP

将后面的共用 Python 示例保存为本仓库中的 `demo_cp.py`，然后运行：

```bash
python demo_cp.py
```

### 仅 Ulysses CP

与无 CP 示例一样，直接使用 raster 输入。四张 GPU 在层堆栈前切分 token，
每层内部通过交换将 token 分片转换为 head 分片，无需预先重排输入：

```bash
torchrun --standalone --nproc-per-node=4 demo_cp.py --ulysses-size 4
```

### Ring + Ulysses CP

八张 GPU 使用 `U=4, R=2` 时，Ulysses 组为 `[0,1,2,3]` 和
`[4,5,6,7]`；Ring 组为 `[0,4]`、`[1,5]`、`[2,6]`、`[3,7]`。
每对 Ring rank 处理相同的 head 分片，分别拥有互补的 block。
两种 block 归属策略使用相同的输入准备和 attention 调用：

```bash
torchrun --standalone --nproc-per-node=8 demo_cp.py \
  --ulysses-size 4 --ring-size 2 --ring-partition cyclic
torchrun --standalone --nproc-per-node=8 demo_cp.py \
  --ulysses-size 4 --ring-size 2 --ring-partition contiguous
```

### 共用示例

示例使用 PyTorch 支持自动求导的 All-to-All、复制的模型权重，以及每个 rank 上
相同的完整 raster 数据。Ulysses、投影、loss 归一化和梯度归约属于调用方代码。
为便于阅读，示例先物化 Ring 分片再切 Ulysses；数据加载也可以合并两种映射，
直接提取最终的本地行。

<details>
<summary>完整 Python 示例：demo_cp.py</summary>

```python
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.nn.functional import all_to_all_single

from vsa import raster_to_block_wo_padding, vsa_attention
from vsa_layout import Segment, build_vsa_layout


def exchange_heads(mixed, group):
    # [A,H,4,D] -> four [U*A,H/U,D] tensors; backward exchanges gradients.
    if group is None:
        return mixed.unbind(2)
    u = dist.get_world_size(group)
    a, h, n, d = mixed.shape
    send = mixed.reshape(a, u, h // u, n, d).transpose(0, 1).contiguous()  # [U,A,H/U,4,D]
    recv = all_to_all_single(torch.empty_like(send), send, group=group)  # [U,A,H/U,4,D]
    return recv.reshape(u * a, h // u, n, d).unbind(2)


def exchange_tokens(out, group):
    # [U*A,H/U,D] -> [A,H,D], reversing the forward head exchange.
    if group is None:
        return out
    u = dist.get_world_size(group)
    t, h, d = out.shape
    send = out.reshape(u, t // u, h, d).contiguous()  # [U,A,H/U,D]
    recv = all_to_all_single(torch.empty_like(send), send, group=group)  # [U,A,H/U,D]
    return recv.permute(1, 0, 2, 3).reshape(t // u, u * h, d)


parser = argparse.ArgumentParser()
parser.add_argument("--ulysses-size", type=int, default=1)
parser.add_argument("--ring-size", type=int, default=1)
parser.add_argument("--ring-partition", choices=("cyclic", "contiguous"), default="cyclic")
args = parser.parse_args()
U, R, H, D = args.ulysses_size, args.ring_size, 8, 128
input_order = "block" if R > 1 else "raster"
C = H * D  # Hidden size: 1024.
world = int(os.environ.get("WORLD_SIZE", "1"))
if U < 1 or R < 1 or H % U or world != U * R:
    parser.error("Require U >= 1, R >= 1, H % U == 0 and WORLD_SIZE == U * R")
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
torch.cuda.set_device(device)
rank = 0
if world > 1:
    dist.init_process_group("nccl")
    rank = dist.get_rank()

# All ranks create all subgroups in the same order, once before the layers.
ulysses_group = ring_group = None
if U > 1:
    for r in range(R):
        group = dist.new_group(list(range(r * U, (r + 1) * U)))
        if rank // U == r:
            ulysses_group = group
if R > 1:
    for u in range(U):
        group = dist.new_group([r * U + u for r in range(R)])
        if rank % U == u:
            ring_group = group

samples = [
    [Segment("video", (5, 9, 11), sparsity=0.9), Segment("text", 67)],
    [Segment("video", (1, 3, 5)), Segment("text", 7)],
]
T = sum(segment.tokens for sample in samples for segment in sample)  # T_global = 584.
layout = build_vsa_layout(
    samples, head_num=H // U, input_order=input_order, device=device,
    ring_cp_group=ring_group, ring_partition=args.ring_partition,
)
# T_ring_local = layout.tokens: Ring-local valid token count (T if R=1); H is total heads.
# T_ring_local can differ across Ring ranks; it need not equal T/R.
torch.manual_seed(1234)  # Identical mock input and replicated weights on all ranks.
tokens_global = torch.randn(T, C, device=device, dtype=torch.bfloat16)  # [T,C] = [584,1024]
# Example 3D RoPE convention: text uses (1..L, 0, 0), video time starts at L+1.
position_rows = []
for video, text in samples:
    axes = [torch.arange(n, device=device) for n in video.size]
    axes[0] += text.tokens + 1
    video_ids = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    text_ids = torch.zeros(text.tokens, 3, device=device, dtype=torch.long)
    text_ids[:, 0] = torch.arange(1, text.tokens + 1, device=device)
    position_rows.extend((video_ids, text_ids))
position_ids_global = torch.cat(position_rows)  # [T,3] = [584,3]: original (f, h, w).

# Ring sharding + reorder: layout.ring.partition selects cyclic/contiguous blocks.
tokens_attention = tokens_global
position_ids_attention = position_ids_global
if R > 1:
    # Global [T,C] -> Ring-local [T_ring_local,C]: valid tokens in compact block order.
    tokens_attention = raster_to_block_wo_padding(tokens_global, vsa_layout=layout)
    # Global [T,3] -> Ring-local [T_ring_local,3]: select/reorder the same token rows.
    position_ids_attention = raster_to_block_wo_padding(position_ids_global, vsa_layout=layout)

# Ulysses preparation: equal message sizes, then U token slices of this Ring shard.
# Tail padding only; per-block padding is added inside vsa_attention.
max_tokens = layout.tokens
if layout.ring is not None:
    max_tokens = int(layout.ring.peer_block_sizes.sum(dim=1).max().item())
A = (max_tokens + U - 1) // U  # Per-GPU local rows after Ulysses slicing, including padding.
tail = U * A - layout.tokens  # U*A - T_ring_local.
start = (rank % U) * A
# Ring-local [T_ring_local,C] -> padded [U*A,C] -> per-GPU local [A,C].
x = F.pad(tokens_attention, (0, 0, 0, tail))[start:start + A].contiguous()
# Ring-local [T_ring_local,3] -> padded [U*A,3] -> per-GPU local [A,3].
position_ids_local = F.pad(position_ids_attention, (0, 0, 0, tail), value=-1)[start:start + A]
valid = torch.arange(start, start + A, device=device) < layout.tokens  # [A]

layers = torch.nn.ModuleList([
    torch.nn.ModuleList([
        torch.nn.Linear(C, 4 * C, bias=False, device=device, dtype=torch.bfloat16),
        torch.nn.Linear(C, C, bias=False, device=device, dtype=torch.bfloat16),
    ]) for _ in range(2)
])
optimizer = torch.optim.SGD(layers.parameters(), lr=1e-3)
for qkvg_projection, output_projection in layers:
    mixed = qkvg_projection(x).view(A, H, 4, D)  # [A,C] -> [A,4*C] -> [A,H,4,D]
    # All-to-All: four [U*A,H/U,D] tensors; trim each to [T_ring_local,H/U,D].
    q, k, v, gate = (
        t[:layout.tokens].contiguous() for t in exchange_heads(mixed, ulysses_group)
    )
    # RoPE uses position_ids_attention [T_ring_local,3] for these unpadded Q/K rows.
    # VSA adds/removes block padding internally; its public shapes stay compact.
    out = vsa_attention(q, k, v, gate, layout)  # [T_ring_local,H/U,D]
    out = F.pad(out, (0, 0, 0, 0, 0, tail))  # [U*A,H/U,D]
    out = exchange_tokens(out, ulysses_group).reshape(A, C)  # -> [A,H,D] -> [A,C]
    x = (x + output_projection(out)).masked_fill(~valid[:, None], 0)  # [A,C]

# Sum local contributions to a global-token-normalized illustrative loss.
loss = x.float().square().masked_fill(~valid[:, None], 0).sum() / (T * C)
loss.backward()
if world > 1:
    # SUM, not average: each local loss already uses the global denominator.
    for parameter in layers.parameters():
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
optimizer.step()
reported_loss = loss.detach()
if world > 1:
    dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)
if rank == 0:
    print(f"U={U}, R={R}, order={input_order}, loss={reported_loss.item():.6f}")
if world > 1:
    dist.destroy_process_group()
```

</details>

本例中，每层将 `[A, 1024]` 投影为形状为 `[A, 8, 128]` 的 Q/K/V/gate。
Ulysses 交换并移除传输 padding 后，每个张量的形状是
`[layout.tokens, 8/U, 128]`；逆向交换和输出投影将其恢复为 `[A, 1024]`。
无 CP 时，`A = layout.tokens = 584`。
输出在各层之间保持本 rank 的输入顺序。对 target 和 mask 应用相同的行选择；
保留原始 position 值，不要对本地分片重新编号。
只有应用在层堆栈结束后需要完整全局张量时，才收集各 rank 的输出 token 分片；
attention 内部的 Ring 结果合并已由 VSA 完成。
使用训练框架的参数梯度归约时，应以其替代示例中的手动归约，避免重复归约梯度。

## 性能

MFU 统计有效注意力 GEMM FLOPs；反向 FLOPs 是前向的两倍。
Padding、路由和非 GEMM 操作不计入 FLOP 分子。
单 GPU TFLOP/s 用全局 FLOPs 除以延迟和 GPU 数量；
MFU 再将该值除以稠密 BF16 峰值（`--peak-tflops`）。各阶段的中位数独立计算。
一次性输入分片/重排、graph capture 和 profiling 不计入计时；
注意力内部的 slot 转换和 padding 计入计时。

环境：GB200（2500 TFLOP/s BF16）、PyTorch 2.12.0a0/NV26.04、CUDA 13.2、cuDNN 9.22 / frontend 1.28.0、Triton 3.6、CuTe 4.6.2、FA4 4.0.0b35。

`benchmark.perf_vsa` 支持普通和分布式工作负载（`--ring`）。
两种模式共享输入生成、数值检查、graph capture 和计时逻辑。
`--cases` 选择工作负载，`--heads` 选择 head 数，`--check-only` 执行参考实现对比
和保留计算图后的再次反向检查，`--output` 接受 JSON 文件或目录。
`vsa-perf/v5` JSON 使用 `<case>-h<heads>` 作为 key；每个 case 记录检查结果、
全局有效 FLOPs，以及单 GPU 吞吐/MFU。
Ring 计时先取每次 replay 的各 rank 最大延迟，再汇总。Ring 默认使用 H=2，且不生成 trace；
添加 `--profile` 可为每个 rank 保存一份 trace。
比较性能时，在两种模式中显式使用相同的 `--seed`：
普通 VSA 的默认值为 2026，Ring 为 937。
相同的 seed、几何形状和 head 数会生成相同的全局 Q/K/V/gate 张量及上游输出梯度（dY）。

### 单设备 720p5s：FA4 完整注意力与 VSA

Benchmark 打包一个 720p、5 秒视频和两个独立的 720p 图像，每个样本都带文本：
视频有 121 帧，时间压缩 4×、空间压缩 16×，每个样本有 2048 个文本 token，
总共 124944 个 token，H=2/4，D=128，视频稀疏度为 0.9，
`dense_query_full=False`。VSA 使用默认的 `input_order="raster"`。

```bash
# VSA forward/backward time, MFU and torch traces.
python -m benchmark.perf_vsa --peak-tflops 2500
# FA4 full vs VSA, alternating order on the same GPU.
python -m benchmark.compare --peak-tflops 2500 --warmup 30 --repeats 500 --rounds 2
```

两者都使用 CUDA Graph，并对照 eager 执行验证 replay。
VSA 的计时覆盖 slot 转换、pooling、路由、fine、coarse、恢复和合并。
`benchmark/results/perf_vsa/` 包含 `perf.json` 和
`perf-default-h{2,4}-rank0.trace.json`；trace 在计时之后采集。
使用 `--input-order block` 测量紧凑 block 输入，使用 `--eager` 进行 eager 执行，
或使用 `--no-profile` 不采集 trace。输入生成/重排不计入计时。
对比结果写入 `benchmark/results/compare.json`。

| 方法 | H | 前向 ms | 前向 MFU | 反向 ms | 反向 MFU | 前向+反向 ms | 前向+反向 TFLOP/s | 前向+反向 MFU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FA4 完整注意力 | 2 | 9.357 | 56.82% | 25.409 | 41.85% | 34.772 | 1146.72 | 45.87% |
| cuDNN VSA（raster） | 2 | 2.366 | 24.81% | 6.746 | 17.40% | 9.114 | 482.91 | 19.32% |
| FA4 完整注意力 | 4 | 19.001 | 55.96% | 50.988 | 41.71% | 70.019 | 1138.94 | 45.56% |
| cuDNN VSA（raster） | 4 | 4.665 | 25.15% | 13.352 | 17.57% | 18.017 | 488.36 | 19.53% |

使用 CUDA Graph，每次测量预热 30 次，每种方法测两轮，每轮计时 replay 500 次。
FA4 完整注意力和 VSA 在同一 GPU 上交替执行。
每个时间列都是独立计算的 1000 次 replay 中位数。
这些是独立算子测量；VSA 时间包含 raster-to-slot 重排和 padding。

### 单 head 720p44s：普通 VSA 与 Ring4

`thd720p44s` 打包一个 `265 x 45 x 80` 视频（稀疏度 0.9）和一个独立的
`1 x 45 x 80` 图像，每个样本带 2048 个文本 token：**共 961696 个 token**。
三种模式均使用 BF16、H=1、D=128、紧凑 block 输入、seed 937 和
`dense_query_full=False`。计时覆盖 VSA 前反向及 Ring 通信。

在同一次分配的四张 GB200 上，先用单卡无 Ring 运行完整序列，
再比较使用普通优先级 NCCL 流的两种 Ring4 策略：

```bash
vsa_args=(--heads 1 --cases thd720p44s --input-order block --seed 937
          --warmup 10 --repeats 50 --no-profile --peak-tflops 2500)
for vsa_run in single_0 cyclic_0 contiguous_0 contiguous_1 cyclic_1 single_1; do
  vsa_mode="${vsa_run%_*}"
  if [ "$vsa_mode" = single ]; then
    CUDA_VISIBLE_DEVICES=0 python -m benchmark.perf_vsa "${vsa_args[@]}" \
      --output "benchmark/results/720p44s/$vsa_run.json"
  else
    torchrun --standalone --nproc-per-node=4 --module benchmark.perf_vsa \
      --ring --ring-size 4 --ring-partition "$vsa_mode" --batch-p2p \
      --bwd-bucket-size backend "${vsa_args[@]}" \
      --output "benchmark/results/720p44s/$vsa_run.json"
  fi
done
```

| 模式 | 前向 ms | 前向 MFU | 反向 ms | 反向 MFU | 前向+反向 ms | 前向+反向 TFLOP/s/GPU | 前向+反向 MFU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 无 Ring（单卡） | 67.460 | 25.17% | 185.257 | 18.33% | 252.572 | 504.15 | 20.17% |
| Ring4（连续） | 18.077 | 23.48% | 48.780 | 17.40% | 66.356 | 479.74 | 19.19% |
| Ring4（循环） | 17.914 | 23.69% | 49.173 | 17.26% | 66.733 | 477.03 | 19.08% |

按上述顺序，每种模式运行两次，每次预热 10 次、计时 CUDA Graph replay 50 次。
每列独立取两次运行中位数的中位数；Ring 每次 replay 使用各 rank 的最大延迟。
所有参考实现对比和 graph 检查均通过。

相对单卡，完整前反向加速比为**连续 3.806x / 循环 3.785x**；
并行效率 `T1 / (4 * T4)` 为 **95.16% / 94.62%**。
这些单机随机输入结果包含通信和 kernel 形状变化的影响；
实际模型路由及跨主机部署可能改变效率。

## 测试

测试覆盖两种输入顺序与独立 token 级数学实现的对比、
不同行形状/dtype 的融合单张量重排往返、padding 及其伴随操作，
以及使用原始和修改后输入的重排/注意力/恢复 CUDA Graph replay。
CUDA 不可用时跳过 GPU 测试。

```bash
python -m pip install 'pytest>=8'
python -m pytest -q -m 'not fastvideo'

# Optional FastH3 reference
python -m pip install -r tests/requirements-fastvideo.txt
python -m pytest -q tests/test_fastvideo.py
```

进行 Ring 验证时，所有 worker 都必须具备 CUDA 和 NCCL。
小规模 case 将输出及 Q/K/V/gate 梯度与独立 token 级数学实现对比；
64k case 则与单设备 VSA 对比。
计时之前会检查保留计算图后的再次反向，以及使用原始/修改后输入的 CUDA Graph replay。
添加 `--dense-query-full` 重复运行，可验证另一种稠密 query 策略。

```bash
# Ragged samples and partial/dummy slots; every rank owns valid blocks.
torchrun --standalone --nproc-per-node=8 --module benchmark.perf_vsa --ring \
  --cases ragged
# Interleaved groups; partial_blocks has three blocks and requires R<=3.
torchrun --standalone --nproc-per-node=8 --module benchmark.perf_vsa --ring \
  --ring-size 2 --cases ragged partial_blocks dense
# Device-bound groups and batched P2P.
torchrun --standalone --nproc-per-node=4 --module benchmark.perf_vsa --ring \
  --eager-group --cases ragged dense
# Single/packed 64k workloads, plus timing after numerical checks.
torchrun --standalone --nproc-per-node=8 --module benchmark.perf_vsa --ring \
  --cases single64k thd64k --warmup 30 --repeats 500
# Strict reproduction of the known fine-only-sample numerical failure.
torchrun --standalone --nproc-per-node=8 --module benchmark.perf_vsa --ring \
  --ring-size 2 --cases fine_only_peer --check-only
```

Ring 测试使用 block 输入。默认 case 为 `ragged`、`single64k` 和 `thd64k`；
每个 case 至少需要 R 个全局 block。小规模 case `partial_blocks`、
`fine_only_peer` 和 `dense` 分别包含 3、2 和 6 个 block。
多节点运行时，将 `--standalone` 替换为 torchrun rendezvous 参数。
JSON 记录数值/graph 检查和源码 hash；计时与 MFU 遵循**性能**一节的定义。
长序列 profile 使用 `thd720p44s`。

严格的 BF16 `fine_only_peer` case 仍有未解决的数值失败：
六 token、仅使用 fine 分支的样本，其 dQ 和 dK 相对原生参考实现的相似度
可能低于 0.99999 阈值。上面的命令会报告此失败；case 和阈值保持不变。

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `vsa.py` | 公共注意力入口，以及显式的 block/raster 重排辅助函数 |
| `vsa_core.py` | 路由、cuDNN BSA/FA4 调用、单设备执行，以及 Ring 通信和 autograd |
| `vsa_layout.py` | 静态样本/block 几何信息、行映射、路由模板和 coarse 边界 |
| `vsa_kernels.py` | 融合 token 重排、slot gather、Q/K/V padding、FP32 pooling 和门控合并 kernel |

两种输入顺序共享 core 中的注意力数学计算、各 region 的预算，以及先 fine 后 coarse 的调度。
Core 不导入公共入口。

<details>
<summary>精确的 block 顺序与小例子</summary>

1. 保持样本顺序。每个样本内先放 `sparsity=None` 的 segment，再放数值稀疏度的视频，
   每类内部保持 segment 的声明顺序。
2. None 稀疏度的 segment 按连续最多 64 个 token 分块；数值稀疏度的视频按
   4×4×4 tile 分块。Tile 起点和 tile 内有效 token 均按 F/H/W 顺序遍历
   （W 最快变化），边缘 tile 直接截断，不补 padding。
3. Block 不跨 segment 或样本边界。数值 `sparsity=0` 仍使用 tile；
   `sparsity=None` 使用稠密连续块。Ring 按全局 block ID 递增顺序保留拥有的 block，
   每个 block 内部顺序不变。

对于 `video=(1, 2, 5), sparsity=0.9` 后接三个文本 token，raster 行
`0..9` 是视频，`10..12` 是文本。Block 序列为：

| Block | 按输入顺序排列的原始 raster 行 | 有效 token 数 |
| --- | --- | ---: |
| b0：文本 | `10, 11, 12` | 3 |
| b1：第一个视频 tile | `0, 1, 2, 3, 5, 6, 7, 8` | 8 |
| b2：边缘视频 tile | `4, 9` | 2 |

输入只有 13 行，attention 内部分配 3×64 个 slot。将这个视频的稀疏度改成
`None` 后，保留视频在文本之前的顺序，产生大小为 10 和 3 的两个 block。
`layout.input_rows` 将每个输入行映射回全局 raster 行；
`raster_to_block_wo_padding(x, vsa_layout=layout)` 的值等于 `x[layout.input_rows]`。

</details>

<details>
<summary>重排算子与 layout 映射</summary>

| 转换 | 执行位置 | 前向 / 输出恢复 |
| --- | --- | --- |
| Raster → 带 padding 的 block | 每次使用 raster 输入的 VSA 调用内部 | Q/K/V 分别调用 `raster_to_block_w_padding`；`block_w_padding_to_raster` 恢复 raster 输出 |
| Raster → 紧凑 block | 层堆栈之前执行一次 | `raster_to_block_wo_padding`，Ring 时同时选择拥有的 block；`block_wo_padding_to_raster` 恢复完整 layout |
| 紧凑 block → 带 padding 的 block | 每次使用 block 输入的 VSA 调用内部 | `block_wo_padding_to_block_w_padding`；`block_w_padding_to_block_wo_padding` 去掉输出 padding |

名称注明源顺序、目标顺序和 padding：`w_padding` 表示带 block padding，
`wo_padding` 表示不带。带 `_qkv` 后缀的
`block_wo_padding_to_block_w_padding_qkv` 融合三个张量，供 `vsa_attention` 使用；
无后缀的版本转换单个张量。融合版让 Q/K/V 共用一次映射读取和 kernel 调用，
反向也如此；分别处理则需要三次调用。融合范围是 padding/恢复，不包含 QKV 线性投影。
Slot 辅助函数位于 `vsa_kernels.py`，
两个 raster/紧凑 block 边界转换函数位于 `vsa.py`。

边界转换 `raster_to_block_wo_padding` 接受一个内存连续的 CUDA
`[T_global, ...]` 张量，返回 `[layout.tokens, ...]`，保持 dtype 和后续维度。
它不添加 padding，Ring 时同时选择拥有的行。两个边界转换函数都要求
`input_order="block"`：layout 描述 VSA 的输入顺序。
仅对完整 layout，可在层堆栈结束后调用
`block_wo_padding_to_raster(out_block, vsa_layout=layout)` 恢复 raster 输出。

逐层转换在 `vsa_attention` 内部执行，位于 Ulysses head 交换之后（若使用 Ulysses）。
转换由输入顺序决定，Ulysses 本身不要求其中某种顺序。Ring 要求紧凑 block 输入。
这里的 padding 是 VSA 内部每块 64 个 slot 的 padding，与调用方的 Ulysses 尾部
padding 分开处理。

三类转换均支持 autograd。Slot 转换的反向使用逆行映射，紧凑 Q/K/V 的梯度融合恢复。
层堆栈边界重排的反向将梯度 scatter 到原始 raster 形状。
Ring 未拥有的行梯度为零，不执行跨 rank 归约。这个输入重排的伴随操作不会收集
输出 token 分片。层堆栈结束后，可选的完整全局输出重建由调用方负责；
Ring 内部的 attention 结果合并已由 VSA 完成。

`layout.input_rows` 将输入行映射到全局 raster 行（完整 raster layout 为恒等映射）。
`tokens`、`blocks` 和 row/slot 映射描述本地输入；
`groups`、`sample_slices` 和 `sample_lengths` 描述全局样本。
`slot_row` 将每块 64 个 slot 映射到输入行（`-1` 表示 padding），
`row_slot` 是有效行上的逆映射。Padding 填零，不参与均值和 attention。

底层有三套 kernel：`_reorder_tokens_kernel`（边界 gather/scatter）、
`_gather`（单张量 slot 转换）和 `_gather_qkv`（融合 Q/K/V slot 转换）。
相反方向复用这些 kernel。只有 block 输入满足 `tokens == blocks * 64`、
没有不完整/dummy block 时，`slots_are_input` 才跳过 slot 转换；
raster 输入始终经过 raster-to-slot 路径。

`layout.ring` 保存进程组/选项、rank/size、全局 token 数、block 归属、
peer 大小和 all-gather 排列。容量为 C 时，
`p = layout.ring.global_to_gather[b]` 给出 `owner = p // C` 和
`local_block = p % C`。`layout.ring.block_ids` 列出本地 block 的全局 ID，
dummy block 用 `-1` 表示。所有 rank 根据几何形状生成这些映射，无需通信。
Coarse Q 边界是本地的，KV 边界覆盖全局样本；
完整 layout 共享 Q/KV 边界和 block-ID 张量。

</details>

## 致谢

感谢 [FastVideo / FastH3](https://github.com/hao-ai-lab/FastVideo/blob/d5253fd71055927f12aa6c4aa679b93a00238851/fastvideo/attention/backends/video_sparse_attn_h3.py)
提供 VSA 参考实现和语义，以及
[NVIDIA cuDNN BSA](https://github.com/NVIDIA/cudnn-frontend/blob/v1.28.0/python/cudnn/block_sparse_attention/_interface.py)
提供本实现使用的核心块稀疏注意力 kernel。

[MIT 许可证](LICENSE)。
