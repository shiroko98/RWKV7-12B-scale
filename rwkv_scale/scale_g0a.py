import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import re # <-- 用于匹配参数名
from collections import OrderedDict # <-- 用于构建新的 state_dict
import itertools


########################################################################################################
# 核心配置：模型扩展
########################################################################################################
args = types.SimpleNamespace()

# --- 路径配置 ---
# 路径到原始的、L=32, C=4096 的模型
ORIGINAL_MODEL_PATH = "/mnt/lab/Models/pt/rwkv7-g0a2-7.2b-20251005-ctx4096.pth" # <--- 这是你的4096模型路径
# 插值后新模型的保存路径
EXPANDED_MODEL_PATH = "/mnt/data/train/models/rwkv_state_dict_expanded_L64_C6144_bo_init.pth" 
# 提供一个可选的初始化模型路径，用于“嫁接”扩展维度
INIT_MODEL_PATH = "/mnt/data/train/models/rwkv_30B_init_L64_6144.pth"  # 例如 "rwkv_30B_init_L64_6144.pth"

# --- 深度扩展配置 (L: 32 -> 64) ---
ORIGINAL_N_LAYER = 60

# 如果要从 32 扩展到 64 (2*L)，我们需要复制边界层（True），否则62层为False
STACK_LAST_LAYER = True

# 插值系数 (0.5 = 50/50 混合)
INTERPOLATION_ALPHA = 0.5

# --- 宽度扩展配置 (C: 4096 -> 6144) ---
# 原始 args.n_embd (C_orig)
ORIGINAL_N_EMBD = 4096
# 目标 args.n_embd (C_new) (6144 = 4096 * 1.5)
NEW_N_EMBD = 6144
# 原始 head_size
ORIGINAL_HEAD_SIZE = 64
# 新 head_size
NEW_HEAD_SIZE = 96

########################################################################################################
# 原始模型配置
########################################################################################################
#
#
args.n_layer = ORIGINAL_N_LAYER # 32
args.n_embd = ORIGINAL_N_EMBD   # 4096

C = args.n_embd # 4096
# D_DECAY = max(32, int(round((1.8 * (C**0.5)) / 32) * 32)) = 128
D_DECAY_LORA = 128
# D_AAA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32)) = 128
D_AAA_LORA = 128
# D_MV = max(32, int(round((1.3 * (C**0.5)) / 32) * 32)) = 96
D_MV_LORA = 96
# D_GATE = max(32, int(round((0.6 * (C**0.8)) / 32) * 32)) = 448
D_GATE_LORA = 480

args.vocab_size = 65536
DTYPE = torch.half
args.head_size_a = 96
HEAD_SIZE = args.head_size_a
USE_CUDA_KERNEL = False 
MyModule = nn.Module

# (确保 RWKV_TOKENIZER, RWKV7_OP, RWKV_Tmix_x070, RWKV_CMix_x070, Block, RWKV 的类定义在这里)
# ...
# (确保 interpolate_tensor, expand_model_width, expand_model_scheme_1 的函数定义在这里)
# ...

########################################################################################################
# 7.2b model 参数
new_D_DECAY_LORA = 128
# D_AAA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32)) = 128
new_D_AAA_LORA = 128
# D_MV = max(32, int(round((1.3 * (C**0.5)) / 32) * 32)) = 96
new_D_MV_LORA = 96
# D_GATE = max(32, int(round((0.6 * (C**0.8)) / 32) * 32)) = 448
new_D_GATE_LORA = 640

########################################################################################################

########################################################################################################
# 主执行函数：扩展并运行推理 (已修改)
########################################################################################################
class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

tokenizer = RWKV_TOKENIZER("/mnt/data/Models/Huggingface/RWKV7-Goose-World2.8-0.1B-HF/rwkv_vocab_v20230424.txt")

########################################################################################################
# CUDA Kernel (来自你的推理脚本)
########################################################################################################

if USE_CUDA_KERNEL:

    from torch.utils.cpp_extension import load

    load(name="wkv7", sources=["cuda/wkv7_op.cpp", f"cuda/wkv7.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            with torch.no_grad():
                HEAD_SIZE = 64
                B, T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                assert HEAD_SIZE == C // H
                assert r.dtype == DTYPE
                assert w.dtype == DTYPE
                assert k.dtype == DTYPE
                assert v.dtype == DTYPE
                assert a.dtype == DTYPE
                assert b.dtype == DTYPE
                assert r.is_contiguous()
                assert w.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert a.is_contiguous()
                assert b.is_contiguous()
                y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
                torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
                return y

    def RWKV7_OP(r, w, k, v, a, b):
        return WKV_7.apply(r, w, k, v, a, b)

else:

    def RWKV7_OP(r, w, k, v, a, b):
        HEAD_SIZE = 64
        B, T, C = r.size()
        H = C // HEAD_SIZE
        N = HEAD_SIZE
        r = r.view(B, T, H, N).float()
        k = k.view(B, T, H, N).float()
        v = v.view(B, T, H, N).float()
        a = a.view(B, T, H, N).float()
        b = b.view(B, T, H, N).float()
        w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
        out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

        for t in range(T):
            kk = k[:, t, :].view(B, H, 1, N)
            rr = r[:, t, :].view(B, H, N, 1)
            vv = v[:, t, :].view(B, H, N, 1)
            aa = a[:, t, :].view(B, H, N, 1)
            bb = b[:, t, :].view(B, H, 1, N)
            state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
            out[:, t, :] = (state @ rr).view(B, H, N)

        return out.view(B, T, C).to(dtype=torch.half)

########################################################################################################
# RWKV TimeMix (来自你的推理脚本)
########################################################################################################

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        H = self.n_head
        N = self.head_size
        C = args.n_embd # C (n_embd) 会被 args 控制 (4096 或 6144)

        # --- 关键修改：从 args 读取 LoRA 维度 ---
        # 这样，当我们在脚本末尾用新的 args 实例化模型时，
        # 它会使用新的 (扩展后的) D 维度。
        # getattr(args, 'D_DECAY_LORA', D_DECAY_LORA) 的意思是：
        # 尝试从 args 获取 D_DECAY_LORA，如果找不到，就使用全局变量 D_DECAY_LORA
        
        D_DECAY_LORA_dim = getattr(args, 'D_DECAY_LORA', D_DECAY_LORA)
        D_AAA_LORA_dim   = getattr(args, 'D_AAA_LORA', D_AAA_LORA)
        D_MV_LORA_dim    = getattr(args, 'D_MV_LORA', D_MV_LORA)
        D_GATE_LORA_dim  = getattr(args, 'D_GATE_LORA', D_GATE_LORA)
        # ----------------------------------------

        self.x_r = nn.Parameter(torch.empty(1,1,C))
        self.x_w = nn.Parameter(torch.empty(1,1,C))
        self.x_k = nn.Parameter(torch.empty(1,1,C))
        self.x_v = nn.Parameter(torch.empty(1,1,C))
        self.x_a = nn.Parameter(torch.empty(1,1,C))
        self.x_g = nn.Parameter(torch.empty(1,1,C))

        self.w0 = nn.Parameter(torch.empty(1,1,C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA_dim)) # <-- 使用 args 中的维度
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA_dim, C)) # <-- 使用 args 中的维度

        self.a0 = nn.Parameter(torch.empty(1,1,C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA_dim))   # <-- 使用 args 中的维度
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA_dim, C))   # <-- 使用 args 中的维度

        if self.layer_id > 0:
            self.v0 = nn.Parameter(torch.empty(1,1,C))
            self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA_dim))    # <-- 使用 args 中的维度
            self.v2 = nn.Parameter(torch.empty(D_MV_LORA_dim, C))    # <-- 使用 args 中的维度

        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA_dim))  # <-- 使用 args 中的维度
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA_dim, C))  # <-- 使用 args 中的维度

        self.k_k = nn.Parameter(torch.empty(1,1,C))
        self.k_a = nn.Parameter(torch.empty(1,1,C))
        self.r_k = nn.Parameter(torch.empty(H,N))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    
########################################################################################################
# RWKV ChannelMix (来自你的推理脚本)
########################################################################################################

class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

########################################################################################################
# RWKV Block (来自你的推理脚本)
########################################################################################################

class Block(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        if self.layer_id == 0:
                    self.ln0 = nn.LayerNorm(args.n_embd) # 只为 block 0 创建
        # self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)
        
    def forward(self, x, v_first):

        if self.layer_id == 0:
            x = self.ln0(x)

        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))

        return x, v_first

########################################################################################################
# RWKV Model (来自你的推理脚本)
########################################################################################################

class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 在 __init__ 中动态设置 dim_att 和 dim_ffn
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4 
        self.args = args # 保存 args
        
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)

        return x

########################################################################################################
# 核心算法：模型扩展 (方案 2)
########################################################################################################


########################################################################################################
# 辅助函数 1: 张量插值
########################################################################################################

def interpolate_tensor(tensor, target_shape):
    """
    使用 F.interpolate 将张量智能地调整到目标形状。
    """
    orig_dtype = tensor.dtype
    orig_shape = tensor.shape
    
    if list(orig_shape) == list(target_shape):
        return tensor.clone()

    tensor_float = tensor.float()
    orig_dims = tensor_float.dim()
    
    if orig_dims == 1:
        # 1D 向量: (C,) -> (C_new,)
        tensor_float = tensor_float.view(1, 1, -1) # (1, 1, C_orig)
        interp_tensor = F.interpolate(tensor_float, size=target_shape[0], mode='linear', align_corners=True)
        return interp_tensor.view(target_shape).to(orig_dtype)
        
    elif orig_dims == 2:
        # 2D 矩阵: (D1, D2) -> (D1_new, D2_new)
        tensor_float = tensor_float.unsqueeze(0).unsqueeze(0) # (1, 1, D1_orig, D2_orig)
        interp_tensor = F.interpolate(tensor_float, size=target_shape, mode='bilinear', align_corners=True)
        return interp_tensor.squeeze(0).squeeze(0).to(orig_dtype)
        
    elif orig_dims == 3 and orig_shape[0:2] == (1, 1):
        # 3D 向量: (1, 1, C) -> (1, 1, C_new)
        interp_tensor = F.interpolate(tensor_float, size=target_shape[2], mode='linear', align_corners=True)
        return interp_tensor.to(orig_dtype)
        
    else:
        print(f"[警告] 跳过插值：不支持的形状 {orig_shape}。将复制原始张量。")
        # 尝试只复制，如果形状不同则会失败，但这总比返回错误形状好
        if list(orig_shape) == list(target_shape):
             return tensor.clone()
        else:
             print(f"  [错误] 形状不匹配且无法插值！ {orig_shape} -> {target_shape}")
             # 创建一个零张量作为后备
             return torch.zeros(target_shape, dtype=orig_dtype)

# # 扩展宽度 (Width/Embedding) ---
#     new_state_dict = expand_model_width(
#         original_state_dict,
#         C_orig=ORIGINAL_N_EMBD,
#         C_new=NEW_N_EMBD,
#         head_size=args.head_size_a 
#     )

########################################################################################################
# 辅助函数: 张量填充 (Net2WiderNet)
# 辅助函数 1A: 张量 0 填充
########################################################################################################
def pad_tensor_zeros(tensor, target_shape):
    """
    使用 0 填充 (Padding) 来将张量扩展到目标形状。
    """
    orig_shape = tensor.shape
    
    if list(orig_shape) == list(target_shape):
        return tensor.clone()

    # 创建一个目标形状的全 0 张量
    new_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    
    # 构造一个切片元组，表示原始张量在
    # 新张量中的位置 (即 [0:dim1, 0:dim2, ...])
    slicing = tuple(slice(0, dim) for dim in orig_shape)
    
    # 将原始张量的数据复制到新张量的左上角
    new_tensor[slicing] = tensor
    
    return new_tensor

########################################################################################################
# 辅助函数 1B: 张量噪声填充 (用于对称性打破)
########################################################################################################
def pad_tensor_noise(tensor, target_shape, std_dev=1e-5):
    """
    使用微小的随机噪声填充张量的新增部分。
    """
    orig_shape = tensor.shape
    
    if list(orig_shape) == list(target_shape):
        return tensor.clone()

    # 创建一个目标形状的张量
    # 注意：我们使用 torch.empty 来获取未初始化的内存，然后填充
    new_tensor = torch.empty(target_shape, dtype=tensor.dtype, device=tensor.device)
    
    # 构造切片
    slicing = tuple(slice(0, dim) for dim in orig_shape)
    
    # 1. 复制原始数据
    new_tensor[slicing] = tensor
    
    # 2. 为新部分生成噪声
    # (这部分逻辑有点复杂，我们需要找到 "非" 切片的部分)
    # 为了简单起见，我们先用噪声填充所有，再覆盖原始数据
    
    # 用正态分布噪声填充整个张量
    # 我们使用 0 均值和极小的标准差
    noise = torch.randn(target_shape, dtype=tensor.dtype, device=tensor.device) * std_dev
    new_tensor = noise
    
    # 再次将原始张量的数据复制到左上角
    new_tensor[slicing] = tensor
    
    print(f"     > [注意] 使用了噪声填充 (std={std_dev})")
    return new_tensor



########################################################################################################
# 辅助函数 1D: 张量平铺 (Copying/Tiling)
########################################################################################################
def pad_tensor_tile(tensor, target_shape):
    """
    使用“复制/平铺”策略将张量扩展到目标形状。
    它会重复使用原始张量的数据来填充新的空间。
    
    例如：(4096,) -> (6144,) 会变成 [tensor, tensor[0:2048]]
    例如：(4096, 4096) -> (6144, 6144) 会变成:
         [ [Orig,          Orig[0:4096, 0:2048]],
           [Orig[0:2048, :], Orig[0:2048, 0:2048]] ]
    """
    orig_shape = tensor.shape
    
    # 如果形状相同，直接克隆
    if list(orig_shape) == list(target_shape):
        return tensor.clone()

    # 创建一个目标形状的空张量
    new_tensor = torch.empty(target_shape, dtype=tensor.dtype, device=tensor.device)

    # --- 1. 为每个维度计算源和目标切片 ---
    target_slices_per_dim = []
    source_slices_per_dim = []

    for i in range(tensor.dim()):
        orig_d = orig_shape[i]
        target_d = target_shape[i]
        
        # 如果此维度不变，则切片就是它本身
        if orig_d == target_d:
            target_slices_per_dim.append( [slice(0, orig_d)] )
            source_slices_per_dim.append( [slice(0, orig_d)] )
            continue

        # 如果此维度需要扩展
        target_d_slices = []
        source_d_slices = []
        
        current_pos = 0
        while current_pos < target_d:
            # 计算我们需要复制的块有多大
            chunk_size = min(orig_d, target_d - current_pos)
            
            # 目标切片
            target_d_slices.append(slice(current_pos, current_pos + chunk_size))
            # 源切片 (!!! 关键 !!! 总是从头开始复制)
            source_d_slices.append(slice(0, chunk_size)) 
            
            current_pos += chunk_size
            
        target_slices_per_dim.append(target_d_slices)
        source_slices_per_dim.append(source_d_slices)

    # --- 2. 组合所有维度的切片并执行复制 ---
    
    # itertools.product 会创建所有切片的组合
    # 例如: [(tgt_dim0_s1, tgt_dim1_s1), (tgt_dim0_s1, tgt_dim1_s2), ...]
    target_slice_combinations = itertools.product(*target_slices_per_dim)
    source_slice_combinations = itertools.product(*source_slices_per_dim)
    
    # 遍历所有块并复制
    for target_slice, source_slice in zip(target_slice_combinations, source_slice_combinations):
        new_tensor[target_slice] = tensor[source_slice]
        
    return new_tensor


def pad_tensor_fpi(tensor, target_shape, param_name):
    """
    使用“不对称零初始化”策略扩展张量，以实现功能保持性初始化 (FPI)。
    [cite: 26, 29, 81]

    - 原始权重被复制到新张量的“左上角”。
    - “新增的列”（输出侧）保持为 0，以确保新维度的输出为 0 。
    - “新增的行”（输入侧）被随机初始化 (Kaiming Uniform)，
       因为它们将乘以 0 (来自零填充的输入)，
       所以它们的值不影响 FPI，但随机初始化有助于后续训练 [cite: 43, 57]。
    - 特殊情况 (LayerNorm, 嵌入) 按文档处理。
    """
    orig_shape = tensor.shape
    new_shape = target_shape

    # 1. 创建一个目标形状的零张量。
    # 这自动处理了所有“新增列”（输出侧）的零初始化 [cite: 35, 58, 64, 73, 78]。
    new_tensor = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)

    # 2. 将原始张量复制到“左上角”
    orig_slices = tuple(slice(0, d) for d in orig_shape)
    new_tensor[orig_slices] = tensor.clone()

    # 3. --- 应用不对称初始化 (随机化或置1) ---
    
    if tensor.dim() == 2:
        orig_d0, orig_d1 = orig_shape
        new_d0, new_d1 = new_shape

        # A. "新增的行" (输入侧) -> 随机初始化
        # (即 dim 0 扩展了)
        if new_d0 > orig_d0:
            # emb.weight [V, C] 是特例，它的 dim 0 是词汇表，不应随机化
            if 'emb.weight' not in param_name:
                # 这适用于所有其他2D矩阵：
                # W_Q/K/V/O, W_1(ffn.value), W_2(ffn.key), 
                # att.r_k, 以及所有 LoRA 矩阵 (w1, w2, a1, a2...)
                # [cite: 57, 63, 72, 77, 81]
                
                # 我们只初始化与 *原始列* 对应的部分
                new_rows_orig_cols_slice = (slice(orig_d0, new_d0), slice(0, orig_d1))
                
                # 使用 Kaiming (He) 初始化 (适用于 GeLU/ReLU 激活)
                torch.nn.init.kaiming_uniform_(new_tensor[new_rows_orig_cols_slice], a=0.01)

        # B. "新增的列" (输出侧) -> 保持为 0 (默认已处理)
        # (即 dim 1 扩展了)
        
        # C. 特殊情况: head.weight [V, C] (即 W_lm_head^T)
        # 根据文档 ，我们打破绑定。
        # W_lm_head [C, V] 的新行 (dim 0) 应随机化 。
        # 这对应于 head.weight [V, C] 的新列 (dim 1) 随机化。
        if new_d1 > orig_d1 and 'head.weight' in param_name:
            # 我们只初始化与 *原始行* 对应的部分
            new_cols_slice = (slice(0, orig_d0), slice(orig_d1, new_d1))
            torch.nn.init.kaiming_uniform_(new_tensor[new_cols_slice], a=0.01)

    elif tensor.dim() == 1:
        # 1D 向量: [C]
        if new_shape[0] > orig_shape[0]:
            new_part_slice = (slice(orig_shape[0], new_shape[0]),)
            
            # LayerNorm 权重 (gamma) 应初始化为 1
            if '.weight' in param_name and 'ln' in param_name:
                torch.nn.init.ones_(new_tensor[new_part_slice])
            
            # LayerNorm 偏置 (beta) 和其他 1D 向量应初始化为 0 (默认已处理)

    elif tensor.dim() == 3:
        # 3D 向量: [1, 1, C]
        # FPI 要求新维度为 0 [cite: 31]，默认已处理
        pass

    return new_tensor

########################################################################################################
# 核心算法：宽度扩展
########################################################################################################

def expand_model_width(original_state_dict, C_orig, C_new, head_size_orig, head_size_new,
                       D_decay_orig, D_decay_new,
                       D_aaa_orig, D_aaa_new,
                       D_mv_orig, D_mv_new,
                       D_gate_orig, D_gate_new):
    """
    扩展模型的宽度 (n_embd) 和 LoRA (D) 维度，
    使用插值来调整所有相关权重。
    """
    print(f"--- [模型宽度扩展] ---")
    print(f"原始 n_embd (C): {C_orig} -> 新 n_embd (C_new): {C_new}")
    print(f"原始 head_size: {head_size_orig} -> 新 head_size: {head_size_new}")
    
    # --- 1. 计算所有相关维度 ---
    F_orig = C_orig * 4
    F_new = C_new * 4
    
    if C_orig % head_size_orig != 0:
        raise ValueError(f"原始 n_embd ({C_orig}) 必须能被原始 head_size ({head_size_orig}) 整除。")
    if C_new % head_size_new != 0:
        raise ValueError(f"新 n_embd ({C_new}) 必须能被新 head_size ({head_size_new}) 整除。")
        
    H_orig = C_orig // head_size_orig
    H_new = C_new // head_size_new
    
    print(f"FFN 维度 (F): {F_orig} -> {F_new}")
    print(f"Head 数量 (H): {H_orig} -> {H_new}")
    print(f"Decay LoRA (D_decay): {D_decay_orig} -> {D_decay_new}")
    print(f"AAA LoRA (D_aaa): {D_aaa_orig} -> {D_aaa_new}")
    print(f"MV LoRA (D_mv): {D_mv_orig} -> {D_mv_new}")
    print(f"Gate LoRA (D_gate): {D_gate_orig} -> {D_gate_new}")
    
    new_state_dict = OrderedDict()
    
    # --- 2. 遍历所有参数进行插值 ---
    for key, W_orig in original_state_dict.items():
        W_shape = list(W_orig.shape)
        new_shape = list(W_shape) # 默认：形状不变
        
        layer_id_str = None # 用于存储层 ID 字符串
        match = re.match(r"blocks\.(\d+)\.(.*)", key)
        if match:
            layer_id_str = match.group(1) # 获取层 ID 字符串
            param_name = match.group(2)
        else:
            param_name = key # 顶层参数 (emb, ln_out, head)

        # 检查是否为 block 0 且参数为 v0, v1, 或 v2
        if layer_id_str == "0" and param_name in ['att.v0', 'att.v1', 'att.v2']:
            print(f"     > [清理] 正在跳过并删除 Block 0 中无效的键: {key}")
            continue # continue 会跳过此循环的剩余部分，即不会将其添加到 new_state_dict

        # --- 3. 根据参数名确定新形状 ---
        
        # 1D 向量 (C_orig) -> (C_new)
        if param_name in [
            'ln0.weight', 'ln0.bias', 
            'ln1.weight', 'ln1.bias', 
            'ln2.weight', 'ln2.bias',
            'att.ln_x.weight', 'att.ln_x.bias',
            'ln_out.weight', 'ln_out.bias'
        ]:
            if W_shape == [C_orig]: new_shape = [C_new]
        
        # 3D 向量 (1, 1, C_orig) -> (1, 1, C_new)
        elif param_name in [
            'att.x_r', 'att.x_w', 'att.x_k', 'att.x_v', 'att.x_a', 'att.x_g',
            'att.w0', 'att.a0', 'att.v0', 'att.k_k', 'att.k_a',
            'ffn.x_k'
        ]:
            if W_shape == [1, 1, C_orig]: new_shape = [1, 1, C_new]

        # --- (!!! 核心修改 !!!) ---
        # 2D LoRA-like (Input/Output expands)
        # (C_orig, D_orig) -> (C_new, D_new)
        
        # att.w1 (C_orig, D_decay_orig) -> (C_new, D_decay_new)
        elif param_name == 'att.w1':
            if W_shape == [C_orig, D_decay_orig]: 
                new_shape = [C_new, D_decay_new]
        # att.w2 (D_decay_orig, C_orig) -> (D_decay_new, C_new)
        elif param_name == 'att.w2':
            if W_shape == [D_decay_orig, C_orig]: 
                new_shape = [D_decay_new, C_new]
        
        # att.a1 (C_orig, D_aaa_orig) -> (C_new, D_aaa_new)
        elif param_name == 'att.a1':
            if W_shape == [C_orig, D_aaa_orig]: 
                new_shape = [C_new, D_aaa_new]
        # att.a2 (D_aaa_orig, C_orig) -> (D_aaa_new, C_new)
        elif param_name == 'att.a2':
            if W_shape == [D_aaa_orig, C_orig]: 
                new_shape = [D_aaa_new, C_new]

        # att.v1 (C_orig, D_mv_orig) -> (C_new, D_mv_new)
        elif param_name == 'att.v1':
            if W_shape == [C_orig, D_mv_orig]: 
                new_shape = [C_new, D_mv_new]
        # att.v2 (D_mv_orig, C_orig) -> (D_mv_new, C_new)
        elif param_name == 'att.v2':
            if W_shape == [D_mv_orig, C_orig]: 
                new_shape = [D_mv_new, C_new]

        # att.g1 (C_orig, D_gate_orig) -> (C_new, D_gate_new)
        elif param_name == 'att.g1':
            if W_shape == [C_orig, D_gate_orig]: 
                new_shape = [C_new, D_gate_new]
        # att.g2 (D_gate_orig, C_orig) -> (D_gate_new, C_new)
        elif param_name == 'att.g2':
            if W_shape == [D_gate_orig, C_orig]: 
                new_shape = [D_gate_new, C_new]
        
        # ----------------------------

        # 2D 全连接 (C_orig, C_orig) -> (C_new, C_new)
        elif param_name in [
            'att.receptance.weight', 'att.key.weight', 
            'att.value.weight', 'att.output.weight'
        ]:
            if W_shape == [C_orig, C_orig]: new_shape = [C_new, C_new]
            
        # 2D 特殊 r_k (H_orig, head_size_orig) -> (H_new, head_size_new)
        elif param_name == 'att.r_k':
            if W_shape == [H_orig, head_size_orig]: 
                new_shape = [H_new, head_size_new]

        # 2D FFN 矩阵
        elif param_name == 'ffn.key.weight': # (F_orig, C_orig)
            if W_shape == [F_orig, C_orig]: new_shape = [F_new, C_new]
        elif param_name == 'ffn.value.weight': # (C_orig, F_orig)
            if W_shape == [C_orig, F_orig]: new_shape = [C_new, F_new]
            
        # 2D Embedding & Head
        elif param_name == 'emb.weight': # (Vocab, C_orig)
            if W_shape[1] == C_orig: new_shape = [W_shape[0], C_new]
        elif param_name == 'head.weight': # (Vocab, C_orig)
            if W_shape[1] == C_orig: new_shape = [W_shape[0], C_new]

        # --- 4. 执行插值或复制 ---
        if W_shape == new_shape:
            new_state_dict[key] = W_orig.clone()

        # 方法1：插值
        # How to ,,,,cionwards, comingual- coming,,,,
        # ..., finally, closest, closest,,,,,,
        # else:
        #     print(f"     > 正在插值 {key}: {W_shape} -> {new_shape}")
        #     new_state_dict[key] = interpolate_tensor(W_orig, tuple(new_shape))
        
        # 方法2：平铺，确定性地复制（总是从张量的开头 [0:2048] 部分进行复制） 不nan，但是乱说话：
        # How to QDOino in

        #  h’’
        # n-  po’ 

        #  s’ s’n’’negierm’
        # else:
        #     # 形状不同，执行平铺
        #     print(f"     > 正在平铺 [Tiling] {key}: {W_shape} -> {new_shape}")
        #     new_state_dict[key] = pad_tensor_tile(W_orig, tuple(new_shape))

        #方法3： padding
        else:
            # padding策略：
            # 1. emb.weight: 使用噪声填充，打破对称性，注入信号
            # 2. 所有其他: 使用 0 填充，保持原始计算路径
            # 输出为Nan
            if param_name == 'emb.weight':
                print(f"     > 正在填充 [Noise] {key}: {W_shape} -> {new_shape}")
                # (你可以调整 std_dev，但 1e-5 是一个安全值)
                new_state_dict[key] = pad_tensor_noise(W_orig, tuple(new_shape), std_dev=1e-5)
            
            # (!!! 针对 LoRA 维度的特殊情况 !!!)
            # 如果 LoRA D 维度也被扩展了 (C, D_orig) -> (C_new, D_new)
            # 我们也应该用噪声填充，否则新的 D 维度也是“死的”
            elif param_name in ['att.w1', 'att.a1', 'att.v1', 'att.g1',
                                'att.w2', 'att.a2', 'att.v2', 'att.g2']:
                print(f"     > 正在填充 [Noise for LoRA] {key}: {W_shape} -> {new_shape}")
                # 对 LoRA 权重也使用噪声填充
                new_state_dict[key] = pad_tensor_noise(W_orig, tuple(new_shape), std_dev=1e-5)
            
            else:
                # 所有其他权重 (ln, ffn.key, head) 均使用 0 填充
                print(f"     > 正在填充 [Zeros] {key}: {W_shape} -> {new_shape}")
                new_state_dict[key] = pad_tensor_noise(W_orig, tuple(new_shape), std_dev=1e-5)
                # new_state_dict[key] = pad_tensor_zeros(W_orig, tuple(new_shape))              #######################################################################################################

        #方法4： pad fpi => Nan
        # else:
        #     print(f"     > 正在 FPI 扩展 [FPI] {key}: {W_shape} -> {new_shape}")
        #     new_state_dict[key] = pad_tensor_fpi(W_orig, tuple(new_shape), key)
    print(f"--- [模型宽度扩展完成] ---")
    return new_state_dict

########################################################################################################
# 辅助函数：用初始化模型权重覆盖新扩展的 4096->6144 维度
########################################################################################################

def graft_init_segments(expanded_state_dict, init_state_dict, C_orig, C_new):
    """
    将 width-scale 得到的权重 (expanded_state_dict) 中「新增的 2048 维」替换为
    init_state_dict 中按照特定规则初始化的权重，其余 0~4095 维保持扩展前的小模型数值。

    Args:
        expanded_state_dict: dict，在 expand_model_width 之后得到的权重。
        init_state_dict: dict，来自 init_and_save_model 生成的权重。
        C_orig: int，原模型的 embedding 维度（4096）。
        C_new: int，扩展后的 embedding 维度（6144）。

    Returns:
        OrderedDict: 已经替换扩展段的 state_dict。
    """
    assert C_new >= C_orig, f"预期 C_new >= C_orig，但得到 {C_new} < {C_orig}"

    grafted = OrderedDict()
    for key, scaled_tensor in expanded_state_dict.items():
        if key not in init_state_dict:
            # 初始化模型中缺少该权重，直接沿用扩展权重
            grafted[key] = scaled_tensor.clone()
            continue

        init_tensor = init_state_dict[key]
        if scaled_tensor.shape != init_tensor.shape:
            print(f"[警告] graft_init_segments: {key} 形状不匹配，跳过。scaled={tuple(scaled_tensor.shape)}, init={tuple(init_tensor.shape)}")
            grafted[key] = scaled_tensor.clone()
            continue

        merged_tensor = scaled_tensor.clone()
        target_axes = [axis for axis, dim in enumerate(merged_tensor.shape) if dim == C_new]

        if not target_axes:
            # 没有 4096->6144 的轴，直接保留
            grafted[key] = merged_tensor
            continue

        # 逐轴覆盖：凡是命中扩展区间 [C_orig:, :] 的切片都来自 init_tensor
        for axis in target_axes:
            if merged_tensor.shape[axis] <= C_orig:
                continue
            slicer = [slice(None)] * merged_tensor.dim()
            slicer[axis] = slice(C_orig, C_new)
            merged_tensor[tuple(slicer)] = init_tensor[tuple(slicer)]

        grafted[key] = merged_tensor

    return grafted


def expand_model_scheme_1(original_state_dict, original_n_layer, 
                          interpolation_alpha=0.5, 
                          stack_last_layer: bool = False):
    """
    实现一个灵活的模型扩展方案。

    核心逻辑:
    - 中间层 (L_1 到 L_{L-2}): 应用 [复制, 插值] 模式。
      L_k' = interp(L_k, L_{k+1})
    - 边界层 (L_0, L_{L-1}): 行为由 `stack_last_layer` 控制。
      - stack_last_layer=True: 应用 [复制] 模式 (L_{L-1}, L_{L-1})，复制两遍
      - stack_last_layer=False: 不复制L_0和L_last

    此版本能正确处理所有特殊情况，包括 L_0 中缺少的 att.v0/v1/v2 参数。
    """
    L = original_n_layer
    print(f"--- [模型扩展] ---")
    print(f"方案 (边界复制={stack_last_layer}): 正在准备扩展模型...")

    # --- 准备工作：分类 state_dict 键 ---
    new_state_dict = OrderedDict()
    block_key_pattern = re.compile(r"blocks\.(\d+)\.(.*)")
    
    original_keys_by_layer = [{} for _ in range(L)] # 使用字典列表以便快速查找
    non_block_keys = []
    
    print("步骤 1: 正在分类原始 state_dict 键...")
    for key, tensor in original_state_dict.items():
        match = block_key_pattern.match(key)
        if match:
            layer_id = int(match.group(1))
            if layer_id < L:
                remaining_key = match.group(2)
                original_keys_by_layer[layer_id][remaining_key] = key
        else:
            non_block_keys.append(key)

    print("步骤 2: 正在复制非 'blocks' 层的参数 (emb, ln_out, head)...")
    for key in non_block_keys:
        if key in original_state_dict:
            new_state_dict[key] = original_state_dict[key].clone()

    # --- 阶段一: 规划新模型的结构 ---
    # 我们将创建一个操作列表，而不是直接计算。这能简化索引管理。
    # 每个操作是元组 (操作类型, [源层索引])
    print("步骤 3: 正在规划新模型的层结构...")
    new_layers_config = []
    for k in range(L):
        # 1. 添加原始层 (总是复制)
        new_layers_config.append(('copy', [k]))

        # 2. 决定是否添加一个 "prime" 层 (L_k')
        # Case A: 中间层 (1 到 L-2)，总是进行插值
        if 0 < k < L - 1:
            new_layers_config.append(('interp', [k, k + 1]))
        # Case B: 边界层 (0 或 L-1)
        elif (k == L - 1):
            if stack_last_layer:
                new_layers_config.extend([('copy', [k]), ('copy', [k])]) # 应用 [复制, 复制]
            # else: 不动，不添加额外层

    new_L = len(new_layers_config)
    print(f"  > 规划完成。新模型总层数: {new_L}")

    # --- 阶段二: 根据规划构建新的 state_dict ---
    print("步骤 4: 正在根据规划构建新权重...")
    for i, (op_type, source_indices) in enumerate(new_layers_config):
        new_layer_idx = i

        # --- 操作: 'copy' ---
        if op_type == 'copy':
            source_k = source_indices[0]
            print(f"  > 正在复制 旧 L_{source_k} -> 新 L_{new_layer_idx}...")
            
            for remaining_key, original_key in original_keys_by_layer[source_k].items():
                # 只有 block.0 应该有 ln0.*。所有其他新层都应跳过它。
                if new_layer_idx > 0 and remaining_key.startswith("ln0."):
                    continue
                
                new_key = f"blocks.{new_layer_idx}.{remaining_key}"
                new_state_dict[new_key] = original_state_dict[original_key].clone()

        # --- 操作: 'interp' ---
        elif op_type == 'interp':
            k, k_plus_1 = source_indices
            print(f"  > 正在插值 L_{k} 和 L_{k_plus_1} -> 新 L_{new_layer_idx}...")

            # 使用 L_k 作为插值键的模板
            for remaining_key, original_key_k in original_keys_by_layer[k].items():
                # 插值层永远不可能是 block.0，所以总是跳过 ln0
                if remaining_key.startswith("ln0."):
                    continue

                new_key_prime = f"blocks.{new_layer_idx}.{remaining_key}"
                W_k = original_state_dict[original_key_k]
                orig_dtype = W_k.dtype
                
                # # 非浮点数参数直接复制
                # if not orig_dtype.is_floating_point:
                #     new_state_dict[new_key_prime] = W_k.clone()
                #     continue

                # 健壮地查找 L_{k+1} 中的对应权重
                try:
                    # 在 L_{k+1} 中查找
                    original_key_k_plus_1 = original_keys_by_layer[k_plus_1][remaining_key]
                    W_k_plus_1 = original_state_dict[original_key_k_plus_1]
                    # 如果形状不匹配，也使用 W_k
                    if W_k_plus_1.shape != W_k.shape:
                        W_k_plus_1 = W_k
                except KeyError:
                    # 如果在 L_{k+1} 中找不到 (例如 L_0 中没有 att.v0)，则使用 L_k 的权重代替
                    print(f"    [信息] 键 '{remaining_key}' 在 L_{k_plus_1} 中未找到。将使用 L_{k} 的值代替。")
                    W_k_plus_1 = W_k

                # 执行插值，转换为32位浮点数，目的是提升计算精度
                W_k_prime = (interpolation_alpha * W_k.float()) + ((1.0 - interpolation_alpha) * W_k_plus_1.float())
                new_state_dict[new_key_prime] = W_k_prime.to(dtype=orig_dtype)

    print(f"步骤 5: 扩展完成。总层数: {new_L}")
    print(f"--- [模型扩展结束] ---")
    return new_state_dict, new_L


def run_expansion_and_inference():
    
    global args # 我们需要修改全局 args 以便 RWKV() 构造函数能正确工作

    # --- 1. 加载原始模型 ---
    print(f"--- [阶段 1/5] ---")
    print(f"正在加载原始 L={ORIGINAL_N_LAYER}, C={ORIGINAL_N_EMBD} 模型: {ORIGINAL_MODEL_PATH}")
    # 确保在 CPU 上加载，避免 GPU 内存问题
    original_state_dict = torch.load(ORIGINAL_MODEL_PATH, map_location="cpu")
    
    # --- 2. 步骤 A: 扩展深度 (Length) ---
    print(f"\n--- [阶段 2/5] ---")
    print(f"正在扩展深度: L={ORIGINAL_N_LAYER} -> L={ORIGINAL_N_LAYER * 2}")
    state_dict_L, new_L = expand_model_scheme_1(
        original_state_dict, 
        ORIGINAL_N_LAYER, 
        interpolation_alpha=INTERPOLATION_ALPHA,
        stack_last_layer=STACK_LAST_LAYER
    )
    print(f"深度扩展完成。新层数: {new_L}")
    
    # # 释放内存
    del original_state_dict
    gc.collect()
    
    # state_dict_L = original_state_dict # 如果跳过深度扩展，请使用它
    # new_L = ORIGINAL_N_LAYER # 如果跳过深度扩展，请使用它

    # --- 3. 步骤 B: 扩展宽度 (Width) ---
    print(f"\n--- [阶段 3/5] ---")
    print(f"正在扩展宽度: C={ORIGINAL_N_EMBD} -> C={NEW_N_EMBD} (同时扩展 D 维度)")
    
    # (!!! 核心修改 !!!)
    # 调用 expand_model_width 时，传入所有原始 D 维度和新的 D 维度
    state_dict_L_C = expand_model_width(
        state_dict_L,
        C_orig=ORIGINAL_N_EMBD,
        C_new=NEW_N_EMBD,
        head_size_orig=ORIGINAL_HEAD_SIZE,
        head_size_new=NEW_HEAD_SIZE,
        D_decay_orig=D_DECAY_LORA, D_decay_new=new_D_DECAY_LORA,
        D_aaa_orig=D_AAA_LORA, D_aaa_new=new_D_AAA_LORA,
        D_mv_orig=D_MV_LORA, D_mv_new=new_D_MV_LORA,
        D_gate_orig=D_GATE_LORA, D_gate_new=new_D_GATE_LORA
    )
    print("宽度扩展完成。")

    # --- 可选：用 init 模型覆盖新增的 2048 维 ---
    if INIT_MODEL_PATH:
        print(f"正在加载初始化模型权重: {INIT_MODEL_PATH}")
        init_state_dict = torch.load(INIT_MODEL_PATH, map_location="cpu")
        print("开始 graft_init_segments：保留原维度，替换新维度段为 init 权重...")
        state_dict_L_C = graft_init_segments(
            expanded_state_dict=state_dict_L_C,
            init_state_dict=init_state_dict,
            C_orig=ORIGINAL_N_EMBD,
            C_new=NEW_N_EMBD
        )
        print("graft_init_segments 完成。")

    # 释放内存
    del state_dict_L
    # if 'original_state_dict' in locals() and state_dict_L is not original_state_dict:
    #     del original_state_dict
    gc.collect()
    
    # --- 4. 保存最终模型 ---
    print(f"\n--- [阶段 4/5] ---")
    print(f"正在保存 L={new_L}, C={NEW_N_EMBD} 扩展后的模型到: {EXPANDED_MODEL_PATH}")
    torch.save(state_dict_L_C, EXPANDED_MODEL_PATH)
    
    # 在加载前释放最后的 state_dict 内存
    del state_dict_L_C
    gc.collect()

    # --- 5. 准备并运行推理 ---
    print(f"\n--- [阶段 5/5] ---")
    
    # !! 关键：更新全局 args 以匹配新模型 !!
    print(f"正在配置推理: L={new_L}, C={NEW_N_EMBD}")
    args.n_layer = new_L      # 60 (或你扩展后的层数)
    args.n_embd = NEW_N_EMBD    # 6144
    args.head_size_a = NEW_HEAD_SIZE  # 更新 head_size
    
    # (!!! 核心修改 !!!)
    # 将新的 D 维度添加到 args 对象中
    # 这样 RWKV_Tmix_x070.__init__ 才能正确构建模型
    args.D_DECAY_LORA = new_D_DECAY_LORA
    args.D_AAA_LORA = new_D_AAA_LORA
    args.D_MV_LORA = new_D_MV_LORA
    args.D_GATE_LORA = new_D_GATE_LORA
    print(f"     ... 使用 head_size={args.head_size_a}, D_decay={args.D_DECAY_LORA}, D_aaa={args.D_AAA_LORA}, D_mv={args.D_MV_LORA}, D_gate={args.D_GATE_LORA}")
    
    # 确保 RWKV __init__ 会使用的 dim_ffn 和 dim_att 被正确设置
    args.dim_att = args.n_embd
    args.dim_ffn = args.n_embd * 4
    
    print(f"正在加载新权重 {EXPANDED_MODEL_PATH} 到 RWKV(args) 实例...")
    model_params = torch.load(EXPANDED_MODEL_PATH, map_location="cpu")

    with torch.no_grad():
        # 实例化新模型，它现在会使用 L=60, C=6144, 和 *新* 的 D_LoRA 维度
        model = RWKV(args).to(dtype=DTYPE).cpu()
        
        # 加载我们精心构建的 state_dict
        model.load_state_dict(model_params, strict=True) # 使用 strict=True 确保所有键都匹配
        print("模型加载成功")

        # --- 运行推理循环 ---
        prompt = "hello"
        input_ids = tokenizer.encode(prompt)
        print(f'{prompt}', end='', flush=True)

        input_tensor = torch.tensor(input_ids).reshape(1, -1).cpu()
        generated_tokens = input_ids.copy()
        output_text = prompt

        for _ in range(100): # 减少到100个token以便快速测试
            out = model.forward(input_tensor)
            logits = out[0, -1]
            probs = F.softmax(logits.float(), dim=-1)

            k = 10
            try:
                _, indices = torch.topk(probs, k)
                next_token = indices[torch.multinomial(probs[indices], 1)].item()
            except Exception as e:
                print(f"\n[错误] 采样失败: {e}. 使用贪婪解码。")
                next_token = torch.argmax(probs).item()

            generated_tokens.append(next_token)
            input_tensor = torch.tensor(generated_tokens).reshape(1, -1).cpu()

            token_text = tokenizer.decode([next_token])
            output_text += token_text
            print(token_text, end='', flush=True)
        
        print("\n--- [推理结束] ---")

# 确保脚本被执行时运行主函数
if __name__ == "__main__":
    # 确保 tokenizer 实例在全局范围内
    # tokenizer = RWKV_TOKENIZER(".../rwkv_vocab_v20230424.txt")
    run_expansion_and_inference()