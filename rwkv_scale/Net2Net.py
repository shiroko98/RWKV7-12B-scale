import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import re # <-- 用于匹配参数名
from collections import OrderedDict # <-- 用于构建新的 state_dict

# ... (你脚本中所有未修改的部分，如 np.set_printoptions, args = ..., RWKV_TOKENIZER, RWKV7_OP, RWKV_Tmix_x070, RWKV_CMix_x070, Block, RWKV, interpolate_tensor, expand_model_width, expand_model_scheme_1) ...
# (我假设你的 tokenizer, 模型定义, 和两个扩展函数都已存在)
# (下面是需要修改/替换的部分)

########################################################################################################
# 核心配置：模型扩展
########################################################################################################
args = types.SimpleNamespace()

# --- 路径配置 ---
# 路径到你原始的、L=32, C=4096 的模型
ORIGINAL_MODEL_PATH = "/mnt/lab/Models/pt/rwkv7-g0a2-7.2b-20251005-ctx4096.pth" # <--- 假设这是你的4096模型路径
# 插值后新模型的保存路径
EXPANDED_MODEL_PATH = "/mnt/data/Codes/RWKV/RWKV-Scale/models/rwkv_state_dict_expanded_L64_C6144_net2net.pth" # <--- 修改了目标路径名

# --- 深度扩展配置 (L: 32 -> 64) ---
ORIGINAL_N_LAYER = 32

# 如果要从 32 扩展到 64 (2*L)，我们需要复制边界层（True），否则62层为False
STACK_LAST_LAYER = True 

# 插值系数 (0.5 = 50/50 混合)
INTERPOLATION_ALPHA = 0.5

# --- 宽度扩展配置 (C: 4096 -> 6144) ---
# 原始 args.n_embd (C_orig)
ORIGINAL_N_EMBD = 4096
# 目标 args.n_embd (C_new) (6144 = 4096 * 1.5)
NEW_N_EMBD = 6144

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
args.head_size_a = 64
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


def _create_split_map(orig_d, new_d, device):
    """
    创建一个“神经元分裂图”。
    
    返回: (indices, counts)
    - indices: 一个 [n_new] 张量，指定每个新神经元从哪个旧神经元复制而来。
    - counts: 一个 [orig_d] 张量，指定每个旧神经元 *总共* 被使用了多少次 (包括它自己)。
    """
    if orig_d == new_d:
        return None # 此维度没有变化
    
    n_new = new_d - orig_d
    
    # 1. 随机选择 (n_new) 个原始索引来进行复制
    indices = torch.randint(0, orig_d, (n_new,), device=device)
    
    # 2. 计算每个原始神经元的总使用次数 (用于补偿)
    # 我们将原始索引 (0..orig_d-1) 和新复制的索引 (indices) 拼接在一起
    all_indices = torch.cat([torch.arange(orig_d, device=device), indices])
    
    # bincount 会统计 0 出现了多少次, 1 出现了多少次...
    # 这就是补偿所需的分母 (m)
    counts = torch.bincount(all_indices, minlength=orig_d).float()
    
    return {
        'indices': indices,  # [n_new]
        'counts': counts,    # [orig_d]
        'orig_d': orig_d,
        'new_d': new_d
    }

def _expand_dim(tensor, dim, split_map, compensate=False):
    """
    对张量的指定维度应用“分裂”和（可选的）“补偿”。
    """
    if split_map is None:
        return tensor # 此维度无需扩展

    orig_d = split_map['orig_d']
    new_d = split_map['new_d']
    indices = split_map['indices']
    counts = split_map['counts']
    
    # 1. 创建一个具有正确目标形状的新张量
    new_shape = list(tensor.shape)
    new_shape[dim] = new_d
    new_tensor = torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)
    
    # 2. 复制原始张量数据
    orig_slices = [slice(None)] * tensor.dim()
    orig_slices[dim] = slice(0, orig_d)
    new_tensor[tuple(orig_slices)] = tensor
    
    # 3. 应用“分裂”（Net2WiderNet - 随机复制）
    #    从原始张量中选择 `indices` 指定的行/列...
    split_data = tensor.index_select(dim, indices)
    #    ...并将它们放入新张量的新增部分
    new_slices = [slice(None)] * tensor.dim()
    new_slices[dim] = slice(orig_d, new_d)
    new_tensor[tuple(new_slices)] = split_data
    
    # 4. 应用“补偿”（Net2WiderNet）
    #    [cite: 20]
    if compensate:
        # 构造一个形状，以便我们可以广播除法
        # (例如，如果 dim=0 且 tensor.dim()=2, shape=[orig_d, 1])
        broadcast_shape = [1] * tensor.dim()
        broadcast_shape[dim] = orig_d
        
        # 补偿原始部分：将原始的 (0..orig_d) 部分除以它们的计数值
        orig_part_slices = [slice(None)] * tensor.dim()
        orig_part_slices[dim] = slice(0, orig_d)
        new_tensor[tuple(orig_part_slices)] /= counts.view(broadcast_shape)
        
        # 补偿新复制的部分：将新的 (orig_d..new_d) 部分除以 *它们源* 的计数值
        new_part_counts = counts[indices] # 获取新索引对应的计数值
        broadcast_shape[dim] = new_d - orig_d
        new_part_slices = [slice(None)] * tensor.dim()
        new_part_slices[dim] = slice(orig_d, new_d)
        new_tensor[tuple(new_part_slices)] /= new_part_counts.view(broadcast_shape)
        
    return new_tensor

def _apply_net2net_rules(W_orig, new_shape, param_name, maps):
    """
    根据参数名，应用 Net2Net 规则。
    
    规则：
    - L 层 (输出): 只分裂 (Split)。
    - L+1 层 (输入): 分裂 (Split) + 补偿 (Compensate)。
    - C 维度 (n_embd) 比较特殊，它既是输入也是输出。
      我们遵循 Llama 的做法：将 MHA/FFN 的输入 (dim 0) 视为 L+1 (补偿)，
      输出 (dim 1) 视为 L (不补偿)。
    """
    
    # 默认从原始张量开始
    new_tensor = W_orig.clone()
    
    # --- 1D 向量 ---
    # (ln*.weight/bias, att.ln_x.weight/bias, ln_out.weight/bias)
    if param_name.startswith('ln') or 'ln_x' in param_name:
        # 视为 L+1 (输入)，应用补偿
        new_tensor = _expand_dim(new_tensor, 0, maps['C'], compensate=True)

    # --- 3D 向量 (1, 1, C) ---
    # (att.x_*, att.w0, att.a0, ...)
    elif W_orig.dim() == 3 and W_orig.shape[0:2] == (1, 1):
        # 视为 L+1 (输入)，应用补偿
        new_tensor = _expand_dim(new_tensor, 2, maps['C'], compensate=True)

    # --- 2D 矩阵 ---
    elif W_orig.dim() == 2:
        # A. Embedding: [V, C]
        if param_name == 'emb.weight':
            # L 层 (输出 C) -> 只分裂
            new_tensor = _expand_dim(new_tensor, 1, maps['C'], compensate=False)
        
        # B. LM Head: [V, C]
        elif param_name == 'head.weight':
            # (操作是 x @ W.T)，所以 C (dim 1) 是输入
            # L+1 层 (输入 C) -> 分裂 + 补偿
            new_tensor = _expand_dim(new_tensor, 1, maps['C'], compensate=True)
            
        # C. FFN 扩展层 (value): [C, F]
        elif param_name == 'ffn.value.weight':
            # L+1 (输入 C) -> 分裂 + 补偿
            new_tensor = _expand_dim(new_tensor, 0, maps['C'], compensate=True)
            # L (输出 F) -> 只分裂
            new_tensor = _expand_dim(new_tensor, 1, maps['F'], compensate=False)
            
        # D. FFN 收缩层 (key): [F, C]
        elif param_name == 'ffn.key.weight':
            # L+1 (输入 F) -> 分裂 + 补偿
            new_tensor = _expand_dim(new_tensor, 0, maps['F'], compensate=True)
            # L (输出 C) -> 只分裂
            new_tensor = _expand_dim(new_tensor, 1, maps['C'], compensate=False)
            
        # E. MHA (Q,K,V,R): [C, C]
        elif param_name in ['att.receptance.weight', 'att.key.weight', 'att.value.weight']:
            # L+1 (输入 C, dim 0) -> 分裂 + 补偿
            new_tensor = _expand_dim(new_tensor, 0, maps['C'], compensate=True)
            # L (输出 C, dim 1) -> 只分裂
            new_tensor = _expand_dim(new_tensor, 1, maps['C'], compensate=False)
            
        # F. MHA (O): [C, C]
        elif param_name == 'att.output.weight':
            # L+1 (输入 C, dim 0) -> 分裂 + 补偿
            new_tensor = _expand_dim(new_tensor, 0, maps['C'], compensate=True)
            # L (输出 C, dim 1) -> 只分裂
            new_tensor = _expand_dim(new_tensor, 1, maps['C'], compensate=False)
            
        # G. LoRA "1" 层 (w1, a1, v1, g1): [C, D_...]
        elif param_name in ['att.w1', 'att.a1', 'att.v1', 'att.g1']:
            lora_map_key = {
                'att.w1': 'D_decay', 'att.a1': 'D_aaa', 
                'att.v1': 'D_mv', 'att.g1': 'D_gate'
            }[param_name]
            # L+1 (输入 C, dim 0) -> 分裂 + 补偿
            new_tensor = _expand_dim(new_tensor, 0, maps['C'], compensate=True)
            # L (输出 D, dim 1) -> 只分裂
            new_tensor = _expand_dim(new_tensor, 1, maps[lora_map_key], compensate=False)
            
        # H. LoRA "2" 层 (w2, a2, v2, g2): [D_..., C]
        elif param_name in ['att.w2', 'att.a2', 'att.v2', 'att.g2']:
            lora_map_key = {
                'att.w2': 'D_decay', 'att.a2': 'D_aaa', 
                'att.v2': 'D_mv', 'att.g2': 'D_gate'
            }[param_name]
            # L+1 (输入 D, dim 0) -> 分裂 + 补偿
            new_tensor = _expand_dim(new_tensor, 0, maps[lora_map_key], compensate=True)
            # L (输出 C, dim 1) -> 只分裂
            new_tensor = _expand_dim(new_tensor, 1, maps['C'], compensate=False)
            
        # I. 特殊 R_K: [H, N] (N=head_size)
        elif param_name == 'att.r_k':
            # L (输出 H) -> 只分裂
            new_tensor = _expand_dim(new_tensor, 0, maps['H'], compensate=False)

    # 如果形状仍然不匹配 (例如，我们没有处理某个分支)，则回退到零填充
    if list(new_tensor.shape) != list(new_shape):
        print(f"  [警告] Net2Net 规则未覆盖 {key} ({W_orig.shape} -> {new_shape})。回退到零填充。")
        temp_tensor = torch.zeros(new_shape, dtype=W_orig.dtype, device=W_orig.device)
        orig_slices = tuple(slice(0, d) for d in W_orig.shape)
        temp_tensor[orig_slices] = W_orig
        return temp_tensor
        
    return new_tensor

########################################################################################################
# 辅助函数: Net2Net/Net2WiderNet (仅分裂，无补偿)
########################################################################################################
def pad_tensor_net2net(tensor, target_shape, param_name, state_map=None, dim_to_split=-1):
    """
    实现 Net2Net 的“神经元分裂”策略 。
    
    - 原始权重被复制到“左上角” 。
    - 新增的维度（行或列）通过“随机选择并复制”原始维度来填充 。
    
    参数:
    - tensor: 原始张量。
    - target_shape: 目标形状。
    - param_name: 参数名 (用于调试)。
    - state_map: (可选) 用于存储拆分信息的字典，以便后续层进行补偿。
    - dim_to_split: (可选) 指定哪个维度代表“神经元”并需要被拆分。
      - 默认-1 (最后一维)，例如 [V, C] -> [V, C_new]
      - 设为 0，例如 [C, F] -> [C_new, F]
    """
    orig_shape = tensor.shape
    new_shape = target_shape
    
    # 1. 创建一个目标形状的空张量
    new_tensor = torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)
    
    # 2. 将原始张量复制到“左上角” 
    orig_slices = tuple(slice(0, d) for d in orig_shape)
    new_tensor[orig_slices] = tensor.clone()
    
    # 3. --- 执行“神经元分裂”（随机复制）  ---
    
    # 我们需要确定哪个维度是“神经元”维度 (n)，哪个是“其他”维度。
    # 我们假设一次只扩展一个维度（这对你的 RWKV 模型来说是成立的）。
    
    if tensor.dim() == 1:
        # 1D 向量: [C_orig] -> [C_new]
        orig_d0 = orig_shape[0]
        new_d0 = new_shape[0]
        if new_d0 > orig_d0:
            # 随机选择 (new_d0 - orig_d0) 个原始索引
            indices = torch.randint(0, orig_d0, (new_d0 - orig_d0,), device=tensor.device)
            # 复制权重
            new_tensor[orig_d0:new_d0] = tensor[indices]

    elif tensor.dim() == 2:
        # 2D 矩阵: [D0_orig, D1_orig] -> [D0_new, D1_new]
        orig_d0, orig_d1 = orig_shape
        new_d0, new_d1 = new_shape

        # A. 扩展行 (dim 0)
        if new_d0 > orig_d0:
            indices_d0 = torch.randint(0, orig_d0, (new_d0 - orig_d0,), device=tensor.device)
            # 复制 *原始列* 的数据
            new_tensor[orig_d0:new_d0, 0:orig_d1] = tensor[indices_d0, :]
            
            # (如果 state_map 存在且我们需要拆分 dim 0)
            if state_map is not None and dim_to_split == 0:
                # 记录哪些行被复制了
                state_map['indices'] = indices_d0
                state_map['dim'] = 0

        # B. 扩展列 (dim 1)
        if new_d1 > orig_d1:
            indices_d1 = torch.randint(0, orig_d1, (new_d1 - orig_d1,), device=tensor.device)
            # 复制 *所有行* (包括新复制的行) 的数据
            new_tensor[0:new_d0, orig_d1:new_d1] = new_tensor[0:new_d0, indices_d1]
            
            # (如果 state_map 存在且我们需要拆分 dim 1)
            if state_map is not None and dim_to_split == 1:
                # 记录哪些列被复制了
                state_map['indices'] = indices_d1
                state_map['dim'] = 1
                
    elif tensor.dim() == 3 and orig_shape[0:2] == (1, 1):
        # 3D 向量: [1, 1, C_orig] -> [1, 1, C_new]
        orig_d2 = orig_shape[2]
        new_d2 = new_shape[2]
        if new_d2 > orig_d2:
            indices = torch.randint(0, orig_d2, (new_d2 - orig_d2,), device=tensor.device)
            new_tensor[0, 0, orig_d2:new_d2] = tensor[0, 0, indices]

    return new_tensor



def expand_model_width_net2net(original_state_dict, C_orig, C_new, head_size,
                               D_decay_orig, D_decay_new,
                               D_aaa_orig, D_aaa_new,
                               D_mv_orig, D_mv_new,
                               D_gate_orig, D_gate_new):
    """
    扩展模型的宽度 (n_embd) 和 LoRA (D) 维度，
    使用 Net2Net 的“分裂-补偿”策略。
    
    [警告] 此方法与 LayerNorm 不完全兼容，只能实现“近似”功能保持。
    
    """
    print(f"--- [模型宽度扩展 (Net2Net)] ---")
    print(f"原始 n_embd (C): {C_orig} -> 新 n_embd (C_new): {C_new}")
    
    # --- 1. 计算所有相关维度 ---
    F_orig = C_orig * 4
    F_new = C_new * 4
    
    if C_orig % head_size != 0 or C_new % head_size != 0:
        raise ValueError(f"n_embd (原始 {C_orig} 和新 {C_new}) 都必须能被 head_size ({head_size}) 整除。")
        
    H_orig = C_orig // head_size
    H_new = C_new // head_size
    
    print(f"FFN 维度 (F): {F_orig} -> {F_new}")
    print(f"Head 数量 (H): {H_orig} -> {H_new}")
    print(f"LoRA Dims: Decay({D_decay_orig}->{D_decay_new}), AAA({D_aaa_orig}->{D_aaa_new}), MV({D_mv_orig}->{D_mv_new}), Gate({D_gate_orig}->{D_gate_new})")
    
    # 获取一个设备 (用于创建 map)
    device = next(iter(original_state_dict.values())).device

    # --- 2. 预先创建所有维度的“分裂图” ---
    # 这是 Net2Net 策略的核心：所有层必须重用 *相同* 的随机分裂方案。
    print("正在为所有扩展维度创建随机分裂图...")
    split_maps = {
        'C': _create_split_map(C_orig, C_new, device),
        'F': _create_split_map(F_orig, F_new, device),
        'H': _create_split_map(H_orig, H_new, device),
        'D_decay': _create_split_map(D_decay_orig, D_decay_new, device),
        'D_aaa': _create_split_map(D_aaa_orig, D_aaa_new, device),
        'D_mv': _create_split_map(D_mv_orig, D_mv_new, device),
        'D_gate': _create_split_map(D_gate_orig, D_gate_new, device),
    }
    
    new_state_dict = OrderedDict()
    
    # --- 3. 遍历所有参数应用规则 ---
    for key, W_orig in original_state_dict.items():
        W_shape = list(W_orig.shape)
        new_shape = list(W_shape) # 默认：形状不变
        
        layer_id_str = None
        match = re.match(r"blocks\.(\d+)\.(.*)", key)
        if match:
            layer_id_str = match.group(1)
            param_name = match.group(2)
        else:
            param_name = key

        if layer_id_str == "0" and param_name in ['att.v0', 'att.v1', 'att.v2']:
            print(f"     > [清理] 正在跳过并删除 Block 0 中无效的键: {key}")
            continue

        # --- 4. 根据参数名确定新形状 (与你原始代码相同) ---
        
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

        # 2D LoRA-like
        elif param_name == 'att.w1':
            if W_shape == [C_orig, D_decay_orig]: new_shape = [C_new, D_decay_new]
        elif param_name == 'att.w2':
            if W_shape == [D_decay_orig, C_orig]: new_shape = [D_decay_new, C_new]
        elif param_name == 'att.a1':
            if W_shape == [C_orig, D_aaa_orig]: new_shape = [C_new, D_aaa_new]
        elif param_name == 'att.a2':
            if W_shape == [D_aaa_orig, C_orig]: new_shape = [D_aaa_new, C_new]
        elif param_name == 'att.v1':
            if W_shape == [C_orig, D_mv_orig]: new_shape = [C_new, D_mv_new]
        elif param_name == 'att.v2':
            if W_shape == [D_mv_orig, C_orig]: new_shape = [D_mv_new, C_new]
        elif param_name == 'att.g1':
            if W_shape == [C_orig, D_gate_orig]: new_shape = [C_new, D_gate_new]
        elif param_name == 'att.g2':
            if W_shape == [D_gate_orig, C_orig]: new_shape = [D_gate_new, C_new]
        
        # 2D 全连接
        elif param_name in [
            'att.receptance.weight', 'att.key.weight', 
            'att.value.weight', 'att.output.weight'
        ]:
            if W_shape == [C_orig, C_orig]: new_shape = [C_new, C_new]
        
        # 2D 特殊 r_k
        elif param_name == 'att.r_k':
            if W_shape[0] == H_orig: new_shape = [H_new, W_shape[1]]

        # 2D FFN 矩阵
        elif param_name == 'ffn.key.weight':
            if W_shape == [F_orig, C_orig]: new_shape = [F_new, C_new]
        elif param_name == 'ffn.value.weight':
            if W_shape == [C_orig, F_orig]: new_shape = [C_new, F_new]
            
        # 2D Embedding & Head
        elif param_name == 'emb.weight':
            if W_shape[1] == C_orig: new_shape = [W_shape[0], C_new]
        elif param_name == 'head.weight':
            if W_shape[1] == C_orig: new_shape = [W_shape[0], C_new]

        # --- 5. 执行 Net2Net 扩展 ---
        if W_shape == new_shape:
            new_state_dict[key] = W_orig.clone()
        else:
            print(f"     > 正在 Net2Net 扩展 {key}: {W_shape} -> {new_shape}")
            # 调用核心规则
            new_tensor = _apply_net2net_rules(W_orig, tuple(new_shape), param_name, split_maps)
            new_state_dict[key] = new_tensor

    print(f"--- [模型宽度扩展 (Net2Net) 完成] ---")
    return new_state_dict

def expand_model_scheme_1(original_state_dict, original_n_layer, 
                          interpolation_alpha=0.5, 
                          stack_last_layer: bool = False):
    """
    实现一个灵活的模型扩展方案。

    核心逻辑:
    - 中间层 (L_1 到 L_{L-2}): 应用 [复制, 插值] 模式。
      L_k' = interp(L_k, L_{k+1})
    - 边界层 (L_0, L_{L-1}): 行为由 `stack_last_layer` 控制。
      - stack_last_layer=True: 应用 [复制] 模式 (L_0, L_0 和 L_{L-1}, L_{L-1})
      - stack_last_layer=False: 不复制L_0和L_last

    [!! 修复 !!]
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
        elif (k == 0 or k == L - 1):
            if stack_last_layer:
                new_layers_config.append(('copy', [k])) # 应用 [复制, 复制]
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
    print(f"正在扩展深度: L={ORIGINAL_N_LAYER} -> L={ORIGINAL_N_LAYER * 2 - 2}")
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
    state_dict_L_C = expand_model_width_net2net(
        state_dict_L,
        C_orig=ORIGINAL_N_EMBD,
        C_new=NEW_N_EMBD,
        head_size=args.head_size_a,
        D_decay_orig=D_DECAY_LORA, D_decay_new=new_D_DECAY_LORA,
        D_aaa_orig=D_AAA_LORA, D_aaa_new=new_D_AAA_LORA,
        D_mv_orig=D_MV_LORA, D_mv_new=new_D_MV_LORA,
        D_gate_orig=D_GATE_LORA, D_gate_new=new_D_GATE_LORA
    )
    print("宽度扩展完成。")

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
    
    # (!!! 核心修改 !!!)
    # 将新的 D 维度添加到 args 对象中
    # 这样 RWKV_Tmix_x070.__init__ 才能正确构建模型
    args.D_DECAY_LORA = new_D_DECAY_LORA
    args.D_AAA_LORA = new_D_AAA_LORA
    args.D_MV_LORA = new_D_MV_LORA
    args.D_GATE_LORA = new_D_GATE_LORA
    print(f"     ... 使用 D_decay={args.D_DECAY_LORA}, D_aaa={args.D_AAA_LORA}, D_mv={args.D_MV_LORA}, D_gate={args.D_GATE_LORA}")
    
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
        prompt = "How to "
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
    # 结果也是乱说话：How to 👍 Team.Da.Fred_app…”, I_dedyeperapry-capcadecide