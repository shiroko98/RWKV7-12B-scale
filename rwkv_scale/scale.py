import torch, types, os, gc, math, json
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import re # <-- 用于匹配参数名
from collections import OrderedDict # <-- 用于构建新的 state_dict

np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

args = types.SimpleNamespace()

########################################################################################################
# 核心配置：模型扩展
########################################################################################################

# 路径到你原始的、较小的模型
ORIGINAL_MODEL_PATH = "/mnt/lab/Models/pt/rwkv7-g0a2-7.2b-20251005-ctx4096.pth"
# 原始模型的层数 (L)
ORIGINAL_N_LAYER = 32 
# 插值后新模型的保存路径
EXPANDED_MODEL_PATH = "/mnt/data/Codes/RWKV/RWKV-Scale/models/rwkv_state_dict_expanded_2L-4.pth"
# 插值系数 (0.5 = 50/50 混合)
INTERPOLATION_ALPHA = 0.5

# True  = 方案 1: [L_1, L_1', ..., L_L, L_L']  (总层数 2*L)
# False = 方案 2: [L_1, L_1', ..., L_{L-1}', L_L] (总层数 2*L - 1)
STACK_LAST_LAYER = False

# --- 宽度扩展配置 (方案二) ---
# 原始 args.n_embd (C_orig)
ORIGINAL_N_EMBD = 4096 
# 目标 args.n_embd (C_new) (3584 = 2560 * 1.4, 且 3584 % 64 == 0)
NEW_N_EMBD = 6144
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 第一层（block0）（没有v0,v1,v2）和最后一层（紧连输出层，建议不动/或者选择复制？不然层数可能不是2的次幂），然后中间的层，进行前后1层的复制和插值（例如，对于block 1'，它的插值层是block1 和 block2），然后函数头可以加一个参数，选择是否根据STACK_LAST_LAYER，是不动，还是直接复制第一层和最后一层
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

########################################################################################################
# 原始模型配置 (来自你的推理脚本)
########################################################################################################

# for 0.1B
# !!! 注意：这里的 args.n_layer 将在扩展后被自动覆盖 !!!
# args.n_layer = ORIGINAL_N_LAYER 
# args.n_embd = 2560
# D_DECAY_LORA = 96
# D_AAA_LORA = 96
# D_MV_LORA = 64
# D_GATE_LORA = 320

# for 7.2B
# !!! 注意：这里的 args.n_layer 将在扩展后被自动覆盖 !!!
args.n_layer = ORIGINAL_N_LAYER 
args.n_embd = 4096
D_DECAY_LORA = 128
# D_AAA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32)) = 128
D_AAA_LORA = 128
# D_MV = max(32, int(round((1.3 * (C**0.5)) / 32) * 32)) = 96
D_MV_LORA = 96
# D_GATE = max(32, int(round((0.6 * (C**0.8)) / 32) * 32)) = 448
D_GATE_LORA = 480

args.vocab_size = 65536

# DTYPE = torch.bfloat16
DTYPE = torch.half # better

args.head_size_a = 64 # don't change
HEAD_SIZE = args.head_size_a

USE_CUDA_KERNEL = False # False => UNOPTIMIZED, VERY SLOW

MyModule = nn.Module
# MyFunction = torch.jit.script_method
# MyStatic = torch.jit.script

########################################################################################################
# RWKV Tokenizer (来自你的推理脚本)
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
        C = args.n_embd

        self.x_r = nn.Parameter(torch.empty(1,1,C))
        self.x_w = nn.Parameter(torch.empty(1,1,C))
        self.x_k = nn.Parameter(torch.empty(1,1,C))
        self.x_v = nn.Parameter(torch.empty(1,1,C))
        self.x_a = nn.Parameter(torch.empty(1,1,C))
        self.x_g = nn.Parameter(torch.empty(1,1,C))

        self.w0 = nn.Parameter(torch.empty(1,1,C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        self.a0 = nn.Parameter(torch.empty(1,1,C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        self.v0 = nn.Parameter(torch.empty(1,1,C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

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

        self.ln0 = nn.LayerNorm(args.n_embd) # only used in block 0, should be fused with emb
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
# 核心算法：宽度扩展
########################################################################################################

def expand_model_width(original_state_dict, C_orig, C_new, head_size=64):
    """
    扩展模型的宽度 (n_embd)，使用插值来调整所有相关权重。
    """
    print(f"--- [模型宽度扩展] ---")
    print(f"原始 n_embd (C): {C_orig} -> 新 n_embd (C_new): {C_new}")
    
    # --- 1. 计算所有相关维度 ---
    # !! 注意 !! 这里的 FFN 维度 (F) 基于你的 *第一个* 脚本 (4*C)，而不是第二个 (model.py)
    F_orig = C_orig * 4
    F_new = C_new * 4
    
    if C_orig % head_size != 0 or C_new % head_size != 0:
        raise ValueError(f"n_embd (原始 {C_orig} 和新 {C_new}) 都必须能被 head_size ({head_size}) 整除。")
        
    H_orig = C_orig // head_size
    H_new = C_new // head_size
    
    print(f"FFN 维度 (F): {F_orig} -> {F_new}")
    print(f"Head 数量 (H): {H_orig} -> {H_new}")
    
    new_state_dict = OrderedDict()
    
    # --- 2. 遍历所有参数进行插值 ---
    for key, W_orig in original_state_dict.items():
        W_shape = list(W_orig.shape)
        new_shape = list(W_shape) # 默认：形状不变
        
        match = re.match(r"blocks\.(\d+)\.(.*)", key)
        if match:
            param_name = match.group(2)
        else:
            param_name = key # 顶层参数 (emb, ln_out, head)

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

        # 2D LoRA-like (Input expands) (C_orig, D) -> (C_new, D)
        elif param_name in ['att.w1', 'att.a1', 'att.v1', 'att.g1']:
            if W_shape[0] == C_orig: new_shape = [C_new, W_shape[1]]

        # 2D LoRA-like (Output expands) (D, C_orig) -> (D, C_new)
        elif param_name in ['att.w2', 'att.a2', 'att.v2', 'att.g2']:
            if W_shape[1] == C_orig: new_shape = [W_shape[0], C_new]
        
        # 2D 全连接 (C_orig, C_orig) -> (C_new, C_new)
        elif param_name in [
            'att.receptance.weight', 'att.key.weight', 
            'att.value.weight', 'att.output.weight'
        ]:
            if W_shape == [C_orig, C_orig]: new_shape = [C_new, C_new]
            
        # 2D 特殊 r_k (H_orig, N) -> (H_new, N)
        elif param_name == 'att.r_k':
            if W_shape[0] == H_orig: new_shape = [H_new, W_shape[1]]

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
        else:
            print(f"  > 正在插值 {key}: {W_shape} -> {new_shape}")
            new_state_dict[key] = interpolate_tensor(W_orig, tuple(new_shape))
            
    print(f"--- [模型宽度扩展完成] ---")
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

########################################################################################################
# 主执行函数：扩展并运行推理
########################################################################################################

def run_expansion_and_inference():
    
    # --- 1. 执行模型扩展 ---
    print(f"正在加载原始模型: {ORIGINAL_MODEL_PATH}")
    # 确保在 CPU 上加载，避免 GPU 内存问题
    original_state_dict = torch.load(ORIGINAL_MODEL_PATH, map_location="cpu")
    
    new_state_dict, new_L = expand_model_scheme_1(
        original_state_dict, 
        ORIGINAL_N_LAYER, 
        interpolation_alpha=INTERPOLATION_ALPHA,
        stack_last_layer=STACK_LAST_LAYER
    )
    
    print(f"正在保存扩展后的模型到: {EXPANDED_MODEL_PATH}")
    torch.save(new_state_dict, EXPANDED_MODEL_PATH)
    
    # 释放内存
    del original_state_dict
    gc.collect()

    # --- 2. 准备运行推理 ---
    
    # !! 关键：更新 args 以匹配新模型 !!
    print(f"\n--- [开始推理] ---")
    print(f"使用新层数配置: args.n_layer = {new_L}")
    global args
    args.n_layer = new_L
    
    # (确保 dim_ffn 和 dim_att 被正确设置，以防万一)
    if not hasattr(args, 'dim_att'):
        args.dim_att = args.n_embd
    if not hasattr(args, 'dim_ffn'):
        args.dim_ffn = args.n_embd * 4

    # --- 3. 加载新模型并运行推理 ---
    
    print(f"正在加载新权重 {EXPANDED_MODEL_PATH} 到 RWKV(args) 实例...")
    # 我们可以直接使用内存中的 new_state_dict，或者从文件重新加载
    # model_params = new_state_dict
    model_params = torch.load(EXPANDED_MODEL_PATH, map_location="cpu")

    for key, value in model_params.items():
        total_params += value.numel()
        # print(key, value.shape)
        # print(f"Parameter: {key}, value: {value}, Shape: {value.shape}, Dtype: {value.dtype}")

    print(f"Total parameters: {total_params}")

    with torch.no_grad():
        # 实例化新模型，此时 RWKV(args) 会使用 args.n_layer = new_L 来构建 blocks
        model = RWKV(args).to(dtype=DTYPE).cpu()
        
        # 加载我们精心构建的 state_dict
        # strict=False 是安全的，以防原始模型中有 x070 v_first 相关的特定参数
        model.load_state_dict(model_params, strict=False)
        print("模型加载成功。")

        # --- 4. 运行原始的推理循环 ---
        prompt = "User: 帮我"
        input_ids = tokenizer.encode(prompt)
        print(f'{prompt}', end='', flush=True)

        input_tensor = torch.tensor(input_ids).reshape(1, -1).cpu()
        generated_tokens = input_ids.copy()
        output_text = prompt

        for _ in range(100): # 减少到100个token以便快速测试
            out = model.forward(input_tensor)
            logits = out[0, -1]
            probs = F.softmax(logits.float(), dim=-1)

            # # 贪婪解码：
            # next_token = torch.argmax(probs).item()
            
            # Top-k 采样
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
    run_expansion_and_inference()