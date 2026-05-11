from compression.util import *
from utils.arithmeticcoding import decompress_from_bit_list, compress_to_bit_list
import math
from typing import Dict, Any, Tuple, Optional

import torch


import math
from typing import Dict, Any, Tuple, Optional

import torch


def _relative_boundary_factor(
    rel_stage: int,
    boundary_temp_boost: float = 0.25,
    boundary_decay: float = 0.55,
    tail_cool_rate: float = 0.03,
    tail_cool_start: int = 4,
    tail_cool_min: float = 0.85,
) -> float:
    """
    rel_stage:
        当前尺度距离“最后一个已传输尺度”的相对距离。
        - rel_stage = 1: 第一个缺失尺度（最关键）
        - rel_stage = 2: 第二个缺失尺度
        - ...

    返回一个温度乘子：
        - 第一个缺失尺度给更高温度
        - 后续逐渐回落
        - 再往后略微降温，避免高尺度过于随机
    """
    rel_stage = max(1, int(rel_stage))

    # 第一个缺失尺度温度更高，后面指数衰减
    boost = boundary_temp_boost * math.exp(-boundary_decay * float(rel_stage - 1))

    # 到更后面的缺失尺度，略微偏保守
    if rel_stage >= tail_cool_start:
        cool = max(tail_cool_min, 1.0 - tail_cool_rate * float(rel_stage - tail_cool_start + 1))
    else:
        cool = 1.0

    return (1.0 + boost) * cool


def apply_entropy_adaptive_temperature(
    raw_logits: torch.Tensor,
    si: int,
    last_observed_scale_idx: int,
    selective_ratio: float = 0.20,
    t0: float = 1.60,
    alpha: float = 0.45,
    theta: float = 0.55,
    base_temperature: float = 0.85,
    min_temperature: float = 0.10,
    max_temperature: float = 2.50,
    boundary_temp_boost: float = 0.25,
    boundary_decay: float = 0.55,
    tail_cool_rate: float = 0.03,
    tail_cool_start: int = 4,
    tail_cool_min: float = 0.85,
    eps: float = 1e-10,
    return_debug: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    """
    输入:
        raw_logits: [B, seq_len, 64]
            默认按 Infinity bitwise 格式解释为 [B, seq_len, 32, 2]
        si:
            当前正在生成的尺度 index
        last_observed_scale_idx:
            最后一个“已传输/已知”的尺度 index
            例如传输了前 3 个 token map，则 last_observed_scale_idx = 2
            若一个都没传，设为 -1
        selective_ratio:
            只对当前尺度中熵最高的前 selective_ratio 比例 token 做动态温度
        t0, alpha, theta:
            动态温度公式参数
            T = t0 * exp(-H_norm / alpha) + theta
            注意这里 H_norm 已经归一化到 [0, 1]
        base_temperature:
            对“不进行动态采样”的 token 使用的固定温度
        min_temperature, max_temperature:
            温度安全上下界

    输出:
        logits_scaled: [B, seq_len, 64]
        debug_info: 可选调试信息
    """
    if raw_logits.ndim != 3:
        raise ValueError(f"raw_logits should be [B, seq_len, 64], got shape={tuple(raw_logits.shape)}")
    if raw_logits.size(-1) % 2 != 0:
        raise ValueError(f"Last dim of raw_logits must be even, got {raw_logits.size(-1)}")

    B, L, V = raw_logits.shape
    num_bits = V // 2

    # [B, L, 64] -> [B, L, 32, 2]
    logits_bits = raw_logits.reshape(B, L, num_bits, 2)

    # bit 概率
    probs_bits = torch.softmax(logits_bits, dim=-1)

    # 每个 bit 的熵，单位 nat，最大值 ln(2)
    entropy_bits = -(probs_bits * torch.log(probs_bits.clamp_min(eps))).sum(dim=-1)  # [B, L, num_bits]

    # 归一化到 [0, 1]
    entropy_bits_norm = (entropy_bits / math.log(2.0)).clamp_(0.0, 1.0)  # [B, L, num_bits]

    # token 级不确定性：对 32 个 bit 的归一化熵取均值
    token_uncertainty = entropy_bits_norm.mean(dim=-1)  # [B, L]

    # 相对缺失边界：第一个缺失尺度最重要
    rel_stage = max(1, int(si - last_observed_scale_idx))
    rel_factor = _relative_boundary_factor(
        rel_stage=rel_stage,
        boundary_temp_boost=boundary_temp_boost,
        boundary_decay=boundary_decay,
        tail_cool_rate=tail_cool_rate,
        tail_cool_start=tail_cool_start,
        tail_cool_min=tail_cool_min,
    )

    # bit-wise 动态温度
    # 低熵 -> 温度更高一些
    # 高熵 -> 温度更低一些
    dynamic_temp_bits = t0 * torch.exp(-entropy_bits_norm / max(alpha, 1e-6)) + theta  # [B, L, num_bits]
    dynamic_temp_bits = dynamic_temp_bits * rel_factor

    # selective sampling: 只对最不确定的一部分 token 用动态温度
    selective_ratio = float(max(0.0, min(1.0, selective_ratio)))
    if selective_ratio <= 0.0:
        use_dynamic_mask = torch.zeros_like(token_uncertainty, dtype=torch.bool)
    elif selective_ratio >= 1.0:
        use_dynamic_mask = torch.ones_like(token_uncertainty, dtype=torch.bool)
    else:
        # 每个 batch 样本内部单独算分位数阈值
        threshold = torch.quantile(token_uncertainty, q=1.0 - selective_ratio, dim=1, keepdim=True)
        use_dynamic_mask = token_uncertainty >= threshold  # [B, L]

    # 未选中的 token 用固定低温
    base_temp_bits = torch.full_like(dynamic_temp_bits, fill_value=base_temperature)

    final_temp_bits = torch.where(
        use_dynamic_mask.unsqueeze(-1),
        dynamic_temp_bits,
        base_temp_bits,
    )

    final_temp_bits = final_temp_bits.clamp_(min=min_temperature, max=max_temperature)

    # 应用到 bit logits
    logits_bits_scaled = logits_bits / final_temp_bits.unsqueeze(-1)  # [B, L, num_bits, 2]
    logits_scaled = logits_bits_scaled.reshape_as(raw_logits)

    if not return_debug:
        return logits_scaled, None

    debug_info = {
        "rel_stage": rel_stage,
        "rel_factor": rel_factor,
        "token_uncertainty_mean": token_uncertainty.mean().item(),
        "token_uncertainty_max": token_uncertainty.max().item(),
        "dynamic_ratio": use_dynamic_mask.float().mean().item(),
        "temp_mean": final_temp_bits.mean().item(),
        "temp_min": final_temp_bits.min().item(),
        "temp_max": final_temp_bits.max().item(),
    }
    return logits_scaled, debug_info


def calc_token_entropy(p_list):
    ent = 0.0
    for p in p_list:
        if 0.0 < p < 1.0:
            ent += -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    return ent
def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def decompress_cfg(infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_leak, gt_ls_Bl, cfg_list=3, tau_list=0.5, cfg_insertion_layer=[0]):
    # infinity.rng.manual_seed(9306)
    rng = infinity.rng
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(vae_scale_schedule)
        cfg_list = [cfg_list] * len(vae_scale_schedule)
    label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    B = 1
    if any(np.array(cfg_list) != 1):
        bs = 2*B
        kv_compact_un = kv_compact.clone()
        total = 0
        for le in lens:
            kv_compact_un[total:total+le] = (infinity.cfg_uncond)[:le]
            total += le
        kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
        cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
    else:
        bs = B
    
    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
    kv_compact = infinity.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    with torch.amp.autocast('cuda', enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)

    abs_cfg_insertion_layers = []
    add_cfg_on_logits, add_cfg_on_probs = False, False
    leng = len(infinity.unregistered_blocks)
    for item in cfg_insertion_layer:
        if item == 0: # add cfg on logits
            add_cfg_on_logits = True
        elif item == 1: # add cfg on probs
            add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
        elif item < 0: # determine to add cfg at item-th layer's output
            assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={infinity.num_block_chunks}'
            abs_cfg_insertion_layers.append(leng+item)
        else:
            raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
    accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
    idx_Bl_list, idx_Bld_list = [], []
    num_stages_minus_1 = len(vae_scale_schedule)-1
    summed_codes = 0
    cur_L = 0
    for si, pn in enumerate(vae_scale_schedule):
        cfg = cfg_list[si]
        cur_L += np.array(pn).prod()
        need_to_pad = 0
        attn_fn = None
        if infinity.use_flex_attn:
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[:(si+1)]), None)
        layer_idx = 0
        for block_idx, b in enumerate(infinity.block_chunks):
            # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
            if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            if not infinity.add_lvl_embeding_only_first_block: 
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            
            for m in b.module:
                last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=vae_scale_schedule, rope2d_freqs_grid=infinity.rope2d_freqs_grid, scale_ind=si)
                layer_idx += 1
        
        # ======== 替换 decompress_cfg 中的解码与采样逻辑 ========
        
        if (cfg != 1) and add_cfg_on_logits:
            context = infinity.get_logits(last_stage, cond_BD)
            cond_context, uncond_context = context.chunk(2, dim=0)
            # 在 Context 层面做 CFG 混合 (因为 MBM Head 变成了强依赖输入的非线性网络)
            context = uncond_context + cfg * (cond_context - uncond_context)
            # # ==================== 新增：HIERAMP 空间 CFG 计算 ====================
            # spatial_mask = infinity.get_hieramp_mask()
            # if spatial_mask is not None:
            #     # print(f"spatial_mask is not None.{spatial_mask.shape}")
            #     # print(spatial_mask)
            #     # 获取的 spatial_mask 维度是 [B, seq_len]
            #     # context 的维度是 [B, seq_len, Dim]，所以需要 unsqueeze(-1) 来广播
            #     spatial_mask = spatial_mask[:B].unsqueeze(-1) 
                
            #     # 定义 HIERAMP 放大倍数 (背景保留原始 cfg，前景放大至 amp_factor)
            #     amp_factor = 8.0 # 论文推荐值，可根据效果调节 3.0~5.0
            #     spatial_cfg_map = cfg + spatial_mask * (amp_factor - cfg)
            # else:
            #     # print("spatial_mask is None")
            #     spatial_cfg_map = cfg # 降级方案
                
            # # 在 Context 层面做“空间自适应”的 CFG 混合
            # context = uncond_context + spatial_cfg_map * (cond_context - uncond_context)
            # # =====================================================================
        else:
            context = infinity.get_logits(last_stage[:B], cond_BD[:B])

        # === 真正的 MBM 多步渐进采样 (Progressive Unmasking) ===
        B_inf, L_inf, num_bits = context.shape[0], context.shape[1], infinity.codebook_dim
        
        # 初始化：全部未知
        final_bits = torch.zeros((B_inf, L_inf, num_bits), device=context.device, dtype=torch.float32)
        bit_mask = torch.zeros((B_inf, L_inf, num_bits), device=context.device, dtype=torch.float32)
        
        num_steps = 3 # BAR 论文推荐的 4 步调度
        
        for step in range(num_steps):
            # 前向传播：模型看见已被锁定的 final_bits
            mbm_logits = infinity.mbm_head(context, final_bits, bit_mask)
            probs = torch.sigmoid(mbm_logits / tau_list[si])
            
            # 计算置信度
            confidence = torch.abs(probs - 0.5)
            # 把已经锁定的比特置信度设为极低 (-1)，防止重复选择
            confidence = torch.where(bit_mask > 0.5, torch.tensor(-1.0, device=context.device), confidence)
            
            # 计算当前步骤需要锁定的总比特数
            bits_to_lock_total = (num_bits // num_steps) * (step + 1)
            if step == num_steps - 1:
                bits_to_lock_total = num_bits # 最后一步全部锁定
                
            # 计算本轮还要新锁定多少个 (处理不能整除的情况)
            current_locked_count = int(bit_mask[0, 0].sum().item())
            bits_to_add = bits_to_lock_total - current_locked_count
            
            if bits_to_add > 0:
                # 找到最确信的 bits_to_add 个比特
                _, topk_indices = torch.topk(confidence, bits_to_add, dim=-1)
                # 更新 mask (设为 1.0)
                bit_mask.scatter_(-1, topk_indices, 1.0)
                
            # 更新已确定的比特值 (采用确定的 Argmax)
            # current_preds = (probs > 0.5).float()
            # ================= 新版：Bit-level Nucleus Sampling (复刻自然感) =================
            # 定义一个置信度阈值，类似于原版的 top_p 参数 (可调，推荐 0.85 ~ 0.95)
            # 这个值越大，画面越具有随机性和细节；越小，画面越平滑。
            p_threshold = 0.9
            
            # 判断哪些 Bit 模型是极度自信的
            is_confident = (probs > p_threshold) | (probs < (1.0 - p_threshold))
            
            # 对于极其自信的位，使用确定性结果 (保证主体结构不崩)
            deterministic_preds = (probs > 0.5).float()
            
            # 对于模棱两可的位，掷骰子！(引入受控随机性，打破网格，生成自然纹理)
            # torch.bernoulli 会根据 probs 的概率分布返回 0 或 1
            stochastic_preds = torch.bernoulli(probs)
            
            # 融合两者：自信的用确定的，不自信的掷骰子
            current_preds = torch.where(is_confident, deterministic_preds, stochastic_preds)
            # =========================================================================
            final_bits = torch.where(bit_mask > 0.5, current_preds, final_bits)
            
        # 迭代结束，赋值回原变量，转回 long 类型继续后续操作
        idx_Bld = final_bits.long()
        
        
        if si <= gt_leak:
            idx_Bld = gt_ls_Bl[si]
        idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1)
        idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
        if si != num_stages_minus_1:
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
            last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
            last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
            last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
        else:
            summed_codes += codes
        if si != num_stages_minus_1:
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs//B, 1, 1)
    for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
    img = vae.decode(summed_codes.squeeze(-3))
    img = (img + 1) / 2
    # img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8)
    return img

def decoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, trans_list, help_list):
    pass


def decoding(args, infinity, vae, vae_scale_schedule, prompt, text_tokenizer, text_encoder, gt_ls_Bl, trans_list, help_list, tau_list=0.5, cfg_insertion_layer=[0]):
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(vae_scale_schedule)
    label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, prompt)
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
    bs = B = 1
    kv_compact = infinity.text_norm(kv_compact)
    sos = cond_BD = infinity.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) 
    kv_compact = infinity.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
    ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
    last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + infinity.pos_start.expand(bs, 1, -1)
    with torch.amp.autocast('cuda', enabled=False):
        cond_BD_or_gss = infinity.shared_ada_lin(cond_BD.float()).float().contiguous()
    for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
    abs_cfg_insertion_layers = []
    add_cfg_on_logits, add_cfg_on_probs = False, False
    leng = len(infinity.unregistered_blocks)
    for item in cfg_insertion_layer:
        if item == 0: # add cfg on logits
            add_cfg_on_logits = True
        elif item == 1: # add cfg on probs
            add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
        elif item < 0: # determine to add cfg at item-th layer's output
            assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={infinity.num_block_chunks}'
            abs_cfg_insertion_layers.append(leng+item)
        else:
            raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
    num_stages_minus_1 = len(vae_scale_schedule)-1
    summed_codes = 0
    cur_L = 0
    decode_idx = []
    for si, pn in enumerate(vae_scale_schedule):
        cur_L += np.array(pn).prod()
        need_to_pad = 0
        attn_fn = None
        if infinity.use_flex_attn:
            attn_fn = infinity.attn_fn_compile_dict.get(tuple(vae_scale_schedule[:(si+1)]), None)
        layer_idx = 0
        for block_idx, b in enumerate(infinity.block_chunks):
            # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
            if infinity.add_lvl_embeding_only_first_block and block_idx == 0:
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            if not infinity.add_lvl_embeding_only_first_block: 
                last_stage = infinity.add_lvl_embeding(last_stage, si, vae_scale_schedule, need_to_pad=need_to_pad)
            
            for m in b.module:
                last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=vae_scale_schedule, rope2d_freqs_grid=infinity.rope2d_freqs_grid, scale_ind=si)
                layer_idx += 1
                
        # 1. 提取 Context
        context = infinity.get_logits(last_stage[:B], cond_BD[:B])
        # 【新代码】
        mbm_logits = infinity.mbm_head(context)
        
        # 3. 经过 Sigmoid 得到概率，并转换为 [P(x=0), P(x=1)] 的格式
        prob_1 = torch.sigmoid(mbm_logits / tau_list[si])
        prob_0 = 1.0 - prob_1
        prob_tensor = torch.stack([prob_0, prob_1], dim=-1) # [B, seq_len, 32, 2]
        
        prob_tensor = prob_tensor.view(-1, 2)
        prob = prob_tensor[:, 0].cpu().tolist() # 取 P(x=0) 的概率，与 Sender 完美对齐
        bit_string = trans_list[si]
        h_string = help_list[si]
        decompressed_string = []
        
        # 直接使用 j 索引，因为发送端的比特序列和指令序列长度是 1:1 对齐的
        for j in range(len(h_string)):
            p_token = prob[j * args.vae_type : (j + 1) * args.vae_type]
            flag = h_string[j]

            if flag == 0:
                # 状态 0：透传了 32 bits 原文
                decompressed_string.extend(bit_string[j])

            elif flag == 1:
                # 状态 1：算术解码恢复出 32 bits
                dec_str = decompress_from_bit_list(bit_string[j], args.vae_type, p_token)
                decompressed_string.extend(dec_str)

            else:
                # Flag 2 已经被彻底废弃
                raise ValueError(f"Unknown flag: {flag}")

        dec_idx = torch.tensor(decompressed_string).to(dtype=torch.int32).to(device=context.device)
        dec_idx = dec_idx.reshape(B, pn[1]*pn[2], -1)
        decode_idx.append(dec_idx)
        idx_Bld = dec_idx.reshape(B, pn[1], pn[2], -1)
        idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
        if si != num_stages_minus_1:
            summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
            last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
            last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
            last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
            last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
        else:
            summed_codes += codes
        if si != num_stages_minus_1:
            last_stage = infinity.word_embed(infinity.norm0_ve(last_stage))
            last_stage = last_stage.repeat(bs//B, 1, 1)
    for b in infinity.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)

    return decode_idx