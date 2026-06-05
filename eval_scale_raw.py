import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ["CC"]  = "gcc"
os.environ["CXX"] = "g++"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
sys.path.append("/workspace/Infinity_codec")

import torch, torchvision
import zlib  
import time  
import re
import json
import argparse
import cv2
import pandas as pd
import numpy as np

# Metrics
import lpips
from pytorch_msssim import ms_ssim
from DISTS_pytorch import DISTS

torch.cuda.set_device(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tools.run_infinity import *
from compression.util import *
from compression.sender_raw import encoding
from compression.reciever_raw import decompress_cfg, decoding

# --- Prompt Arithmetic Codec for T5 ---
def _encode_prompt_ids_t5(tokenizer, prompt: str, max_len: int = 512):
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        add_special_tokens=True,
    )
    ids = enc["input_ids"][0].tolist()
    return list(map(int, ids))


def _arith_encode_ids_packet(ids):

    import torchac

    if ids is None or len(ids) == 0:
        return {
            "payload": b"",
            "cdf_1d": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "unique_chars": [],
            "char_len": 0,
            "ids_len": 0,
            "bits": 0,
        }

    text = ",".join(map(str, ids))

    char_freq = {}
    for ch in text:
        if ch.isdigit() or ch == ",":
            char_freq[ch] = char_freq.get(ch, 0) + 1

    unique_chars = sorted(char_freq.keys())
    total = sum(char_freq.values())
    prob = [char_freq[c] / total for c in unique_chars]

    cdf_1d = torch.zeros(len(unique_chars) + 1, dtype=torch.float32)
    cdf_1d[1:] = torch.cumsum(torch.tensor(prob, dtype=torch.float32), dim=0)
    cdf_1d[-1] = 1.0

    L = len(text)
    cdf = cdf_1d.view(1, 1, -1).expand(1, L, -1).contiguous()
    sym = torch.tensor([unique_chars.index(ch) for ch in text], dtype=torch.int16).view(1, -1)

    payload = torchac.encode_float_cdf(cdf, sym, check_input_bounds=True)

    packet = {
        "payload": payload,
        "cdf_1d": cdf_1d,
        "unique_chars": unique_chars,
        "char_len": L,
        "ids_len": len(ids),
        "bits": len(payload) * 8,
    }
    return packet


def _arith_decode_ids_packet(packet):
    """
    从 packet 恢复出 T5 input_ids
    """
    import torchac

    if packet["char_len"] == 0:
        return []

    cdf = packet["cdf_1d"].view(1, 1, -1).expand(1, packet["char_len"], -1).contiguous()
    decoded_sym = torchac.decode_float_cdf(cdf, packet["payload"])
    decoded_sym = decoded_sym.view(-1).tolist()

    text = "".join(packet["unique_chars"][int(i)] for i in decoded_sym)
    ids = [int(x) for x in text.split(",") if x != ""]
    return ids


def _decode_prompt_text_from_ids_t5(tokenizer, ids):
    """
    最简单接法：恢复字符串，再喂现有 decoding/decompress_cfg。
    """
    if ids is None or len(ids) == 0:
        return ""
    try:
        return tokenizer.decode(ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(ids)
# ----------------------------------

def _prompt_zlib_bytes(prompt: str) -> int:
    b = prompt.encode("utf-8", errors="ignore")
    return len(zlib.compress(b))
# ----------------------------------

def compress(args, vae, scale_schedule, scale_q, infinity, text_tokenizer, text_encoder, img_path, text):
    inp_B3HW = load_img(img_path, args)
    raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW.to(device), scale_schedule=scale_schedule)
    x_BLC_wo_prefix, gt_ms_idx_Bl = mask_quant(vae, scale_q, raw_features, device)
    trans_list, help_list, bpp = encoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl)

    return trans_list, help_list, bpp, gt_ms_idx_Bl

# --- Modified Decompress & Metric Evaluator ---
def decompress_modified(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, 
                        gt_ms_idx_Bl, rec_base_path, img_path, img_name, trans_list, help_list, device, 
                        h, w, prompt_bpp, lpips_fn, dists_fn):
    dec_idx = decoding(args, infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_ms_idx_Bl, trans_list, help_list)
    
    scale_results = {}
    
    # Read original image once for all scales comparison
    std_img_cv = cv2.imread(img_path)
    std_img_cv = cv2.cvtColor(std_img_cv, cv2.COLOR_BGR2RGB)
    std_tensor = torch.tensor(std_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    for gt_leak in range(0, len(gt_ms_idx_Bl)):
        # 1. Calculate BPP
        sum_bit = 0
        for i in range(gt_leak + 1):
            sum_bit += len(help_list[i])
            for j in range(len(trans_list[i])):
                sum_bit += len(trans_list[i][j])
        
        current_img_bpp = sum_bit / (h * w)
        current_total_bpp = current_img_bpp + prompt_bpp

        # 2. Save image
        scale_folder = os.path.join(rec_base_path, f"scale_{gt_leak}")
        os.makedirs(scale_folder, exist_ok=True)
        img = decompress_cfg(infinity, vae, scale_schedule, text, text_tokenizer, text_encoder, gt_leak, dec_idx)
        rec_img_path = os.path.join(scale_folder, img_name)
        torchvision.utils.save_image(img.cpu(), rec_img_path)

        # 3. Calculate Metrics directly
        rec_img_cv = cv2.imread(rec_img_path)
        rec_img_cv = cv2.cvtColor(rec_img_cv, cv2.COLOR_BGR2RGB)
        rec_tensor = torch.tensor(rec_img_cv).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # PSNR
        mse = torch.mean((std_tensor - rec_tensor) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item() if mse.item() > 0 else float('inf')
        
        # MS-SSIM
        msssim_val = ms_ssim(std_tensor, rec_tensor, data_range=1.0, size_average=True).item()

        # LPIPS (LPIPS takes [-1, 1])
        lpips_val = lpips_fn(std_tensor.to(device) * 2 - 1, rec_tensor.to(device) * 2 - 1).item()

        # DISTS (DISTS takes [0, 1])
        dists_val = dists_fn(std_tensor.to(device), rec_tensor.to(device)).item()

        # Store results for this scale
        scale_results[str(gt_leak)] = {
            "bpp": current_total_bpp,
            "psnr": psnr,
            "msssim": msssim_val,
            "lpips": lpips_val,
            "dists": dists_val
        }

    return scale_results

if __name__ == "__main__":

    initial_model_path='/workspace/Infinity_codec/local_output/stage2_1024_layer12c4_16vae_nisp_code_L3_scale0-12_mlp256_w0.03/ar-ckpt-giter010K-ep0-iter10000-last.pth'
    # initial_model_path='/workspace/CKPT/Infinity/slim-ckpt-giter010K-ep0-iter10000-last.pth'

    vae_path='/workspace/CKPT/Infinity/infinity_vae_d16.pth'
    text_encoder_ckpt = '/workspace/CKPT/flan-t5-xl'
    
    args=argparse.Namespace(
        pn='1M',
        model_path=initial_model_path, # Will be dynamically overridden
        cfg_insertion_layer=0,
        vae_type=16,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        # model_type='infinity_2b',
        model_type='infinity_layer12',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/workspace/Infinity_codec/local_output/stage2_1024_layer12c4_16vae_nisp_code_L3_scale0-12_mlp256_w0.03/cache',
        enable_model_cache=1,
        checkpoint_type='torch',
        seed=0,
        bf16=0,
        rec_path='',
        
        add_prompt_bits=1,
        prompt_bits_mode='arith',
        tlen=512,
        keep_gpu_busy=1 
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    vae = load_visual_tokenizer(args)

    # Initialize metrics evaluators once to save memory/time
    lpips_evaluator = lpips.LPIPS(net='alex').to(device)
    dists_evaluator = DISTS().to(device)

    json_data = []
    with open('/workspace/ARPC-main/data/DIV2K.json', 'rt') as f:
        for line in f:
            json_data.append(json.loads(line))

    # ==========================================
    # 动态获取并排序等待测试的模型列表
    # ==========================================
    model_dir = os.path.dirname(initial_model_path)
    base_filename = os.path.basename(initial_model_path)

    match_init = re.search(r'-iter(\d+)-', base_filename)
    if not match_init:
        raise ValueError("无法在给定的 initial_model_path 中找到有效的 '-iterXXXX-' 结构。")
    start_iter = int(match_init.group(1))

    # 扫描目录，找到所有带有 -iter- 的 .pth 文件
    available_models = []
    for f in os.listdir(model_dir):
        if f.endswith('.pth') and 'iter' in f:
            m = re.search(r'-iter(\d+)-', f)
            if m:
                available_models.append((int(m.group(1)), f))
    
    # 按照 iter 大小对模型进行升序排序，并且过滤掉之前的 iter
    available_models.sort(key=lambda x: x[0])
    models_to_test = [f for it, f in available_models if it >= start_iter]

    print(f"[*] 成功扫描到 {len(models_to_test)} 个待测试模型。")

    # ==========================================
    # 开始自动化测试循环
    # ==========================================
    for current_filename in models_to_test:
        current_model_path = os.path.join(model_dir, current_filename)
        current_iter = int(re.search(r'-iter(\d+)-', current_filename).group(1))

        print(f"\n{'='*60}")
        print(f"[*] 开始测试模型: {current_filename} (Iter: {current_iter})")
        print(f"{'='*60}")

        # 动态更新 args
        args.model_path = current_model_path
        args.rec_path = f'/workspace/Infinity_codec/results/stage2_1024_layer12c4_16vae_nisp_code_L3_scale0-12_mlp256_w0.03/DIV2K_ori_{current_iter}'

        if not os.path.exists(args.rec_path):
            os.makedirs(args.rec_path)

        # 重新加载对应权重的 infinity 模型
        infinity = load_transformer(vae, args)
        
        # 存储当前模型下所有图片的指标数据
        dataset_metrics_data = []

        # 执行数据集评估
        for data in json_data:
            img_path, text = data['img_path'], data['txt']
            img_name = img_path.split('/')[-1] 
            
            inp_B3HW = load_img(img_path, args)
            b, c, h, w = inp_B3HW.shape
            h_div_w = h / w
            h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
            scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
            scale_q = [(scale_schedule[i][0], scale_schedule[i][1], scale_schedule[i][2], 
                        int((i+1)//((len(scale_schedule)//3)+1)+2)) for i in range(len(scale_schedule))]
            
            # Prompt bits
            prompt_bits = 0
            prompt_packet = None
            text_for_decode = text

            if args.add_prompt_bits:
                if args.prompt_bits_mode == "arith":
                    prompt_ids = _encode_prompt_ids_t5(text_tokenizer, text, max_len=args.tlen)
                    prompt_packet = _arith_encode_ids_packet(prompt_ids)
                    prompt_bits = prompt_packet["bits"]

                    recovered_ids = _arith_decode_ids_packet(prompt_packet)
                    assert recovered_ids == prompt_ids, "Prompt arithmetic decode mismatch!"
                    text_for_decode = _decode_prompt_text_from_ids_t5(text_tokenizer, recovered_ids)

                elif args.prompt_bits_mode == "zlib":
                    prompt_bits = _prompt_zlib_bytes(text) * 8
                    text_for_decode = text

            prompt_bpp = prompt_bits / (h * w) if args.add_prompt_bits else 0.0
            
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                trans_list, help_list, bpp, gt_ms_idx_Bl = compress(
                    args, vae, scale_schedule, scale_q, infinity, 
                    text_tokenizer, text_encoder, img_path, text
                )
                
                print(f"Processing {img_name}... [Prompt Bits: {prompt_bits} | Prompt BPP: {prompt_bpp:.6f}]")
                
                scale_metrics = decompress_modified(
                args, infinity, vae, scale_schedule, text_for_decode,
                text_tokenizer, text_encoder, gt_ms_idx_Bl,
                args.rec_path, img_path, img_name, trans_list, help_list, device,
                h, w, prompt_bpp, lpips_evaluator, dists_evaluator
            )
            
            # 存入整体验证数组
            dataset_metrics_data.append({
                "image_name": img_name,
                "text": text,
                "prompt_bpp": prompt_bpp,
                "scales_data": scale_metrics
            })
            
            # 终端简单打印跟踪 (只打印 BPP 和 PSNR)
            bpp_list = [f"{v['bpp']:.8f}" for k, v in scale_metrics.items()]
            print(f"Finished {img_name}, Scales total bpp: {bpp_list}")

        # ==========================================
        # 当前模型跑完后，保存所有结果到单独的 JSON 文件
        # ==========================================
        scale_aggregates = {}
        
        # 1. 收集所有图片在各个 scale 的指标
        for item in dataset_metrics_data:
            for scale_idx, metrics in item["scales_data"].items():
                if scale_idx not in scale_aggregates:
                    scale_aggregates[scale_idx] = {
                        "bpp": [], "psnr": [], "msssim": [], "lpips": [], "dists": []
                    }
                scale_aggregates[scale_idx]["bpp"].append(metrics["bpp"])
                scale_aggregates[scale_idx]["psnr"].append(metrics["psnr"])
                scale_aggregates[scale_idx]["msssim"].append(metrics["msssim"])
                scale_aggregates[scale_idx]["lpips"].append(metrics["lpips"])
                scale_aggregates[scale_idx]["dists"].append(metrics["dists"])

        # 2. 计算平均值
        average_metrics_summary = {}
        print(f"\n[*] 模型 Iter: {current_iter} 测试集平均指标汇总:")
        for scale_idx, metric_lists in scale_aggregates.items():
            avg_bpp = float(np.mean(metric_lists["bpp"]))
            avg_psnr = float(np.mean(metric_lists["psnr"]))
            avg_msssim = float(np.mean(metric_lists["msssim"]))
            avg_lpips = float(np.mean(metric_lists["lpips"]))
            avg_dists = float(np.mean(metric_lists["dists"]))
            
            average_metrics_summary[scale_idx] = {
                "avg_bpp": avg_bpp,
                "avg_psnr": avg_psnr,
                "avg_msssim": avg_msssim,
                "avg_lpips": avg_lpips,
                "avg_dists": avg_dists,
                "image_count": len(metric_lists["bpp"])
            }
            
            # 在终端打印平均结果
            print(f"    Scale {scale_idx} | BPP: {avg_bpp:.4f} | PSNR: {avg_psnr:.4f} | "
                  f"MS-SSIM: {avg_msssim:.4f} | LPIPS: {avg_lpips:.4f} | DISTS: {avg_dists:.4f}")

        # 3. 构造最终要保存的字典结构
        final_json_output = {
            "model_iter": current_iter,
            "summary": average_metrics_summary,
            "details": dataset_metrics_data
        }

        # 4. 保存为 JSON 文件
        json_save_path = os.path.join(args.rec_path, f"metrics_iter_{current_iter}.json")
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(final_json_output, f, indent=4, ensure_ascii=False)
            
        # （可选）如果你更喜欢看表格形式的平均值，也可以顺手存一个 CSV
        df_summary = pd.DataFrame(average_metrics_summary).T
        df_summary.index.name = "scale"
        csv_save_path = os.path.join(args.rec_path, f"avg_metrics_iter_{current_iter}.csv")
        df_summary.to_csv(csv_save_path)
        
        print(f"[*] 当前模型详细指标及平均值已保存至: {json_save_path}")
        print(f"[*] 平均值 CSV 已单独保存至: {csv_save_path}\n")

    # ==========================================
    # 所有文件跑完退出测试后，进入 GPU Keep-Alive 逻辑
    # ==========================================
    print("\n[*] 所有可用模型已测试完毕。退出测试循环。")
    if getattr(args, 'keep_gpu_busy', 0) == 1:
        print("[*] 检测到 keep_gpu_busy=1，进入无限循环以保持 GPU 活跃...")
        if len(json_data) > 0:
            dummy_data = json_data[0]
            dummy_img_path, dummy_text = dummy_data['img_path'], dummy_data['txt']
            
            dummy_inp = load_img(dummy_img_path, args)
            _, _, dummy_h, dummy_w = dummy_inp.shape
            dummy_h_div_w = dummy_h / dummy_w
            dummy_h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - dummy_h_div_w))]
            dummy_scale_schedule = dynamic_resolution_h_w[dummy_h_div_w_template_][args.pn]['scales']
            dummy_scale_schedule = [(1, dummy_h, dummy_w) for (_, dummy_h, dummy_w) in dummy_scale_schedule]
            dummy_scale_q = [(dummy_scale_schedule[i][0], dummy_scale_schedule[i][1], dummy_scale_schedule[i][2], 
                        int((i+1)//((len(dummy_scale_schedule)//3)+1)+2)) for i in range(len(dummy_scale_schedule))]
            
            while True:
                try:
                    trans_list, help_list, _, gt_ms_idx_Bl = compress(
                        args, vae, dummy_scale_schedule, dummy_scale_q, infinity, 
                        text_tokenizer, text_encoder, dummy_img_path, dummy_text
                    )
                    dec_idx = decoding(args, infinity, vae, dummy_scale_schedule, dummy_text, text_tokenizer, text_encoder, gt_ms_idx_Bl, trans_list, help_list)
                    for gt_leak in range(0, len(gt_ms_idx_Bl)):
                        _ = decompress_cfg(infinity, vae, dummy_scale_schedule, dummy_text, text_tokenizer, text_encoder, gt_leak, dec_idx)
                except Exception as e:
                    time.sleep(1)
                    pass
