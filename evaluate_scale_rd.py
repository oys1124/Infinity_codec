import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import sys
import csv
import json
import argparse
import tqdm
import torch
import pyiqa
import re

from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from neuralcompression.metrics import update_patch_fid

IMG_GLOB = "*.[jpJP][pnPN]*[gG]"  # jpg/jpeg/png (case-insensitive)
NO_REF_METRICS = ["clipiqa", "musiq", "niqe", "maniqa"]

DEFAULT_PALETTE = [
    "#6baed6", "#9ecae1", "#c6dbef", "#3182bd", "#08519c",
    "#74c476", "#31a354", "#006d2c", "#fd8d3c", "#e6550d",
]


def _list_images(folder: Path):
    return sorted([x for x in folder.glob(IMG_GLOB)])



def _build_stem_map(paths):
    m = {}
    for p in paths:
        m[p.stem] = p
    return m



def _create_no_ref_metrics(device):
    metrics = {}
    for name in NO_REF_METRICS:
        try:
            metrics[name] = pyiqa.create_metric(name).to(device)
        except Exception as e:
            print(f"[Warn] failed to create no-ref metric '{name}': {e}")
    return metrics


@torch.no_grad()
def evaluate_one(recon_dir: Path, gt_dir: Path, ntest=None, device="cuda", fid_min_pairs=51):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    totensor = ToTensor()

    recon_dir = Path(recon_dir)
    gt_dir = Path(gt_dir)

    if (not recon_dir.is_dir()) or (not gt_dir.is_dir()):
        raise FileNotFoundError(f"Missing dir: recon={recon_dir} gt={gt_dir}")

    recon_paths = _list_images(recon_dir)
    gt_paths = _list_images(gt_dir)

    if len(recon_paths) == 0:
        raise RuntimeError(f"No images in recon_dir: {recon_dir}")
    if len(gt_paths) == 0:
        raise RuntimeError(f"No images in gt_dir: {gt_dir}")

    recon_map = _build_stem_map(recon_paths)
    gt_map = _build_stem_map(gt_paths)

    common = sorted(set(recon_map.keys()) & set(gt_map.keys()))
    if ntest is not None:
        common = common[:ntest]
    if len(common) == 0:
        raise RuntimeError(
            f"No matched pairs between {recon_dir} and {gt_dir}. "
            f"(Matched by filename stem.)"
        )

    metric_single = _create_no_ref_metrics(device)
    metric_paired = {
        "psnr": pyiqa.create_metric("psnr").to(device),
        "dists": pyiqa.create_metric("dists").to(device),
        "ms_ssim": pyiqa.create_metric("ms_ssim").to(device),
        "lpips": LearnedPerceptualImagePatchSimilarity(normalize=True).to(device),
    }

    fid_metric = FrechetInceptionDistance().to(device)
    kid_metric = KernelInceptionDistance().to(device)

    result_sum = {}
    n = len(common)

    for stem in tqdm.tqdm(common, desc=f"Eval {recon_dir.parent.name}/{recon_dir.name}", dynamic_ncols=True):
        recon_path = recon_map[stem]
        gt_path = gt_map[stem]

        with open(recon_path, "rb") as f:
            image_recon = Image.open(f).convert("RGB")
        with open(gt_path, "rb") as f:
            image_gt = Image.open(f).convert("RGB")

        recon_tensor = totensor(image_recon).unsqueeze(0).to(device)
        gt_tensor = totensor(image_gt).unsqueeze(0).to(device)

        for k, metric in metric_single.items():
            v = float(metric(recon_tensor).item())
            result_sum[k] = result_sum.get(k, 0.0) + v

        update_patch_fid(gt_tensor, recon_tensor, fid_metric=fid_metric, kid_metric=kid_metric)

        for k, metric in metric_paired.items():
            v = float(metric(recon_tensor, gt_tensor).item())
            result_sum[k] = result_sum.get(k, 0.0) + v

    out = {k: (v / n) for k, v in result_sum.items()}

    if n >= fid_min_pairs:
        out["fid"] = float(fid_metric.compute())
        kid_mean, kid_std = kid_metric.compute()
        out["kid_mean"] = float(kid_mean)
        out["kid_std"] = float(kid_std)

    torch.cuda.empty_cache()
    return out



def _discover_scale_folders(root: Path, scale_names=None):
    if scale_names is not None and len(scale_names) > 0:
        scales = [root / s for s in scale_names]
    else:
        def get_scale_idx(p):
            try:
                return int(p.name.split('_')[1])
            except (IndexError, ValueError):
                return -1
        scales = sorted(
            [p for p in root.glob("scale_*") if p.is_dir()],
            key=get_scale_idx
        )
    if len(scales) == 0:
        raise RuntimeError(f"No scale folders found under: {root}")
    return scales



def _parse_csvs(root: Path):
    bpp_dict = {}
    metrics_dict = {}
    avg_csvs = list(root.glob("avg_metrics_iter_*.csv"))
    bpp_csv = root / "overall_bpp_stats.csv"

    if avg_csvs:
        with open(avg_csvs[0], 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scale = int(row['scale'])
                bpp_dict[scale] = float(row['avg_bpp'])
                metrics_dict[scale] = {
                    'psnr': float(row['avg_psnr']),
                    'ms_ssim': float(row['avg_msssim']),
                    'lpips': float(row['avg_lpips']),
                    'dists': float(row['avg_dists'])
                }
    elif bpp_csv.exists():
        with open(bpp_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                scale = int(row['scale'])
                bpp_dict[scale] = float(row['avg_bpp'])

    return bpp_dict, metrics_dict



def _get_better_metric(eval_val, csv_val, metric_name):
    if eval_val is None and csv_val is None:
        return None
    if eval_val is None:
        return csv_val
    if csv_val is None:
        return eval_val
    if metric_name in ["psnr", "ms_ssim", "clipiqa", "musiq", "maniqa"]:
        return max(eval_val, csv_val)
    elif metric_name in ["lpips", "dists", "niqe"]:
        return min(eval_val, csv_val)
    return eval_val



def insert_into_plot_script(plot_script_path, root_dir_name, entry_dict):
    """联动核心逻辑：分配独占颜色，清洗数据并注入绘图脚本"""
    plot_script_path = Path(plot_script_path)
    if not plot_script_path.exists():
        print(f"\n[Warn] Plot script '{plot_script_path}' not found. Skipping auto-insertion.")
        return

    suffix = root_dir_name.split("_")[-1] if "_" in root_dir_name else root_dir_name
    method_name = f"VAR_nisp_code_mlp256_scale0-12_w0.03_{suffix}"

    with open(plot_script_path, "r", encoding="utf-8") as f:
        content = f.read()

    if f"'{method_name}':" in content or f'"{method_name}":' in content:
        print(f"\n[Info] Method '{method_name}' already exists in {plot_script_path}. Skipping to avoid duplicates.")
        return

    distinct_palette = [
        "#e6194B", "#3cb44b", "#f58231", "#911eb4", "#42d4f4",
        "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
        "#9A6324", "#800000", "#aaffc3", "#808000", "#ffd8b1",
        "#000075", "#a9a9a9", "#ff0000", "#00ff00", "#0000ff"
    ]
    assigned_color = "#000000"
    for c in distinct_palette:
        if c.lower() not in content.lower():
            assigned_color = c
            break

    valid_indices = [i for i, b in enumerate(entry_dict['bpp']) if b is not None]

    final_dict = {
        'bpp': [entry_dict['bpp'][i] for i in valid_indices],
        'color': assigned_color,
        'linestyle': "-",
        'marker': "o",
        'smooth': False
    }

    metrics_mapping = {
        'lpips': 'lpips',
        'dists': 'dists',
        'fid': 'fid',
        'psnr': 'psnr',
        'ms_ssim': 'ms-ssim',
        'clipiqa': 'clipiqa',
        'musiq': 'musiq',
        'niqe': 'niqe',
        'maniqa': 'maniqa',
    }
    for eval_k, plot_k in metrics_mapping.items():
        if eval_k in entry_dict:
            m_list = [entry_dict[eval_k][i] for i in valid_indices]
            if None not in m_list and len(m_list) > 0:
                final_dict[plot_k] = m_list

    dict_str = json.dumps(final_dict, indent=4).replace("false", "False").replace("true", "True")
    insert_text = f"\n    '{method_name}': {dict_str},\n"

    match = re.search(r"DATA\s*=\s*\{", content)
    if match:
        insert_pos = match.end()
        new_content = content[:insert_pos] + insert_text + content[insert_pos:]
        with open(plot_script_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"\n[Success] Injected '{method_name}' into {plot_script_path} with color {assigned_color}")
    else:
        print(f"\n[Error] Could not find 'DATA = {{' in {plot_script_path}.")



def parse_args(argv):
    p = argparse.ArgumentParser("Evaluate and update RD plot script automatically.")
    p.add_argument("--root", type=str, default="/workspace/Infinity_codec/results/stage2_1024_layer12c4_16vae_nisp_code_L3_scale0-12_mlp256_w0.03/DIV2K_ori_10000",
                   help="Root folder containing scale_X directories.")
    p.add_argument("--gt_dir", type=str, default="/workspace/Infinity_codec/data/DIV2K_1024",
                   help="Ground-truth folder")
    p.add_argument("--scales", nargs="*", default=[],
                   help="Optional: specify scale folders explicitly, e.g. scale_0 scale_1 ...")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--ntest", type=int, default=None)
    p.add_argument("--fid_min_pairs", type=int, default=51)
    p.add_argument("--plot_script", type=str, default="/workspace/Infinity_codec/data/DIV2K_metric_full/div2k_1024_inline copy 3.py",
                   help="Path to your RD curve plotting script (e.g. plot_rd.py)")
    return p.parse_args(argv)



def main(argv):
    args = parse_args(argv)
    root = Path(args.root)

    scale_dirs = _discover_scale_folders(root, args.scales)
    bpp_dict, metrics_dict = _parse_csvs(root)

    entry = {
        "bpp": [],
        "lpips": [],
        "dists": [],
        "fid": [],
        "psnr": [],
        "ms_ssim": [],
        "clipiqa": [],
        "musiq": [],
        "niqe": [],
        "maniqa": [],
    }

    print(f"\n==================== Start Evaluating ====================")
    for scale_dir in scale_dirs:
        try:
            scale_idx = int(scale_dir.name.split('_')[1])
        except (IndexError, ValueError):
            scale_idx = -1

        csv_bpp = bpp_dict.get(scale_idx, None)
        csv_m = metrics_dict.get(scale_idx, {})

        if not scale_dir.is_dir():
            entry["bpp"].append(csv_bpp)
            entry["lpips"].append(csv_m.get("lpips", None))
            entry["dists"].append(csv_m.get("dists", None))
            entry["fid"].append(None)
            entry["psnr"].append(csv_m.get("psnr", None))
            entry["ms_ssim"].append(csv_m.get("ms_ssim", None))
            entry["clipiqa"].append(csv_m.get("clipiqa", None))
            entry["musiq"].append(csv_m.get("musiq", None))
            entry["niqe"].append(csv_m.get("niqe", None))
            entry["maniqa"].append(csv_m.get("maniqa", None))
            continue

        res = {}
        try:
            res = evaluate_one(
                recon_dir=scale_dir, gt_dir=Path(args.gt_dir), ntest=args.ntest,
                device=args.device, fid_min_pairs=args.fid_min_pairs,
            )
        except Exception as e:
            print(f"[Error] eval failed for {scale_dir}: {e}")

        best_psnr = _get_better_metric(res.get("psnr"), csv_m.get("psnr"), "psnr")
        best_msssim = _get_better_metric(res.get("ms_ssim"), csv_m.get("ms_ssim"), "ms_ssim")
        best_lpips = _get_better_metric(res.get("lpips"), csv_m.get("lpips"), "lpips")
        best_dists = _get_better_metric(res.get("dists"), csv_m.get("dists"), "dists")
        eval_fid = res.get("fid", None)

        entry["bpp"].append(csv_bpp)
        entry["lpips"].append(best_lpips)
        entry["dists"].append(best_dists)
        entry["fid"].append(eval_fid)
        entry["psnr"].append(best_psnr)
        entry["ms_ssim"].append(best_msssim)
        entry["clipiqa"].append(res.get("clipiqa", None))
        entry["musiq"].append(res.get("musiq", None))
        entry["niqe"].append(res.get("niqe", None))
        entry["maniqa"].append(res.get("maniqa", None))

        print(
            f"[{scale_dir.name}] bpp={csv_bpp:.6f} | "
            f"lpips={best_lpips:.5f} | dists={best_dists:.5f} | fid={eval_fid} | "
            f"psnr={best_psnr:.4f} | ms_ssim={best_msssim:.4f} | "
            f"clipiqa={res.get('clipiqa', None)} | musiq={res.get('musiq', None)} | "
            f"niqe={res.get('niqe', None)} | maniqa={res.get('maniqa', None)}"
        )

    insert_into_plot_script(args.plot_script, root.name, entry)


if __name__ == "__main__":
    main(sys.argv[1:])
