import random
import time
import gc
from functools import partial
from pprint import pformat
from typing import List, Optional, Tuple, Union
import os.path as osp

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as tdist
from torch.amp import autocast
import cv2

import infinity.utils.dist as dist
from infinity.models import Infinity
from infinity.models.ema import update_ema
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.models.tiny_entropy_student import teacher_logits_to_bit_logits_per_scale
from infinity.utils import arg_util, misc, swanlab_utils
from infinity.utils.amp_opt import AmpOptimizer
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

class InfinityTrainer(object):
    def __init__(
        self, is_visualizer: bool, device, raw_scale_schedule: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local, gpt_wo_ddp: Infinity, gpt: DDP, ema_ratio: float, max_it: int,
        gpt_opt: AmpOptimizer, label_smooth: float, z_loss_ratio: float, eq_loss: int, xen: bool,
        dbg_unused=False,zero=0, vae_type=True, reweight_loss_by_scale=False,
        gpt_wo_ddp_ema=None, gpt_ema=None, use_fsdp_model_ema=False, student_wo_ddp=None, student=None, student_opt=None, other_args=None,
    ):
        super(InfinityTrainer, self).__init__()
        self.dbg_unused = dbg_unused
        
        self.zero = zero
        self.vae_type = vae_type
        
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local, self.quantize_local = gpt, vae_local, vae_local.quantize
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[Infinity, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # after torch.compile
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.bitwise_self_correction = BitwiseSelfCorrection(self.vae_local, other_args)
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.seq_len_each = []
        self.reweight_loss_by_scale = reweight_loss_by_scale
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}')
        self.student_wo_ddp = student_wo_ddp
        self.student = student
        self.student_opt = student_opt
        self.enable_student_entropy = bool(getattr(other_args, 'enable_student_entropy', 0)) and (student_wo_ddp is not None)
        self.student_start_step = int(getattr(other_args, 'student_start_step', 0))
        self.student_start_scale = int(getattr(other_args, 'student_start_scale', 1))
        self.student_kd_ratio = float(getattr(other_args, 'student_kd_ratio', 1.0))
        self.student_gt_ratio = float(getattr(other_args, 'student_gt_ratio', 1.0))
        self.student_grad_clip = float(getattr(other_args, 'student_grad_clip', 1.0))
        self.student_codebook_dim = int(getattr(self.gpt_wo_ddp, 'V', self.vae_local.codebook_dim * 2) // 2) if self.enable_student_entropy else 0
        if self.enable_student_entropy:
            print(f'[student] start_step={self.student_start_step}, start_scale={self.student_start_scale}, kd={self.student_kd_ratio}, gt={self.student_gt_ratio}, grad_clip={self.student_grad_clip}')
        
        self.using_ema = ema_ratio != 0 and self.zero == 0
        self.ema_ratio = abs(ema_ratio)
        self.ema_cpu = ema_ratio < 0
        self.is_visualizer = is_visualizer
        
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled
        
        self.cached_state_not_ema = None
        if self.using_ema:
            self.pi_para_copy_for_parallel_ema = []
            all_tot = tot = 0
            for pi, para in enumerate(self.gpt_opt.paras):          # only learnable parameters need ema update
                if pi % dist.get_world_size() == dist.get_rank():   # model-parallel-style split
                    p_ema = para.data.cpu() if self.ema_cpu else para.data.clone()
                    self.pi_para_copy_for_parallel_ema.append((pi, p_ema))
                    tot += p_ema.numel()
                all_tot += para.numel()
            t = torch.zeros(dist.get_world_size())
            t[dist.get_rank()] = float(tot)
            dist.allreduce(t)
            t = [round(x) for x in t.tolist()]
            print(f'[ema tot #para] min={min(t)/1e6:.2f}, max={max(t)/1e6:.2f}, sum={sum(t)/1e6:.2f}, error={sum(t)-all_tot}')
            # lvl_1L, attn_bias_for_masking, zero_k_bias are never changed
            # check we only have these buffers so that we can skip buffer copy in ema update (only perform param update)
            assert all(any(s in name for s in ('lvl_1L', 'attn_bias_for_masking', 'zero_k_bias')) for name, _ in self.gpt_wo_ddp.named_buffers())
        else:
            self.pi_para_copy_for_parallel_ema = None
        
        self.label_smooth = label_smooth
        self.z_loss_ratio = z_loss_ratio
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.eq_loss = eq_loss
        
        if self.eq_loss:
            self.loss_eq_weight = torch.empty(1, self.raw_L, device=device)
            cur = 0
            for raw_pn in raw_scale_schedule:
                l = raw_pn*raw_pn
                self.loss_eq_weight[0, cur:cur+l] = 1./((raw_pn*raw_pn) if self.eq_loss == 2 else raw_pn)
                cur += l
            self.loss_eq_weight /= self.loss_eq_weight.sum()
        else:
            self.loss_eq_weight = 1.
        
        self.cmap_sim: ListedColormap = sns.color_palette('viridis', as_cmap=True)
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.generator = np.random.default_rng(0)
    
    def _build_student_targets(self, gt_ms_idx_Bl, scale_schedule, vae_scale_schedule, device):
        B = gt_ms_idx_Bl[0].shape[0]
        D = gt_ms_idx_Bl[0].shape[-1]
        prefix_maps = []
        gt_bits_bdhw = []
        cum_final = None
        for si, ((pt, ph, pw), idx_bl) in enumerate(zip(scale_schedule, gt_ms_idx_Bl)):
            if pt != 1:
                raise NotImplementedError('Tiny entropy student currently assumes image scales with pt=1.')
            bits = idx_bl.reshape(B, ph, pw, D).contiguous()
            gt_bits_bdhw.append(bits.permute(0, 3, 1, 2).contiguous().float())
            if si == 0:
                prefix_maps.append(torch.zeros((B, D, ph, pw), device=device, dtype=torch.float32))
            else:
                pm = F.interpolate(cum_final, size=vae_scale_schedule[si], mode=self.vae_local.quantizer.z_interplote_up).contiguous()
                prefix_maps.append(pm.squeeze(2).to(dtype=torch.float32))
            bits_btHWD = bits.unsqueeze(1)
            q = self.vae_local.quantizer.lfq.indices_to_codes(bits_btHWD, label_type='bit_label')
            q_up_final = F.interpolate(q, size=vae_scale_schedule[-1], mode=self.vae_local.quantizer.z_interplote_up).contiguous()
            cum_final = q_up_final if cum_final is None else (cum_final + q_up_final)
        return prefix_maps, gt_bits_bdhw

    @torch.no_grad()
    def eval_ep(self, ep: int, args: arg_util.Args, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.gpt_wo_ddp.training
        self.gpt_wo_ddp.eval()
        for inp, label_B in ld_val:
            B = label_B.shape[0]
            label_B = label_B.to(args.device, non_blocking=True)
            V = self.vae_local.vocab_size
            inp = inp.to(args.device, non_blocking=True)
            gt_ms_idx_Bl: List[Ten] = self.vae_local.get_GPT_ground_truth(inp)
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)
            self.gpt_wo_ddp.forward
            logits_BLV = self.gpt_wo_ddp(label_B, self.quantize_local.fuse_multiscale_idx_as_gpt_inp_BL(gt_ms_idx_Bl))
            
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.raw_last_l:].reshape(-1, V), gt_BL[:, -self.raw_last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.raw_last_l:].argmax(dim=-1) == gt_BL[:, -self.raw_last_l:]).sum() * (100/self.raw_last_l)
            tot += B
        self.gpt_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float, metric_lg: misc.MetricLogger, logging_params: bool,
        inp_B3HW: FTen, text_cond_tuple: Union[ITen, FTen], args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        
        B = inp_B3HW.shape[0]  # if isinstance(inp_B3HW, torch.Tensor) else inp_B3HW[0].shape[0]
        T = 1 if inp_B3HW.dim() == 4 else inp_B3HW.shape[2]
        V = self.vae_local.vocab_size
        device = inp_B3HW.device

        h_div_w = inp_B3HW.shape[-2] / inp_B3HW.shape[-1]
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        scale_schedule = [ (min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]
        
        # [forward]
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    if args.apply_spatial_patchify:
                        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                    else:
                        vae_scale_schedule = scale_schedule
                    raw_features, _, _ = self.vae_local.encode_for_raw_features(inp_B3HW, scale_schedule=vae_scale_schedule)
            
            nisp_enabled = bool(getattr(args, 'enable_nisp', 0))
            nisp_mode = str(getattr(args, 'nisp_mode', 'code')).lower()
            if nisp_enabled and nisp_mode == 'shallow':
                if int(getattr(args, 'nisp_target_layer', 0)) <= 0:
                    raise RuntimeError('shallow NITP requires nisp_target_layer > 0.')
                if int(getattr(args, 'nisp_scale_min', 0)) < 1:
                    raise RuntimeError('shallow NITP should not supervise scale 0 because that position uses sos as input.')
            if nisp_enabled:
                if nisp_mode == 'code':
                    x_BLC_wo_prefix, gt_ms_idx_Bl, nisp_target_ms_BlC = self.bitwise_self_correction.flip_requant(
                        vae_scale_schedule, inp_B3HW, raw_features, device, return_target_codes=True)
                    shallow_nitp_ms_BlC = None
                elif nisp_mode == 'shallow':
                    x_BLC_wo_prefix, gt_ms_idx_Bl, shallow_nitp_ms_BlC = self.bitwise_self_correction.flip_requant(
                        vae_scale_schedule, inp_B3HW, raw_features, device, return_shallow_nitp_inputs=True)
                    nisp_target_ms_BlC = None
                else:
                    raise NotImplementedError(f'{nisp_mode=}')
            else:
                x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(
                    vae_scale_schedule, inp_B3HW, raw_features, device)
                nisp_target_ms_BlC = None
                shallow_nitp_ms_BlC = None
            # x_BLC_wo_prefix: torch.Size([bs, 2*2+3*3+...+64*64, d or 4d])

            # truncate scales
            training_scales = min(int(args.always_training_scales), len(scale_schedule))
            training_seq_len = np.array(scale_schedule)[:training_scales].prod(axis=1).sum()
            x_BLC_wo_prefix = x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :]
            shallow_nitp_x_BLC_wo_prefix = None
            if nisp_enabled and nisp_mode == 'shallow':
                if training_scales <= 1:
                    raise RuntimeError('shallow NITP requires at least two training scales.')
                shallow_nitp_x_BLC_wo_prefix = torch.cat(shallow_nitp_ms_BlC[1:training_scales], dim=1).contiguous()
                shallow_nitp_x_BLC_wo_prefix = shallow_nitp_x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :]
            
            self.gpt_wo_ddp.forward  
            if nisp_enabled and nisp_mode == 'code':
                logits_BLV, nisp_aux = self.gpt(
                    text_cond_tuple, x_BLC_wo_prefix,
                    scale_schedule=scale_schedule[:training_scales],
                    return_nisp_pred=True,
                )
            elif nisp_enabled and nisp_mode == 'shallow':
                logits_BLV, nisp_aux = self.gpt(
                    text_cond_tuple, x_BLC_wo_prefix,
                    scale_schedule=scale_schedule[:training_scales],
                    return_shallow_nitp=True,
                    shallow_nitp_x_BLC_wo_prefix=shallow_nitp_x_BLC_wo_prefix,
                    shallow_nitp_target_layer=int(getattr(args, 'nisp_target_layer', 0)),
                )
            else:
                logits_BLV = self.gpt(text_cond_tuple, x_BLC_wo_prefix, scale_schedule=scale_schedule[:training_scales]) # [bs, 1*1+...+64*64, vocab_size or log2(vocab_size)*2]
                nisp_aux = None
            self.batch_size, self.seq_len = logits_BLV.shape[:2]

            self.seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]
            if args.use_bit_label:
                tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
                loss = self.train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
                if args.bitloss_type == 'mean':
                    loss = loss.mean(dim=-1)
                elif args.bitloss_type == 'sum':
                    loss = loss.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{args.bitloss_type=}')
            else:
                loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)

            if self.reweight_loss_by_scale:
                lw = []
                last_scale_area = np.sqrt(int(np.prod(scale_schedule[-1])))
                for (pt, ph, pw) in scale_schedule[:training_scales]:
                    this_scale_area = np.sqrt(pt * ph * pw)
                    lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
                lw = torch.tensor(lw, device=loss.device)[None, ...]
                lw = lw / lw.sum()
            else:
                lw = 1. / self.seq_len
            ce_loss = loss.mul(lw).sum(dim=-1).mean()

            nisp_loss = ce_loss.new_tensor(0.0)
            if nisp_enabled:
                if nisp_mode == 'code':
                    nisp_pred_BLC = nisp_aux['nisp_pred']
                    nisp_target_BLC = torch.cat(nisp_target_ms_BlC[:training_scales], dim=1)[:, :training_seq_len].contiguous()
                    nisp_target_BLC = nisp_target_BLC.to(device=nisp_pred_BLC.device, dtype=nisp_pred_BLC.dtype).detach()
                elif nisp_mode == 'shallow':
                    nisp_pred_BLC = nisp_aux['shallow_nitp_pred']
                    nisp_target_BLC = nisp_aux['shallow_nitp_target'].to(device=nisp_pred_BLC.device, dtype=nisp_pred_BLC.dtype).detach()
                else:
                    raise NotImplementedError(f'{nisp_mode=}')
                if nisp_pred_BLC.shape != nisp_target_BLC.shape:
                    raise RuntimeError(f'NISP pred/target shape mismatch: pred={nisp_pred_BLC.shape}, target={nisp_target_BLC.shape}')

                nisp_mask = []
                nisp_scale_min = int(getattr(args, 'nisp_scale_min', 0))
                nisp_scale_max = int(getattr(args, 'nisp_scale_max', training_scales - 1))
                for scale_ind, (pt, ph, pw) in enumerate(scale_schedule[:training_scales]):
                    active = nisp_scale_min <= scale_ind <= nisp_scale_max
                    nisp_mask.extend([1.0 if active else 0.0 for _ in range(pt * ph * pw)])
                nisp_mask = torch.tensor(nisp_mask, device=nisp_pred_BLC.device, dtype=nisp_pred_BLC.dtype)[None, :self.seq_len]
                if torch.is_tensor(lw):
                    nisp_lw = lw.to(device=nisp_pred_BLC.device, dtype=nisp_pred_BLC.dtype)
                else:
                    nisp_lw = torch.full((1, self.seq_len), float(lw), device=nisp_pred_BLC.device, dtype=nisp_pred_BLC.dtype)
                nisp_weights = nisp_lw * nisp_mask
                nisp_denom = nisp_weights.sum().clamp_min(1e-12)

                nisp_loss_type = getattr(args, 'nisp_loss_type', 'cosine')
                if nisp_loss_type == 'cosine':
                    pred_norm = F.normalize(nisp_pred_BLC.float(), dim=-1)
                    target_norm = F.normalize(nisp_target_BLC.float(), dim=-1)
                    nisp_token_loss = 1.0 - (pred_norm * target_norm).sum(dim=-1)
                elif nisp_loss_type == 'mse':
                    nisp_token_loss = F.mse_loss(nisp_pred_BLC.float(), nisp_target_BLC.float(), reduction='none').mean(dim=-1)
                else:
                    raise NotImplementedError(f'{nisp_loss_type=}')
                nisp_loss = (nisp_token_loss * nisp_weights.float()).sum(dim=-1).div(nisp_denom.float()).mean()

            loss = ce_loss + float(getattr(args, 'nisp_weight', 0.0)) * nisp_loss

        student_loss = None
        student_grad_norm = loss.new_tensor(0.0)
        student_is_active = self.enable_student_entropy and (g_it >= self.student_start_step)
        if student_is_active:
            teacher_bit_logits_per_scale = teacher_logits_to_bit_logits_per_scale(logits_BLV.detach(), scale_schedule[:training_scales], self.student_codebook_dim)
            prefix_maps, gt_bits_bdhw = self._build_student_targets(gt_ms_idx_Bl[:training_scales], scale_schedule[:training_scales], vae_scale_schedule[:training_scales], device)
            student_terms = []
            student_model = self.student if self.student is not None else self.student_wo_ddp
            for si in range(max(0, self.student_start_scale), training_scales):
                s_logits = student_model(prefix_maps[si].detach(), text_cond_tuple, scale_id=si)
                t_probs = torch.sigmoid(teacher_bit_logits_per_scale[si].detach())
                hard_bits = gt_bits_bdhw[si].to(dtype=s_logits.dtype)
                cur = 0.0
                if self.student_kd_ratio > 0:
                    cur = cur + self.student_kd_ratio * F.binary_cross_entropy_with_logits(s_logits, t_probs)
                if self.student_gt_ratio > 0:
                    cur = cur + self.student_gt_ratio * F.binary_cross_entropy_with_logits(s_logits, hard_bits)
                student_terms.append(cur)
            if len(student_terms):
                student_loss = torch.stack(student_terms).mean()
        
        # [backward]
        if student_is_active and student_loss is not None:
            (student_loss * self.gpt_opt.r_accu).backward(retain_graph=False, create_graph=False)
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=ep, it=it, g_it=g_it, stepping=stepping, logging_params=logging_params, loss=loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable)
        if stepping and student_is_active and self.student_opt is not None:
            if self.student_grad_clip > 0:
                if isinstance(self.student, FSDP):
                    student_grad_norm = self.student.clip_grad_norm_(self.student_grad_clip)
                else:
                    student_grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters() if self.student is not None else self.student_wo_ddp.parameters(), self.student_grad_clip)
            self.student_opt.step()
        
        # update ema
        if args.use_fsdp_model_ema:
            update_ema(self.gpt_ema, self.gpt)

        # [zero_grad]
        if stepping:
            if self.using_ema: self.ema_update(g_it)
            if self.dbg_unused:
                ls = []
                for n, p in self.gpt_wo_ddp.named_parameters():
                    if p.grad is None:
                        ls.append(n)
                if len(ls):
                    raise AttributeError(f'unused param: {ls}')
        
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
            if self.enable_student_entropy and self.student_opt is not None:
                self.student_opt.zero_grad(set_to_none=True)
        
        # [metric logging]
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
            B, seq_len = logits_BLV.shape[:2]
            if args.use_bit_label:
                res_loss = self.train_loss(logits_BLV.reshape(B, seq_len, -1, 2).permute(0,3,1,2), gt_BL).mean(dim=-1).mean(0)
                bitwise_acc = (logits_BLV.reshape(B, seq_len, -1, 2).argmax(dim=-1) == gt_BL).float() # shape: [bs, seq_len, codebook_dim]
            else:
                res_loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1).mean(0)
                pred_BL = logits_BLV.argmax(dim=-1)
                mask = self.vae_local.quantizer.lfq.mask
                pred_bits = ((pred_BL[..., None].int() & mask) != 0)
                gt_bits = ((gt_BL[..., None].int() & mask) != 0)
                bitwise_acc = (pred_bits == gt_bits).float() # shape: [bs, seq_len, codebook_dim]
            res_bit_acc = bitwise_acc.mean(-1).mean(0)
            res_token_acc = (bitwise_acc.sum(-1) == self.vae_local.codebook_dim).float().mean(0)
            
            loss_token_mean, acc_bit_mean, acc_token_mean = res_loss.mean().item(), res_bit_acc.mean().item() * 100., res_token_acc.mean().item() * 100.
            ptr = 0
            L_list, acc_bit_list, acc_token_list = [], [], []
            for scale_ind in range(min(training_scales, len(scale_schedule))):
                start, end = ptr, ptr + np.array(scale_schedule[scale_ind]).prod()
                L_list.append(res_loss[start:end].mean().item())
                acc_bit_list.append(res_bit_acc[start:end].mean().item() * 100.)
                acc_token_list.append(res_token_acc[start:end].mean().item() * 100.)
                ptr = end
            
            nisp_loss_item = float(nisp_loss.detach().item()) if torch.is_tensor(nisp_loss) else float(nisp_loss)
            nisp_weighted_item = nisp_loss_item * float(getattr(args, 'nisp_weight', 0.0))
            metrics = torch.tensor(L_list + acc_bit_list + acc_token_list +[
                grad_norm_t.item(), loss_token_mean, acc_bit_mean, acc_token_mean,
                nisp_loss_item, nisp_weighted_item,
            ], device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            metrics = metrics.cpu().data.numpy() / dist.get_world_size()
            leng = len(L_list)
            L_list, acc_bit_list, acc_token_list, grad_norm_t, loss_token_mean, acc_bit_mean, acc_token_mean = metrics[:leng], \
                metrics[leng:2*leng], metrics[2*leng:3*leng], metrics[-6], metrics[-5], metrics[-4], metrics[-3]
            nisp_loss_mean, nisp_weighted_mean = metrics[-2], metrics[-1]
            Lmean = loss_token_mean
            Ltail = L_list[-1]
            acc_mean = acc_bit_mean if args.use_bit_label else acc_token_mean
            acc_tail = acc_bit_list[-1] if args.use_bit_label else acc_token_list[-1]
            student_loss_mean = float(student_loss.item()) if student_loss is not None else 0.0
            student_grad_norm_f = float(student_grad_norm.item()) if torch.is_tensor(student_grad_norm) else float(student_grad_norm)
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_t, Stu=student_loss_mean, StuGD=student_grad_norm_f, NISP=nisp_loss_mean, NISPw=nisp_weighted_mean)    # todo: Accm, Acct
            swanlab_log_dict = {
                "Overall/L_mean": Lmean, 'Overall/Acc_bit_mean': acc_bit_mean, 'Overall/Acc_token_mean': acc_token_mean, 'Overall/grad_norm_t': grad_norm_t,
                'Student/loss': student_loss_mean, 'Student/grad_norm': student_grad_norm_f, 'Student/active': float(student_is_active),
                'NISP/loss': nisp_loss_mean, 'NISP/weighted_loss': nisp_weighted_mean, 'NISP/active': float(nisp_enabled),
                'NISP/scale_min': float(getattr(args, 'nisp_scale_min', 0)), 'NISP/scale_max': float(getattr(args, 'nisp_scale_max', 0)),
                'NISP/mode_is_shallow': float(str(getattr(args, 'nisp_mode', 'code')).lower() == 'shallow'),
                'NISP/target_layer': float(getattr(args, 'nisp_target_layer', 0)),
            }
            for si, (loss_si, acc_bit_si, acc_token_si) in enumerate(zip(L_list, acc_bit_list, acc_token_list)):
                swanlab_log_dict[f'Detail/L_s{si+1:02d}'] = loss_si
                swanlab_log_dict[f'Detail/Acc_bit_s{si+1:02d}'] = acc_bit_si
                swanlab_log_dict[f'Detail/Acc_token_s{si+1:02d}'] = acc_token_si
            swanlab_utils.log(swanlab_log_dict, step=g_it)
        
        return grad_norm_t, scale_log2_t
    
    def __repr__(self):
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(InfinityTrainer, self).__repr__().replace(InfinityTrainer.__name__, "")}'
        )
    
    def ema_load(self):
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())
    
    def ema_recover(self):
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None
    
    # p_ema = p_ema*0.9 + p*0.1 <==> p_ema.lerp_(p, 0.1)
    # p_ema.mul_(self.ema_ratio).add_(p.mul(self.ema_ratio_1))
    # @profile(precision=4, stream=open('ema_update.log', 'w+'))
    def ema_update(self, g_it): # todo: 将来再用离线ema
        # if self.using_ema and (g_it + 1) in self.ema_upd_it:
        stt = time.time()
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            p = self.gpt_opt.paras[pi]
            p_ema.data.mul_(self.ema_ratio).add_(p.data.to(p_ema.device), alpha=1-self.ema_ratio)
        # ii = self.ema_upd_it.index(g_it + 1)
        ii = g_it
        if ii < 3:
            print(f'[ema upd {self.ema_ratio}, cpu={self.ema_cpu}, @ g_it={g_it}] cost: {time.time()-stt:.2f}s')
    
    def get_config(self):
        return {
            'dynamic_resolution_h_w': dynamic_resolution_h_w,
            'label_smooth': self.label_smooth, 'eq_loss': self.eq_loss,
            'ema_ratio':    self.ema_ratio,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}
        
        if self.zero:   # TODO: fixme
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                state['gpt_fsdp'] = self.gpt.state_dict()
                if self.use_fsdp_model_ema:
                    state['gpt_ema_fsdp'] = self.gpt_ema.state_dict()
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=self.gpt_opt.optimizer.state_dict())
            if self.student is not None:
                with FSDP.state_dict_type(self.student, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                    state['student_fsdp'] = self.student.state_dict()
                    if self.student_opt is not None:
                        state['student_fsdp_opt'] = FSDP.optim_state_dict(model=self.student, optim=self.student_opt, optim_state_dict=self.student_opt.state_dict())
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()
        
        else:
            if self.using_ema:  # TODO: fixme
                self.ema_load()
                state['gpt_ema_for_vis'] = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
                self.ema_recover()
            
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    state[k] = m.state_dict()
            if self.student_wo_ddp is not None:
                m = self.student_wo_ddp._orig_mod if hasattr(self.student_wo_ddp, '_orig_mod') else self.student_wo_ddp
                state['student_wo_ddp'] = m.state_dict()
            if self.student_opt is not None:
                state['student_opt'] = self.student_opt.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        if self.zero:
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                self.gpt.load_state_dict(state['gpt_fsdp'])
                if self.use_fsdp_model_ema:
                    self.gpt_ema.load_state_dict(state['gpt_ema_fsdp'])
                one_group_opt_state = state['gpt_fsdp_opt']
                """
                AdamW state['gpt_fsdp_opt']:
                {
                    'state': { <para_name>: {'exp_avg': <unsharded_tensor>, 'exp_avg_sq': <unsharded_tensor>, 'step': <int>} },
                    'param_groups': [
                        {
                            'wd_sc': 1.0, 'lr_sc': 1.0, 'lr': xxx, 'betas': (0.9, 0.97), 'eps': 1e-08, 'weight_decay': 0.02,
                            'amsgrad': False, 'foreach': None, 'maximize': False, 'capturable': False, 'differentiable': False, 'fused': True,
                            'params': [<para_name> x m]
                        } x n
                    ]
                }
                one_group_opt_state['param_groups'] = self.gpt_opt.optimizer.state_dict()['param_groups']
                """
                optim_state_dict = FSDP.optim_state_dict_to_load(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=one_group_opt_state)
                self.gpt_opt.optimizer.load_state_dict(optim_state_dict)

            if self.student is not None and 'student_fsdp' in state:
                with FSDP.state_dict_type(self.student, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                    self.student.load_state_dict(state['student_fsdp'])
                    if self.student_opt is not None and 'student_fsdp_opt' in state:
                        student_opt_state = FSDP.optim_state_dict_to_load(model=self.student, optim=self.student_opt, optim_state_dict=state['student_fsdp_opt'])
                        self.student_opt.load_state_dict(student_opt_state)

            if self.gpt_opt.scaler is not None:
                try: self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        else:
            for k in ('gpt_wo_ddp', 'gpt_opt'):
                if skip_vae and 'vae' in k: continue
                m = getattr(self, k)
                if m is not None:
                    if hasattr(m, '_orig_mod'):
                        m = m._orig_mod
                    ret = m.load_state_dict(state[k], strict=strict)
                    if ret is not None:
                        missing, unexpected = ret
                        print(f'[VGPTTr.load_state_dict] {k} missing:  {missing}')
                        print(f'[VGPTTr.load_state_dict] {k} unexpected:  {unexpected}')
            if self.student_wo_ddp is not None and 'student_wo_ddp' in state:
                ret = self.student_wo_ddp.load_state_dict(state['student_wo_ddp'], strict=False)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VGPTTr.load_state_dict] student_wo_ddp missing:  {missing}')
                    print(f'[VGPTTr.load_state_dict] student_wo_ddp unexpected:  {unexpected}')
            if self.student_opt is not None and 'student_opt' in state:
                self.student_opt.load_state_dict(state['student_opt'])
            
            if self.using_ema:
                if 'gpt_ema_for_vis' in state:
                    for pi, para in self.pi_para_copy_for_parallel_ema:
                        para.copy_(state['gpt_ema_for_vis'][self.gpt_opt.names[pi]])
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: load succeed')
                else:
                    print(f'[VGPTTr.load_state_dict] gpt_ema_for_vis: key NOT FOUND in state!!')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VGPT.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
