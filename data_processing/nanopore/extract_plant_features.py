import sys
import os
import numpy as np
import gzip
import argparse
import random
import multiprocessing as mp
import time

# ================= 核心配置 =================
SEQ_LEN = 13
HALF_LEN = SEQ_LEN // 2
DEFAULT_POINTS_PER_BASE = 16
ABNORMAL_SIGNAL_CAP = 2000
BATCH_SIZE = 50

# 全局变量
GLOBAL_LABELS = {}

# ================= 1. 核心算法 =================

def get_norm_params(all_signals):
    if not all_signals: return 0.0, 1.0
    data = np.array(all_signals, dtype=np.float32) 
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0: mad = 1.0
    scale = mad * 1.4826
    return median, scale

def fixed_len_sampling(vals, target_len=16):
    n = len(vals)
    if n == 0: return [0.0] * target_len
    if n > target_len: 
        return [vals[x] for x in sorted(random.sample(range(n), target_len))]
    elif n < target_len:
        pad_total = target_len - n
        pad_left = pad_total // 2
        return [0.0] * pad_left + vals + [0.0] * (pad_total - pad_left)
    else: return vals

def get_reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return "".join(complement.get(base, base) for base in reversed(seq))

# ================= 2. Worker 批处理任务 =================

def process_batch_task(batch_data):
    batch_results = []
    batch_unlabelled = []
    
    _int = int
    _float = float
    _mean = np.mean
    _std = np.std
    
    for read_id, raw_events in batch_data:
        try:
            sorted_events = sorted(raw_events, key=lambda x: _int(x[1]))
        except: continue
        if not sorted_events: continue

        try:
            start_ev_idx = _int(sorted_events[0][4])
            end_ev_idx = _int(sorted_events[-1][4])
            is_forward_read = (end_ev_idx >= start_ev_idx)
        except:
            is_forward_read = True 

        all_raw_signals = []
        pos_map = {} 
        parsed_events = []

        for i, ev in enumerate(sorted_events):
            try:
                p = _int(ev[1])
                if p not in pos_map: pos_map[p] = []
                pos_map[p].append(i) 
                
                s_str = ev[3]
                if s_str and s_str != "invalid":
                    vals = [_float(x) for x in s_str.split(',')]
                    all_raw_signals.extend(vals)
                else: vals = []
                
                parsed_events.append({'p': p, 'kmer': ev[2], 'vals': vals, 'contig': ev[0]})
            except ValueError: continue

        if not parsed_events: continue

        read_median, read_scale = get_norm_params(all_raw_signals)
        raw_chrom = parsed_events[0]['contig']
        chrom_keys = [raw_chrom, raw_chrom[3:] if raw_chrom.startswith('chr') else f"chr{raw_chrom}"]
        processed_sites = set()

        for idx, event in enumerate(parsed_events):
            kmer_start_pos = event['p']
            center_pos = kmer_start_pos
            if center_pos in processed_sites: continue
            center_base_ref = event['kmer'][0]
            target_strand = None
            
            if center_base_ref == 'C': 
                if not is_forward_read: continue 
                target_strand = '+'
            elif center_base_ref == 'G': 
                if is_forward_read: continue 
                target_strand = '-'
            else: continue 

            # 提取上下文以供基序推断
            full_context_found = True
            context_indices_lists = []
            for i in range(-HALF_LEN, HALF_LEN + 1):
                p = center_pos + i  
                if p in pos_map: context_indices_lists.append(pos_map[p]) 
                else:
                    full_context_found = False
                    break
            if not full_context_found: continue
            
            seq_list = []
            base_means, base_stds, base_lens, base_signals = [], [], [], []
            
            for c_indices in context_indices_lists:
                seq_list.append(parsed_events[c_indices[0]]['kmer'][0])
                combined_vals = []
                for c_idx in c_indices: combined_vals.extend(parsed_events[c_idx]['vals'])
                
                base_lens.append(str(len(combined_vals)))
                proc_vals = combined_vals[:ABNORMAL_SIGNAL_CAP]
                norm_scale = read_scale if read_scale != 0 else 1.0 
                norm_vals = [(v - read_median) / norm_scale for v in proc_vals]
                
                base_means.append(f"{_mean(norm_vals) if norm_vals else 0.0:.5f}")
                base_stds.append(f"{_std(norm_vals) if norm_vals else 0.0:.5f}")
                sampled_vals = fixed_len_sampling(norm_vals, DEFAULT_POINTS_PER_BASE)
                base_signals.append(",".join([f"{x:.3f}" for x in sampled_vals]))

            seq_str = "".join(seq_list)
            
            # --- 关键修改：检查标签并动态推断 Unlabelled 位点基序 ---
            found_label_info = None
            final_chrom_used = raw_chrom
            for c_key in chrom_keys:
                if (c_key, center_pos, target_strand) in GLOBAL_LABELS:
                    found_label_info = GLOBAL_LABELS[(c_key, center_pos, target_strand)]
                    final_chrom_used = c_key 
                    break
            
            is_unlabelled = False
            if found_label_info is None:
                is_unlabelled = True
                label_val = -1
                # 根据序列上下文推断是 CG, CHG 还是 CHH
                if target_strand == '+':
                    if HALF_LEN + 2 < len(seq_str):
                        if seq_str[HALF_LEN+1] == 'G': motif_type = 'CG'
                        elif seq_str[HALF_LEN+2] == 'G': motif_type = 'CHG'
                        else: motif_type = 'CHH'
                    else: motif_type = 'Unknown'
                else:
                    if HALF_LEN - 2 >= 0:
                        p1 = get_reverse_complement(seq_str[HALF_LEN-1])
                        p2 = get_reverse_complement(seq_str[HALF_LEN-2])
                        if p1 == 'G': motif_type = 'CG'
                        elif p2 == 'G': motif_type = 'CHG'
                        else: motif_type = 'CHH'
                    else: motif_type = 'Unknown'
            else:
                label_val, motif_type = found_label_info
            # --------------------------------------------------------

            if target_strand == '-':
                seq_str = get_reverse_complement(seq_str)
                base_means, base_stds, base_lens, base_signals = base_means[::-1], base_stds[::-1], base_lens[::-1], base_signals[::-1]

            row = [
                final_chrom_used, str(center_pos), target_strand, str(center_pos), read_id, 't',
                seq_str, ",".join(base_means), ",".join(base_stds), ",".join(base_lens), ";".join(base_signals),
                str(label_val), motif_type       
            ]
            
            # 分流保存
            if is_unlabelled:
                batch_unlabelled.append("\t".join(row))
            else:
                batch_results.append("\t".join(row))
            
            processed_sites.add(center_pos)
            
    return batch_results, batch_unlabelled

# ================= 3. 主进程逻辑 =================

def load_labels_multi(file_config):
    global GLOBAL_LABELS
    for item in file_config:
        f_path = item['path']
        lbl = item['label']
        mtf = item['motif']
        if not os.path.exists(f_path): continue
        opener = gzip.open(f_path, 'rt') if f_path.endswith('.gz') else open(f_path, 'r')
        with opener as f:
            for line in f:
                if not line.strip() or line.startswith('#'): continue
                parts = line.strip().split()
                try: 
                    chrom, pos = parts[0], int(parts[1])
                    strand = '+'
                    if len(parts) >= 3 and parts[2] in ['+', '-']: strand = parts[2]
                    elif len(parts) >= 6 and parts[5] in ['+', '-']: strand = parts[5]
                    GLOBAL_LABELS[(chrom, pos, strand)] = (lbl, mtf)
                except: continue

def write_results(res_tuple, f_out, f_unlbl):
    res_lbl, res_unlbl = res_tuple
    for l in res_lbl: f_out.write(l + '\n')
    for l in res_unlbl: f_unlbl.write(l + '\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cg_pos", required=True)
    parser.add_argument("--cg_neg", required=True)
    parser.add_argument("--chg_pos", required=True)
    parser.add_argument("--chg_neg", required=True)
    parser.add_argument("--chh_pos", required=True)
    parser.add_argument("--chh_neg", required=True)
    
    parser.add_argument("--out", required=True)
    parser.add_argument("--out_unlabelled", required=True) # 新增参数
    parser.add_argument("--p", type=int, default=10)
    args = parser.parse_args()
    
    load_config = [
        {'path': args.cg_pos,  'label': 1, 'motif': 'CG'},
        {'path': args.cg_neg,  'label': 0, 'motif': 'CG'},
        {'path': args.chg_pos, 'label': 1, 'motif': 'CHG'},
        {'path': args.chg_neg, 'label': 0, 'motif': 'CHG'},
        {'path': args.chh_pos, 'label': 1, 'motif': 'CHH'},
        {'path': args.chh_neg, 'label': 0, 'motif': 'CHH'},
    ]
    load_labels_multi(load_config)
    
    pool = mp.Pool(processes=args.p)
    
    with gzip.open(args.out, 'wt', newline='') as f_out, \
         gzip.open(args.out_unlabelled, 'wt', newline='') as f_unlbl:
        try:
            header_line = sys.stdin.readline()
            if not header_line: return
            header = header_line.strip().split('\t')
            col_map = {name: i for i, name in enumerate(header)}
            
            idx_read = col_map.get('read_name', 0)
            idx_contig = col_map.get('contig', 1)
            idx_pos = col_map.get('position', 2)
            idx_kmer = col_map.get('reference_kmer', 3)
            idx_samples = col_map.get('samples', 8)
            idx_event_index = col_map.get('event_index', 5) 
            if idx_samples is None: idx_samples = 8
        except Exception as e:
            sys.exit(1)

        curr_batch = []
        temp_read_id = None
        temp_events = []
        active_tasks =[]

        for line in sys.stdin:
            parts = line.split('\t')
            try:
                rid = parts[idx_read]
                event_data = (parts[idx_contig], parts[idx_pos], parts[idx_kmer], parts[idx_samples].strip(), parts[idx_event_index])
            except IndexError: continue

            if rid != temp_read_id:
                if temp_read_id is not None:
                    curr_batch.append((temp_read_id, temp_events))
                    if len(curr_batch) >= BATCH_SIZE:
                        task = pool.apply_async(process_batch_task, (curr_batch,), 
                                         callback=lambda res: write_results(res, f_out, f_unlbl))
                        active_tasks.append(task)
                        curr_batch = []
                        while len(active_tasks) >= args.p * 2:
                            active_tasks = [t for t in active_tasks if not t.ready()]
                            if len(active_tasks) >= args.p * 2:
                                time.sleep(0.05)
                temp_read_id = rid
                temp_events = []
            temp_events.append(event_data)
        
        if temp_read_id is not None:
            curr_batch.append((temp_read_id, temp_events))
        if curr_batch:
            pool.apply_async(process_batch_task, (curr_batch,), 
                             callback=lambda res: write_results(res, f_out, f_unlbl))

        pool.close()
        pool.join()

if __name__ == "__main__":
    main()