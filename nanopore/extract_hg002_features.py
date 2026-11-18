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
# Key: (chrom, pos, strand) -> Value: (label_int, motif_str)
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
    
    # 【修复2：按时间索引抽样并排序，绝对保留真实波形起伏】
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
    
    _int = int
    _float = float
    _mean = np.mean
    _std = np.std
    
    for read_id, raw_events in batch_data:
        try:
            sorted_events = sorted(raw_events, key=lambda x: _int(x[1]))
        except: continue
        if not sorted_events: continue

        # 【修复3：通过首尾事件的时间轴 (event_index) 差值推断 Read 物理方向】
        try:
            start_ev_idx = _int(sorted_events[0][4])
            end_ev_idx = _int(sorted_events[-1][4])
            is_forward_read = (end_ev_idx >= start_ev_idx)
        except:
            is_forward_read = True # 备用容错

        # 预处理 Event
        all_raw_signals = []
        pos_map = {} 
        parsed_events = []

        for i, ev in enumerate(sorted_events):
            try:
                p = _int(ev[1])
                
                # 【修复1：改用列表追加，完整保留该位点因停滞产生的所有信号片段】
                if p not in pos_map:
                    pos_map[p] = []
                pos_map[p].append(i) 
                
                s_str = ev[3]
                if s_str and s_str != "invalid":
                    vals = [_float(x) for x in s_str.split(',')]
                    all_raw_signals.extend(vals)
                else:
                    vals = []
                
                parsed_events.append({
                    'p': p, 'kmer': ev[2], 'vals': vals, 'contig': ev[0]
                })
            except ValueError: continue

        if not parsed_events: continue

        # Read 级归一化
        read_median, read_scale = get_norm_params(all_raw_signals)
        
        # 染色体名称兼容处理
        raw_chrom = parsed_events[0]['contig']
        chrom_keys = [raw_chrom]
        if raw_chrom.startswith('chr'): chrom_keys.append(raw_chrom[3:])
        else: chrom_keys.append(f"chr{raw_chrom}")

        processed_sites = set()

        for idx, event in enumerate(parsed_events):
            kmer_start_pos = event['p']
            center_pos = kmer_start_pos
            if center_pos in processed_sites: continue
            center_base_ref = event['kmer'][0]
            target_strand = None
            
            # 【配合修复3：根据靶点链和 Read 真实方向进行阻断，避免正负链错配污染】
            if center_base_ref == 'C': 
                if not is_forward_read: continue 
                target_strand = '+'
            elif center_base_ref == 'G': 
                if is_forward_read: continue 
                target_strand = '-'
            else: continue 

            # 查找标签和类型
            found_label_info = None
            final_chrom_used = raw_chrom
            
            for c_key in chrom_keys:
                if (c_key, center_pos, target_strand) in GLOBAL_LABELS:
                    found_label_info = GLOBAL_LABELS[(c_key, center_pos, target_strand)]
                    final_chrom_used = c_key 
                    break
            
            if found_label_info is None: continue
            
            label_val, motif_type = found_label_info

            # 提取上下文
            full_context_found = True
            context_indices_lists = []
            for i in range(-HALF_LEN, HALF_LEN + 1):
                p = center_pos + i  
                if p in pos_map: 
                    context_indices_lists.append(pos_map[p]) 
                else:
                    full_context_found = False
                    break
            
            if not full_context_found: continue

            # 组装特征
            seq_list = []
            base_means = []
            base_stds = []
            base_lens = []
            base_signals = []
            
            for c_indices in context_indices_lists:
                # 取该物理位置第一个 event 的参考碱基
                first_e_obj = parsed_events[c_indices[0]]
                seq_list.append(first_e_obj['kmer'][0])
                
                # 【配合修复1：拼接同一位置的多个 Event 信号，还原 Dwell Time】
                combined_vals = []
                for c_idx in c_indices:
                    combined_vals.extend(parsed_events[c_idx]['vals'])
                
                base_lens.append(str(len(combined_vals)))

                if len(combined_vals) > ABNORMAL_SIGNAL_CAP: proc_vals = combined_vals[:ABNORMAL_SIGNAL_CAP]
                else: proc_vals = combined_vals
                
                if read_scale == 0: norm_scale = 1.0 
                else: norm_scale = read_scale
                
                norm_vals = [(v - read_median) / norm_scale for v in proc_vals]
                
                # 【修复4：在定长采样和补零之前，基于完整信号计算真实均值与方差】
                _m = _mean(norm_vals) if norm_vals else 0.0
                _s = _std(norm_vals) if norm_vals else 0.0
                base_means.append(f"{_m:.5f}")
                base_stds.append(f"{_s:.5f}")

                # 仅将定长采样用于序列信号拼接
                sampled_vals = fixed_len_sampling(norm_vals, DEFAULT_POINTS_PER_BASE)
                base_signals.append(",".join([f"{x:.3f}" for x in sampled_vals]))

            seq_str = "".join(seq_list)
            
            if target_strand == '-':
                seq_str = get_reverse_complement(seq_str)
                base_means = base_means[::-1]
                base_stds = base_stds[::-1]
                base_lens = base_lens[::-1]
                base_signals = base_signals[::-1]

            row = [
                final_chrom_used, str(center_pos), target_strand, str(center_pos), read_id, 't',
                seq_str,
                ",".join(base_means),
                ",".join(base_stds),
                ",".join(base_lens),
                ";".join(base_signals),
                str(label_val),  
                motif_type       
            ]
            
            batch_results.append("\t".join(row))
            processed_sites.add(center_pos)
            
    return batch_results

# ================= 3. 主进程逻辑 =================

def load_labels_multi(file_config):
    print(f"[Info] Loading Labels for ALL contexts...", file=sys.stderr)
    global GLOBAL_LABELS
    total_count = 0
    
    for item in file_config:
        f_path = item['path']
        lbl = item['label']
        mtf = item['motif']
        
        print(f"  -> Loading {mtf} ({'Pos' if lbl==1 else 'Neg'}): {os.path.basename(f_path)}", file=sys.stderr)
        
        if not os.path.exists(f_path):
            print(f"  [Warning] File not found: {f_path}, skipping.", file=sys.stderr)
            continue

        opener = gzip.open(f_path, 'rt') if f_path.endswith('.gz') else open(f_path, 'r')
        
        local_count = 0
        with opener as f:
            for line in f:
                if not line.strip() or line.startswith('#'): continue
                parts = line.strip().split()
                try: 
                    chrom = parts[0]
                    pos = int(parts[1])
                    
                    strand = '+'
                    if len(parts) >= 3 and parts[2] in ['+', '-']: strand = parts[2]
                    elif len(parts) >= 6 and parts[5] in ['+', '-']: strand = parts[5]
                    
                    GLOBAL_LABELS[(chrom, pos, strand)] = (lbl, mtf)
                    local_count += 1
                except: continue
        total_count += local_count
        print(f"     Loaded {local_count} sites.", file=sys.stderr)

    print(f"[Info] Total Labels Loaded: {total_count}", file=sys.stderr)
    if GLOBAL_LABELS:
        print(f"[Info] Sample check: {list(GLOBAL_LABELS.keys())[0]} -> {list(GLOBAL_LABELS.values())[0]}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cg_pos", required=True)
    parser.add_argument("--cg_neg", required=True)
    parser.add_argument("--chg_pos", required=True)
    parser.add_argument("--chg_neg", required=True)
    parser.add_argument("--chh_pos", required=True)
    parser.add_argument("--chh_neg", required=True)
    
    parser.add_argument("--out", required=True)
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
    print(f"[Info] Processing started. Output will include 'motif' column.", file=sys.stderr)
    
    with gzip.open(args.out, 'wt', newline='') as f_out:
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
            
            # 【配合修复3：动态提取 event_index 列用于推断链方向】
            idx_event_index = col_map.get('event_index', 5) 
            if idx_samples is None: idx_samples = 8
                
        except Exception as e:
            sys.exit(1)

        curr_batch = []
        temp_read_id = None
        temp_events = []
        tasks_submitted = 0

        active_tasks =[]

        for line in sys.stdin:
            parts = line.split('\t')
            try:
                rid = parts[idx_read]
                # 【配合修复3：将 event_index 一并传入元组用于排序和方向比对】
                event_data = (parts[idx_contig], parts[idx_pos], parts[idx_kmer], parts[idx_samples].strip(), parts[idx_event_index])
            except IndexError: continue

            if rid != temp_read_id:
                if temp_read_id is not None:
                    curr_batch.append((temp_read_id, temp_events))
                    if len(curr_batch) >= BATCH_SIZE:
                        task = pool.apply_async(process_batch_task, (curr_batch,), 
                                         callback=lambda res: [f_out.write(l + '\n') for l in res])
                        active_tasks.append(task)
                        curr_batch = []
                        tasks_submitted += 1
                        if tasks_submitted * BATCH_SIZE % 2000 == 0:
                            sys.stderr.write(f"\rReads Processed: ~{tasks_submitted * BATCH_SIZE}")
                        
                        while len(active_tasks) >= args.p * 2:
                            # 过滤掉已经执行完毕的任务
                            active_tasks = [t for t in active_tasks if not t.ready()]
                            # 如果清理后还是拥堵，主进程就睡 0.05 秒，强行踩刹车，不继续读文件了
                            if len(active_tasks) >= args.p * 2:
                                time.sleep(0.05)
                temp_read_id = rid
                temp_events = []
            temp_events.append(event_data)
        
        if temp_read_id is not None:
            curr_batch.append((temp_read_id, temp_events))
        if curr_batch:
            pool.apply_async(process_batch_task, (curr_batch,), 
                             callback=lambda res: [f_out.write(l + '\n') for l in res])

        pool.close()
        pool.join()
        
    print(f"\n[Success] All Done.", file=sys.stderr)

if __name__ == "__main__":
    main()