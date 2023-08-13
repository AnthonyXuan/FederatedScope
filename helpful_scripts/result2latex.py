def dict_to_latex_table_no_textbf(data):
    latex_table = r"""
    \begin{table*}[]
    \begin{tabular}{l|cc|cc|cc|cc|cc}
    \hline
    \multicolumn{1}{l}{\multirow{2}{*}{Methods}} & \multicolumn{2}{c|}{Benign}    & \multicolumn{2}{c|}{BadNet}    & \multicolumn{2}{c|}{SIG}     & \multicolumn{2}{c|}{Blended} & \multicolumn{2}{c}{Our Attack}    \\ \cline{2-11}
    \multicolumn{1}{l}{}                         & \textbf{ASR} & \textbf{CACC}   & \textbf{ASR} & \textbf{CACC}   & \textbf{ASR} & \textbf{CACC} & \textbf{ASR} & \textbf{CACC} & \textbf{ASR}    & \textbf{CACC}   \\ \hline"""
    
    for method, values in data.items():
        latex_table += "\n" + f"{method}"
        for key in ['naive', 'badnet', 'signal', 'hk', 'narci']:
            latex_table += " & " + " & ".join(f"{v:.4f}" for v in values[key])
        latex_table += r" \\"
    
    latex_table += r"\n\hline\n\end{tabular}\n\end{table*}"
    
    return latex_table

def dict_to_latex_table(data):
    # Preprocessing: find the max values for each method
    max_values = {}
    for method, values in data.items():
        max_values[method] = {
            'max_asr': max(values[key][0] for key in values),
            'max_cacc': max(values[key][1] for key in values)
        }
    
    # Generate LaTeX table
    latex_table = r"""
    \begin{table*}[]
    \begin{tabular}{l|cc|cc|cc|cc|cc}
    \hline
    \multicolumn{1}{l}{\multirow{2}{*}{Methods}} & \multicolumn{2}{c|}{Benign}    & \multicolumn{2}{c|}{BadNet}    & \multicolumn{2}{c|}{SIG}     & \multicolumn{2}{c|}{Blended} & \multicolumn{2}{c}{Our Attack}    \\ \cline{2-11}
    \multicolumn{1}{l}{}                         & \textbf{ASR} & \textbf{CACC}   & \textbf{ASR} & \textbf{CACC}   & \textbf{ASR} & \textbf{CACC} & \textbf{ASR} & \textbf{CACC} & \textbf{ASR}    & \textbf{CACC}   \\ \hline"""
    
    for method, values in data.items():
        latex_table += "\n" + f"{method}"
        for key in ['naive', 'badnet', 'signal', 'hk', 'narci']:
            asr, cacc = values[key]
            # Bold the max values
            asr_str = f"{asr:.4f}"
            cacc_str = f"{cacc:.4f}"
            if asr == max_values[method]['max_asr']:
                asr_str = f"\\textbf{{{asr_str}}}"
            if cacc == max_values[method]['max_cacc']:
                cacc_str = f"\\textbf{{{cacc_str}}}"
            latex_table += " & " + asr_str + " & " + cacc_str
        latex_table += r" \\"
    
    latex_table += r"\n\hline\n\end{tabular}\n\end{table*}"
    
    return latex_table

def get_data(path):
    import re
    from pathlib import Path

    PREFIXES = ['FedAvg', 'Ditto', 'pFedMe', 'FedRep']
    
    SUBDIRS = ['backdoor_hkTrigger', 'backdoor_narciTrigger', 'backdoor_signalTrigger']
    SUBDIRS += ['backdoor_squareTrigger']
    
    KEYS = ['hk', 'narci', 'signal']
    KEYS += ['badnet','naive']

    # 初始化二层字典
    data = {prefix: {key: () for key in KEYS} for prefix in PREFIXES}

    def find_deepest_dirs(root):
        dirs = [d for d in root.iterdir() if d.is_dir()]
        if not dirs:  # 如果不存在子目录，说明已经到了最深层，返回当前目录
            return [root]
        else:  # 否则，递归查找所有子目录的最深层目录
            return [deepest for d in dirs for deepest in find_deepest_dirs(d)]

    root = Path(path)
    for prefix in PREFIXES:
        prefix_dir = next((d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix)), None)
        if prefix_dir is not None:
            for subdir, key in zip(SUBDIRS, KEYS):
                subdir_dir = next((d for d in prefix_dir.iterdir() if d.is_dir() and d.name.startswith(subdir)), None)
                if subdir_dir is not None:
                    # 针对 'backdoor_squareTrigger' 子目录进行额外处理
                    if key == 'badnet' or key == 'naive':
                        subdir_dirs = [d for d in subdir_dir.iterdir() if d.is_dir() and (d.name.startswith('badnet') or d.name.startswith('naive'))]
                        for d in subdir_dirs:
                            key = 'badnet' if d.name.startswith('badnet') else 'naive'
                            for dirpath in find_deepest_dirs(d):
                                log_file = dirpath / 'eval_results.log'
                                if log_file.exists():
                                    with log_file.open() as f:
                                        for line in f:
                                            # 用正则表达式来查找和提取 'Round', 'test_acc', 'poison_acc' 的值
                                            round_match = re.search(r"'Round': (\d+)", line)
                                            test_acc_match = re.search(r"'test_acc': ([\d\.]+)", line)
                                            poison_acc_match = re.search(r"'poison_acc': ([\d\.]+)", line)
                                            if round_match and test_acc_match and poison_acc_match:
                                                round_value = round_match.group(1)
                                                if round_value == '100':  # 如果 'Round' 是 100，那么提取 'test_acc' 和 'poison_acc' 的值
                                                    test_acc = float(test_acc_match.group(1))
                                                    poison_acc = float(poison_acc_match.group(1))
                                                    data[prefix][key] = (poison_acc, test_acc)
                                                    break
                    else:
                        for dirpath in find_deepest_dirs(subdir_dir):
                            log_file = dirpath / 'eval_results.log'
                            if log_file.exists():
                                with log_file.open() as f:
                                    for line in f:
                                        # 用正则表达式来查找和提取 'Round', 'test_acc', 'poison_acc' 的值
                                        round_match = re.search(r"'Round': (\d+)", line)
                                        test_acc_match = re.search(r"'test_acc': ([\d\.]+)", line)
                                        poison_acc_match = re.search(r"'poison_acc': ([\d\.]+)", line)
                                        if round_match and test_acc_match and poison_acc_match:
                                            round_value = round_match.group(1)
                                            if round_value == '100':  # 如果 'Round' 是 100，那么提取 'test_acc' 和 'poison_acc' 的值
                                                test_acc = float(test_acc_match.group(1))
                                                poison_acc = float(poison_acc_match.group(1))
                                                data[prefix][key] = (poison_acc, test_acc)
                                                break

    print(data)
    return data

path = './new-output'
latex = dict_to_latex_table(get_data(path))
print(latex)