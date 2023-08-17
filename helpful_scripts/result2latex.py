def dict_to_latex_table(data, keys):
    # Define the mapping from keys to column names
    column_names = {
        'naive': 'Benign',
        'badnet': 'BadNet',
        'signal': 'SIG',
        'hk': 'Blended',
        'narci': 'Our Attack'
    }

    # Preprocessing: find the max values for each method
    max_values = {}
    for method, values in data.items():
        max_values[method] = {
            'max_asr': max(values[key][0] for key in keys),  # Only consider chosen keys
            'max_cacc': max(values[key][1] for key in keys)  # Only consider chosen keys
        }

    # Generate LaTeX table header
    latex_table = r"""
    \begin{table*}[]
    \begin{tabular}{l"""

    for key in keys:
        latex_table += "|cc"
    latex_table += "}\n\\hline\n"

    # Generate column names
    latex_table += "\\multicolumn{1}{l}{\\multirow{2}{*}{Methods}}"
    for key in keys:
        latex_table += f" & \\multicolumn{2}{{c|}}{{{column_names[key]}}}"  # Use column names
    latex_table += "\\\\ \\cline{2-" + str(2*len(keys)+1) + "}\n"
    latex_table += "\\multicolumn{1}{l}{}"
    for _ in keys:
        latex_table += " & \\textbf{ASR} & \\textbf{CACC}"
    latex_table += "\\\\ \\hline"

    # Generate rows
    for method, values in data.items():
        latex_table += "\n" + f"{method}"
        for key in keys:
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
    
    latex_table += "\n" + r"\hline"+"\n"+r"\end{tabular}"+"\n"+r"\end{table*}"
    
    return latex_table

def get_dic(path, inspect_round='100'):
    import re
    from pathlib import Path

    PREFIXES = ['Ditto', 'FedAvg', 'FedRep', 'pFedMe']
    SUBDIRS = ['normal', 'backdoor_hkTrigger', 'backdoor_narciTrigger', 'backdoor_signalTrigger', 'backdoor_squareTrigger']
    KEYS = ['naive', 'hk', 'narci', 'signal', 'badnet']

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
                    for dirpath in find_deepest_dirs(subdir_dir):
                        log_file = dirpath / 'eval_results.log'
                        if log_file.exists():
                            with log_file.open() as f:
                                for line in f:
                                    # 用正则表达式来查找和提取 'Round', 'test_acc', 'poison_acc' 的值
                                    round_match = re.search(r"'Round': (\d+)", line)
                                    test_acc_match = re.search(r"'test_acc': ([\d\.]+)", line)
                                    poison_acc_match = re.search(r"'poison_acc': ([\d\.]+)", line)
                                    if round_match and test_acc_match:
                                        
                                        round_value = round_match.group(1)
                                        if round_value == inspect_round:  # 如果 'Round' 是 100，那么提取 'test_acc' 和 'poison_acc' 的值
                                            test_acc = float(test_acc_match.group(1))
                                            if poison_acc_match:
                                                poison_acc = float(poison_acc_match.group(1))
                                            else:
                                                # naive没有poison_acc
                                                poison_acc = 0.0
                                            data[prefix][key] = (poison_acc, test_acc)
                                            break

    return data

if __name__ == '__main__':
    all_keys = ['naive', 'badnet', 'signal', 'hk', 'narci']
    keys_omit_narci = ['naive', 'badnet', 'signal', 'hk']
    # path = './new-output'
    inspect_round = '200'
    # path = './200-rounds-output'
    # path = '200-multiattack-output-2-20-resnet'
    path = '200-single-output-resnet'
    # path = '200-multiattack-output-2-20'
    dic = get_dic(path, inspect_round)
    print(dic)
    # dic['Ditto']['hk'] = (0.8172,0.7569)
    dic = {
        'Ditto': {
            'naive': (0.0, 0.75509409),
            'hk': (0.14683204827355012, 0.7493),
            'narci': (0.7576645626690712, 0.7464),
            'signal': (0.01994808, 0.7443),
            'badnet': (0.02017799, 0.7409)
        },
        'FedAvg': {
            'naive': (0.0, 0.7293),
            'hk': (0.36350430215666557, 0.7361),
            'narci': (0.7459422903516681, 0.7292),
            'signal': (0.017208626662196892, 0.7264),
            'badnet': (0.019443513241702983, 0.7276)
        },
        'FedRep': {
            'naive': (0.0, 0.7972),
            'hk': (0.04045144708906023, 0.7886),
            'narci': (0.7994815148782687, 0.791),
            'signal': (0.02257235445301151, 0.7845),
            'badnet': (0.021231422505307854, 0.7843)
        },
        'pFedMe': {
            'naive': (0.0, 0.6293),
            'hk': (0.12191306291205721, 0.6359),
            'narci': (0.79661158, 0.6427),
            'signal': (0.024918985361492903, 0.6276),
            'badnet': (0.039557492457257794, 0.6297)
        }
    }
    latex = dict_to_latex_table(dic, keys=all_keys)
    print(latex)