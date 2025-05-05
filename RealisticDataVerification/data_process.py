import random
import numpy as np
import re
# set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def modify_output(example):
    output = example['output']
    ans = output.split('The answer is: ')[-1]
    example['output'] =  'The answer is: {}'.format(ans)
    return example


def add_step_noise_to_gsm_fobar(data, error_prob=1.0):
    """
    Precise Noise Addition for GSM_FOBAR Format
    Preserving Final Answer Markers (#### and The answer is)
    """
    output_lines = data['output'].split('\n')
    
    # Locating Target Reasoning Steps for Processing
    process_lines = []
    for i, line in enumerate(output_lines):
        if any(pattern in line for pattern in ['=', '+', '-', '*', '/', '(']) \
           and not line.startswith(('####', 'The answer is')):
            process_lines.append(i)
    
    # Noise Injection into Each Reasoning Step
    for i in process_lines:
        if random.random() < error_prob:
            choice = random.choice(['skip', 'result'])
            if choice == 'skip':
                output_lines[i] = _skip_step_gsm(output_lines[i])
            else:
                output_lines[i] = _wrong_result_gsm(output_lines[i])
    
    return {
        'output': '\n'.join(output_lines),
        'instruction': data['instruction'],
        'type': data['type'],
        'input': data['input']
    }

def _skip_step_gsm(line):
    """Preserving LaTeX Formula Structure in Step Skipping"""
    parts = line.split('=', 1)
    if len(parts) == 2:
        return f"{parts[0]}= ..."
    return line

def _wrong_result_gsm(line):
    """Intelligent Result Modification (with Variable Name Protection)"""
    def modify_number(match):
        num = match.group()
        if num in ['x', 't', '5', '10']:  # Variable Name Protection
            return num
        delta = random.choice([-1, 1, 2])
        return str(int(num) + delta)
    
    # Exclusive Modification of Right-hand-side Numeric Values
    if '=' in line:
        left, right = line.split('=', 1)
        modified_right = re.sub(r'\b\d+\b', modify_number, right)
        return f"{left}= {modified_right}"
    return line