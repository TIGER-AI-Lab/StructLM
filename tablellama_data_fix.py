import json
import re
from tqdm import tqdm


prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. "\
        "Write a response that appropriately completes the request.\n\n"\
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Question:\n{question}\n\n### Response:\n"

with open("./data/processed/skginstruct_test_file_7b.json", "r") as f:
    data = json.load(f)

# select everything between <</SYS>> and [/INST]
new_data = []
for i, each in tqdm(enumerate(data)):
    add = False
    for name in ['bird', 'cosql', 'finqa', 'wikitabletext', 'infotabs', 'sqa']:
        if name in each['arg_path']:
            add = True
    if not add:
        continue
    struct_in_idx = each['formatted_input'].find(each['struct_in'][:10])
    sysend_idx = each['formatted_input'].find("<</SYS>>")
    instruction = each['formatted_input'][sysend_idx + 8:struct_in_idx]
    text_in_idx = each['formatted_input'].find(each['text_in'])
    struct_in = each['formatted_input'][struct_in_idx:text_in_idx]
    data[i]['formatted_input'] = prompt.format(instruction=instruction, input=struct_in, question=each['text_in'])
    new_data.append(data[i])

with open("data/processed/skginstruct_test_file_tablellama.json", "w") as f:
    json.dump(new_data, f)