import json
import re
from tqdm import tqdm


prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. "\
        "Write a response that appropriately completes the request.\n\n"\
        "### Instruction:\n{instruction}\n\n{input}\n\n{question}\n\n### Response:\n"

with open("./data/processed/skginstruct_test_file_7b.json", "r") as f:
    data = json.load(f)

# select everything between <</SYS>> and [/INST]
examples = {}
new_data = []
for i, each in tqdm(enumerate(data)):
    struct_in_idx = each['formatted_input'].find(each['struct_in'][:10])
    sysend_idx = each['formatted_input'].find("<</SYS>>")
    instruction = each['formatted_input'][sysend_idx + 8:struct_in_idx]
    if each['text_in']:
        text_in_idx = each['formatted_input'].find(each['text_in'])
        struct_in = each['formatted_input'][struct_in_idx:text_in_idx].strip()
    else:
        end  = each['formatted_input'].find("[/INST]")
        struct_in = each['formatted_input'][struct_in_idx:end].strip()
    data[i]['formatted_input'] = prompt.format(instruction=instruction, input=struct_in, question=each['text_in'])
    if each['arg_path'] not in examples:
        examples[each['arg_path']] = data[i]
    new_data.append(data[i])

# write examples formatted inputs into a text file
printstr = ""
for k, v in examples.items():
    printstr += f"### {k}\n\n"
    printstr += f"{v['formatted_input']}\n\n"
    printstr += "--------------------------------\n\n"

with open("data/processed/skginstruct_test_file_mistral_examples.txt", "w") as f:
    f.write(printstr)


with open("data/processed/skginstruct_test_file_mistral.json", "w") as f:
    json.dump(new_data, f)
