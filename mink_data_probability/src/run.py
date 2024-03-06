import logging
logging.basicConfig(level='ERROR')
import numpy as np
from pathlib import Path
import torch
import zlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from options import Options
from eval import *
from accelerate import Accelerator



def load_model(name1, device):
    olmo_path = "/cs/student/projects3/aisoc/snlp_stuff_tmp/olmo1b_unlearned"
    model1 = AutoModelForCausalLM.from_pretrained(name1, return_dict=True, cache_dir=args.cache_dir, trust_remote_code=True)
    model1.to(device)
    model1.eval()
    # TODO: TEMP ADD OLMO TOKENIZER NAME
    #olmo_name = "allenai/OLMo-1B"
    olmo_name = "facebook/opt-1.3B"
    tokenizer1 = AutoTokenizer.from_pretrained(olmo_name, cache_dir=args.cache_dir)
    """
    model1 = AutoModelForCausalLM.from_pretrained(olmo_path)
    tokenizer1 = AutoTokenizer.from_pretrained(olmo_name, cache_dir=args.cache_dir)
    """

    return model1, tokenizer1


     
def calculatePerplexity(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(gpu)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    
    '''
    extract logits:
    '''
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


def inference(model1, tokenizer1, text, ex, modelname1):
    pred = {}

    p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, gpu=model1.device)
    #p_lower, _, p_lower_likelihood = calculatePerplexity(text.lower(), model1, tokenizer1, gpu=model1.device)

   # ppl
    #pred["ppl"] = p1

    # Ratio of log ppl of lower-case and normal-case
    #pred["ppl/lowercase_ppl"] = -(np.log(p_lower) / np.log(p1)).item()
    # Ratio of log ppl of large and zlib
    #zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    #pred["ppl/zlib"] = np.log(p1)/zlib_entropy
    # min-k prob
    for ratio in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        k_length = int(len(all_prob)*ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        pred[f"Min_{ratio*100}% Prob"] = -np.mean(topk_prob).item()
    ex["pred"] = pred
    return ex


def evaluate_data(test_data, model1, tokenizer1, col_name, modelname1):
    print(f"all data size: {len(test_data)}")
    all_output = []
    test_data = test_data
    for ex in tqdm(test_data): 
        text = ex[col_name]
        new_ex = inference(model1, tokenizer1, text, ex, modelname1)
        all_output.append(new_ex)
    return all_output


if __name__ == '__main__':
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    args = Options()
    args = args.parser.parse_args()
    args.output_dir = f"{args.output_dir}/{args.target_model}/{args.key_name}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and data
    model1, tokenizer1 = load_model(args.target_model, device)
    if "jsonl" in args.data:
        data = load_jsonl(f"{args.data}")
    else: # load data from huggingface
        #dataset = load_dataset(args.data, split=f"WikiMIA_length{args.length}", cache_dir=args.cache_dir)
        dataset = load_dataset(args.data, split=args.split, cache_dir=args.cache_dir)
        data = convert_huggingface_data_to_list_dic(dataset, args.max_samples)

    all_output = evaluate_data(data, model1, tokenizer1, args.key_name, args.target_model)
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ratios_list = [[] for i in  ratios]
    for example in all_output:
        for i, ratio in  enumerate(ratios):
            ratios_list[i].append(example["pred"][f"Min_{ratio*100}% Prob"])

    for i, l in enumerate(ratios_list):
        print(f"Avg over all samples: Min_{ratios[i]*100}% Prob", np.mean(l))
    #model1.save_pretrained("../olmo_model")
    #fig_fpr_tpr(all_output, args.output_dir)#, args.key_name)

