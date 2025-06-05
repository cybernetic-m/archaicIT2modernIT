import json
import time
import random
from collections import defaultdict, Counter
import os, sys
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np


# You can set this variable for debug if you want to evaluate a subset of translations
# If you want to evaluate all the translations, set it to 999999999
number_of_translation = 600000


def prometheus_choice(model, tokenizer, user_content, device='cuda'):
    messages = [
    {"role": "user", "content": user_content},
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1500, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]

'''
def prometheus_choice(model, tokenizer, user_content, device='cuda'):
    messages = [{"role": "user", "content": user_content}]

    # Apply chat template and tokenize with padding and truncation
    encodeds = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )

    input_ids = encodeds["input_ids"].to(device)
    attention_mask = encodeds["attention_mask"].to(device)

    # Ensure pad_token_id is defined
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    model.to(device)

    # Generate tokens
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1000,
        do_sample=True,
        pad_token_id=model.config.pad_token_id
    )

    # Decode output
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded

'''

def single_char(judge_output):
  # Extract score from output
  match = re.search(r"\[Result\]:\s*Response\s+([AB])\s+is\s+better", judge_output)
  if match:
      return str(match.group(1))
  else:
      return "error: no match" + judge_output

def single_num(judge_output):
  # Extract score from outputmatch
  match = re.search(r"\[RESULT\]\s*:?\s*(\d+)", judge_output)
  if match:
      return str(match.group(1))
  else:
      return "error: no match" + judge_output

def clean_text(text):
    """just useful to make prints smaller"""
    text = text.split("/")[-1]
    text = text.replace("CaponataLovers-hw2_transl-", "")
    text = text.replace("-r1-distill-llama-70b", "")
    return text


def load_gold(path):
    """Loads gold translations from a CSV into a dictionary using pandas."""
    df = pd.read_csv(path)
    return dict(zip(df['Sentence'].str.strip(), df['Modern'].str.strip()))



def load_translations(path, num):
    """loads a JSON lines e returns a dict {Sentence: Translation}"""
    sentence_map = {}
    cont = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
          if cont < num:
            data = json.loads(line)
            sentence = data["Sentence"].strip()
            translation = data["Translation"].strip()
            sentence_map[sentence] = translation
            cont+=1
    return sentence_map


def compare_translations(fileA, fileB, FileGold):
    """
    Takes 2 translation files and a gold standard CSV,
    returns a list of dicts with original sentence, gold, A, and B translations.
    """
    data_a = load_translations(fileA, number_of_translation)  # should return a dict: {sentence: translation}
    data_b = load_translations(fileB, number_of_translation)
    gold_data = load_gold(FileGold)

    # Intersect keys present in all three sources
    common_sentences = set(data_a.keys()) & set(data_b.keys()) & set(gold_data.keys())

    result = []
    for sentence in common_sentences:
        original_length = len(sentence)
        result.append({
            "Sentence": sentence,
            "gold": gold_data[sentence],
            "A": data_a[sentence][:original_length * 2],
            "B": data_b[sentence][:original_length * 2]
        })
    return result


def get_winner(A, B, gold, judge_model, judge_tokenizer, prompt_builder):
   
    user_prompt = prompt_builder.build_prometheus_prompt(mode='relative', A= A, B = B, gold = gold)
    system_prompt = prompt_builder.getSystemPrompt()
    
    try:
        user_content = system_prompt + "\n\n" + user_prompt
        return single_char(prometheus_choice(judge_model, judge_tokenizer, user_content))
    except Exception as e:
        return e


def make_match(sentences_data, judge_model, judge_tokenizer, prompt_builder):
    """Given all the translations, returns the model who performed best between two"""
    score_A = 0
    score_B = 0
    cont = 0

    for sentences in sentences_data:
        print(f"{cont}, ", end='')

        # randomize A e B
        items = [('A', sentences['A']), ('B', sentences['B'])]
        random.shuffle(items)
        label1, trans1 = items[0]
        label2, trans2 = items[1]

        # get best
        winner = get_winner(trans1, trans2, sentences['gold'], judge_model, judge_tokenizer, prompt_builder)


        # Map answer to real A and B
        if winner == 'A':
            actual_winner = label1
        elif winner == 'B':
            actual_winner = label2
        else:
            print(f"\n - llm did not answer with A or B on {cont}")
            cont += 1
            continue

        # Assegna il punto al vero A o vero B
        if actual_winner == 'A':
            score_A += 1
        elif actual_winner == 'B':
            score_B += 1

        cont += 1

    print(f"\n - Score A: {score_A}, Score B: {score_B}", end = ' ')
    if score_A > score_B:
        return "A"
    elif score_A < score_B:
        return "B"
    else:
        return random.choice(["A", "B"])



def save_winner(text):
    name =  "winners.txt"
    with open(name, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def tournament(files, judge_model, judge_tokenizer, prompt_builder, gold_path):
    """makes the tournament where an llm decides if it's better translation A or B"""
    if len(files) == 1:
        print("\n\n ----- Final winner:", files[0].split("/")[-1])
        save_winner(files[0])
        return files[0]

    match_winner = []

    if len(files) % 2 == 1:
        print(" - Bye:", clean_text(files[-1]), "goes to next round")
        match_winner.append(files[-1])
        files = files[:-1]

    for i in range(0, len(files), 2):
        player1 = files[i]
        player2 = files[i + 1]
        data = compare_translations(player1, player2, gold_path)
        print("\n", clean_text(player1), "vs", clean_text(player2))

        winner = make_match(data, judge_model, judge_tokenizer, prompt_builder)

        if winner == "A":
            winner = player1
            print(f"  - winner: {clean_text(player1)}")
        elif winner == "B":
            winner = player2
            print(f"  - winner: {clean_text(player2)}")

        match_winner.append(winner)

    print("\n - Winners of this round:", [clean_text(w) for w in match_winner])
    print("\n ---STARTING NEXT ROUND---")

    return tournament(match_winner, judge_model, judge_tokenizer, prompt_builder, gold_path)


def make_evaluation(to_eval, output_file_path, judge_model, judge_tokenizer, prompt_builder, rubrics, gold_path):
    """asks the llm to make the evaluation given a file path to_eval that contains the original sentence and translations"""
    data = load_translations(to_eval, number_of_translation)
    gold_data = load_gold(gold_path)

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        count = 0
        for original in data:
            print(f"Sentence {count} \n")
            evaluations = {}
            translation = data[original]
            gold = gold_data[original]

            for key, rubric in rubrics.items():
                user_prompt = prompt_builder.build_prometheus_prompt(mode="absolute",response=translation, gold=gold, rubric=rubric)
                system_prompt = prompt_builder.getSystemPrompt()

                user_content = system_prompt + "\n\n" + user_prompt

                try:
                    prometheus_evaluation = single_num(prometheus_choice(judge_model, judge_tokenizer, user_content)) 
                    #print(f'evaluation for "{translation} on {key} ": {prometheus_evaluation}, the gold is: {gold}')
                    print(f' - {key}: {prometheus_evaluation} \n')
                except Exception as e:
                    print(e)
                    prometheus_evaluation = ''
                count+=1
                evaluations[key] = prometheus_evaluation

            json_line = {
                "original": original,
                "translation": translation,
                "evaluation": evaluations
            }
            f_out.write(json.dumps(json_line, ensure_ascii=False) + '\n')



def compute_evaluation_stats(jsonl_path):
    """
    Reads a JSONL file with evaluation scores and returns a dictionary
    with statistics for each evaluation metric: sum, mean, and distribution.
    """

    metrics = defaultdict(list)

    # Read the file line by line
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            evaluation = record.get("evaluation", {})

            # Collect values for each metric
            for metric, value in evaluation.items():
                try:
                    metrics[metric].append(float(value))
                except ValueError:
                    continue  # Skip non-numeric values

    # Compute statistics for each metric
    stats = {}

    for metric, values in metrics.items():
        total = sum(values)
        mean = total / len(values) if values else 0
        distribution = dict(Counter(values))  # frequency of each score

        stats[metric] = {
            "sum": total,
            "mean": mean,
            "distribution": distribution
        }

    return stats

def read_file_paths(txt_file_path):
    """
    Reads a text file and returns a list of non-empty, stripped lines (file paths).
    """
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def evaluate_models(file_, model, tokenizer, prompt_builder, rubrics, gold_path):
    for elem in read_file_paths(file_):
        print()
        make_evaluation(elem, f"{clean_text(elem)}_evaluated.jsonl", model, tokenizer, prompt_builder, rubrics, gold_path)


def plot_multiple_models_comparison(model_stats_dict, labels_for_metrics):
    """
    Plots a grouped bar chart comparing mean scores across multiple models for each evaluation metric.

    Parameters:
        model_stats_dict (dict): keys are model names, values are stats dicts from compute_evaluation_stats()
    """
    model_names = list(model_stats_dict.keys())
    metrics = list(next(iter(model_stats_dict.values())).keys())  # get metrics from the first model

    n_models = len(model_names)
    x = np.arange(len(metrics))  # base x positions for metrics
    width = 0.8 / n_models  # bar width depending on number of models

    plt.figure(figsize=(12, 6))

    for i, (model_name, stats) in enumerate(model_stats_dict.items()):
        means = [stats[m]["mean"] for m in metrics]
        x_offset = x + (i - n_models / 2) * width + width / 2
        plt.bar(x_offset, means, width=width, label=model_name)

    plt.xlabel("Evaluation Metric")
    plt.ylabel("Mean Score")
    plt.title("Comparison of Evaluation Metrics Across Models")
    plt.xticks(x, [labels_for_metrics.get(m, m) for m in metrics], rotation=30)

    plt.ylim(0, 5)  # Adjust based on expected score range
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()