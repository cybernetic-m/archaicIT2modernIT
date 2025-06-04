import json
import time
import random
from collections import defaultdict, Counter
import os, sys
import pandas as pd
import re


prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../prompt'))
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
gold_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/dataset_gold.csv'))
sys.path.append(prompt_path)
sys.path.append(utils_path)
sys.path.append(gold_path)
from PromptBuilder import PromptBuilder
from config import load_config
from translate import clean_reasoning, call_translation_api

def prometheus_choice(model, tokenizer, user_content, device='cuda'):
    messages = [
    {"role": "user", "content": user_content},
    ]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]


def single_char(judge_output):
  # Extract score from output
  match = re.search(r"\[Result\]:\s*Response\s+([AB])\s+is\s+better", judge_output)
  if match:
      return str(match.group(1))
  else:
      return None
  
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



def load_translations(path):
    """loads a JSON lines e returns a dict {Sentence: Translation}"""
    sentence_map = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentence = data["Sentence"].strip()
            translation = data["Translation"].strip()
            sentence_map[sentence] = translation
    return sentence_map


def compare_translations(fileA, fileB):
    """
    Takes 2 translation files and a gold standard CSV,
    returns a list of dicts with original sentence, gold, A, and B translations.
    """
    data_a = load_translations(fileA)  # should return a dict: {sentence: translation}
    data_b = load_translations(fileB)
    gold_data = load_gold(gold_path)

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
    except:
        return ""


def make_match(sentences_data, judge_model, judge_tokenizer):
    """guven all the translations returns the model who performed best between two"""
    score_A = 0
    score_B = 0
    cont = 0
    for sentences in sentences_data:
        print(f"{cont}, ", end='')
        winner = get_winner(sentences['A'], sentences['B'], sentences['gold'], judge_model, judge_tokenizer).strip()
        time.sleep(60 / 29)

        if winner == 'A':
            score_A += 1
        elif winner == 'B':
            score_B += 1
        else:
            print \
                (
                    f"\n - llm did not answer with A or B on '{sentences['Sentence']}', but: '{winner}', thus no score assigned")

        cont += 1

    print(f"\n - Score A: {score_A}, Score B: {score_B}")
    if score_A > score_B:
        return "A"
    elif score_A < score_B:
        return "B"
    else:
        return random.choice(["A", "B"])



def save_winner(text):
    with open("winners.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def tournament(files, judge_model, judge_tokenizer, prompt_builder):
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
        data = compare_translations(player1, player2)
        print(player1, "vs", player2)

        winner = make_match(data, judge_model, judge_tokenizer, prompt_builder)

        if winner == "A":
            winner = player1
            print(f"  - winner: {player1}")
        elif winner == "B":
            winner = player2
            print(f"  - winner: {player2}")

        match_winner.append(winner)

    print("\n - Winners of this round:", [clean_text(w) for w in match_winner])

    return tournament(match_winner, judge_model, judge_tokenizer, prompt_builder)


def make_evaluation(to_eval, output_file_path, api_key):
    """asks the llm to make the evaluation given a file path to_eval that contains the original sentence and translations"""

    if not api_key:
        print("Warning: insert an api key")
        return

    data = load_translations(to_eval)
    config = load_config("archaicIT2modernIT/config.yaml")
    prompt = (config['prompt']['system_template_judge'], config['prompt']['user_template_judge'])

    system_prompt_template = prompt[0]
    print(system_prompt_template)
    prompt_builder_judge = PromptBuilder(prompt_template=prompt,
                                         mode="zero-shot",
                                         lang='en'
                                         )

    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for original in data:
            translation = data[original]
            user_prompt_template = prompt_builder_judge.build_judge_prompt(
                tournament=False,
                old_sentence=original,
                translation=translation
            )

            try:
                evaluation = clean_reasoning(
                    call_translation_api(
                        api_key,
                        "llama3-70b-8192",
                        system_prompt_template,
                        user_prompt_template,
                        0.0
                    )
                )
            except Exception as e:
                evaluation = ''

            print(original.strip())
            print(translation.strip())
            print(evaluation.strip())
            print()
            json_line = {
                "original": original,
                "translation": translation,
                "evaluation": evaluation
            }
            f_out.write(json.dumps(json_line, ensure_ascii=False) + '\n')

            time.sleep(60 / 29)


def parse_evaluation(evaluation_str):
    """cleans the evaluation string"""
    parts = [part.strip() for part in evaluation_str.split(",")]
    eval_dict = {}
    for part in parts:
        if ":" in part:
            key, value = part.split(":")
            if value:
                eval_dict[key.strip()] = int(value.strip())
    return eval_dict


def print_stats(file_path):
    """returns and prints the stats for each metric given an evaluation file"""
    metrics = ["Meaning Preservation", "Grammar", "Style Matching", "Structural Alignment", "Completeness"]

    scores = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            eval_dict = parse_evaluation(record["evaluation"])
            for metric in metrics:
                score = eval_dict.get(metric)
                if score is not None:
                    scores[metric].append(score)

    # statistics for each metric
    total_score = 0
    for metric in metrics:
        values = scores[metric]
        print(f"\n {metric}:")
        print(f"  - Media: {sum(values) / len(values):.2f}")
        # print(f"  - Min: {min(values)}, Max: {max(values)}")
        print(f"  - Distribuzione: {dict(Counter(values))}")
        total_score += sum(values)

    print(f"\n  - Totale: {total_score}")

    return scores
