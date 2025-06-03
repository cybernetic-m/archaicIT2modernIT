import json
import time
import random
from collections import defaultdict, Counter
import os, sys

prompt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../prompt'))
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(prompt_path)
sys.path.append(utils_path)

from PromptBuilder import PromptBuilder
from config import load_config
from translate import clean_reasoning, call_translation_api


def clean_text(text):
    """just useful to make prints smaller"""
    text = text.split("/")[-1]
    text = text.replace("CaponataLovers-hw2_transl-", "")
    text = text.replace("-r1-distill-llama-70b", "")
    return text


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
    """takes 2 files and returns original sentence, translation A, translation B"""
    data_a = load_translations(fileA)
    data_b = load_translations(fileB)

    # print("\n\n", "test", "\n\n")
    # print(data_a)
    # print(data_b)

    common_sentences = list(set(data_a.keys()) & set(data_b.keys()))

    print(len(common_sentences))
    result = []
    for sentence in common_sentences:
        original_length = len(sentence)
        result.append({
            "Sentence": sentence,
            "A": data_a[sentence][0:original_length * 2],  # to not waste tokens on wrong translations
            "B": data_b[sentence][0:original_length * 2]
        })
    return result


def make_match(sentences_data, api_key):
    """guven all the translations returns the model who performed best between two"""
    score_A = 0
    score_B = 0
    cont = 0
    for sentences in sentences_data:
        print(f"{cont}, ", end='')
        winner = get_winner(sentences['Sentence'], sentences['A'], sentences['B'], api_key).strip()
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


def get_winner(old_sentence, A, B, api_key):
    """given the original sentence and two translations asks the llm which one is better A or B"""
    if (api_key == '') or (not api_key):
        print("Warning: submit an api key")
        return

    config = load_config("archaicIT2modernIT/config.yaml")

    prompt_judge = (
    config['prompt']['system_template_judge_tournament_en'], config['prompt']['user_template_judge_tournament_en'])

    prompt_builder_tournament = PromptBuilder(prompt_template=prompt_judge,
                                              mode="zero-shot",
                                              lang='en'
                                              )

    user_prompt_template = prompt_builder_tournament.build_judge_prompt(True, old_sentence, A, B)

    system_prompt_template = prompt_builder_tournament.getSystemPrompt()
    try:
        return clean_reasoning(
            call_translation_api(api_key, "llama3-70b-8192", system_prompt_template, user_prompt_template, 0.0))
    except:
        return ""


def save_winner(text):
    with open("winners.txt", "a", encoding="utf-8") as f:
        f.write(text + "\n")


def tournament(files, api_key):
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

        winner = make_match(data, api_key)

        if winner == "A":
            winner = player1
            print(f"  - winner: {player1}")
        elif winner == "B":
            winner = player2
            print(f"  - winner: {player2}")

        match_winner.append(winner)

    print("\n - Winners of this round:", [clean_text(w) for w in match_winner])

    return tournament(match_winner, api_key)


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
