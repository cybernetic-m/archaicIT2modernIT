import pandas as pd
import json 
from sklearn.metrics import cohen_kappa_score

def RubricDictOfList(original_rubric_dict):

  # This function takes a Pandas column of dictionaries like:
  # Sentence 0 -> {"meaning_preservation": "3", 'grammar': '4', ...}
  # Sentence 1 -> {"meaning_preservation": "2", 'grammar': '5', ...}
  # and return a dictionary of the type
  # {'MP': [...], 'MSE': [...],...}
  # containing a single rubric as key, and as values a list for all the sentences of votes

  # List of all the metrics of the rubric
  final_output_dict = {}
  MP = [] # Meaning Preservation List
  GR = [] # Grammar List
  MSE = [] # Modern Structural Effectiveness List
  CO = [] # Completeness List
  LM = [] # Lexical Modernization List

  for idx, row in original_rubric_dict.items():
    for key, value in row.items():
      if value == '':
        print(f"missing value at {idx}")
      if key == 'meaning_preservation':
        MP.append(value)
      elif key=='grammar':
        GR.append(value)
      elif key=='modern_structural_effectiveness':
        MSE.append(value)
      elif key=='completeness':
        CO.append(value)
      elif key=='lexical_modernization':
        LM.append(value)
      else:
        print(f"no such key: {key} in item: {idx}\n")
  final_output_dict['MP'] = MP
  final_output_dict['GR'] = GR
  final_output_dict['MSE'] = MSE
  final_output_dict['CO'] = CO
  final_output_dict['LM'] = LM
  return final_output_dict

def PrepareData4CohenKappa(path_list, eval_dict):
  
  for path in path_list: 
    model_name = path.split('/')[-1].split('_')[0]
    if 'transformer' in model_name:
      model_name += '_' 
      model_name += path.split('/')[-1].split('_')[1]
      if 'non' in model_name:
        model_name += path.split('/')[-1].split('_')[2]
    eval_dict[model_name] = {}
    model_pd = pd.read_json(path, lines=True)
    llmEval = model_pd['evaluation']
    humanEval = model_pd['human_vote']
    judge_dict = RubricDictOfList(llmEval)
    human_dict = RubricDictOfList(humanEval)
    eval_dict[model_name]['prometheus'] = judge_dict
    eval_dict[model_name]['human'] = human_dict



def CohenKappaComputation(eval_dict, cohen_kappa_dict):
  rubric_names = ['MP', 'GR', 'MSE', 'CO', 'LM']
  for model_name, prometheus_human_rubrics in eval_dict.items():
    cohen_kappa_dict[model_name] = {}
    for rubric in rubric_names:
      cohen_kappa_dict[model_name][rubric] = cohen_kappa_score(prometheus_human_rubrics['prometheus'][rubric], prometheus_human_rubrics['human'][rubric])
  
  with open("cohen_kappa.json", "w") as f:
    json.dump(cohen_kappa_dict, f, indent=2)