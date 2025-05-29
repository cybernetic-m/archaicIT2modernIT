import requests
import time
from typing import Optional
import pandas as pd


def clean_reasoning(text):

# This function takes a text as input and return the text after the marker <\think>, excluding all the reasoning 
# Args: - text: <str> the original text
# Output : - text: <str> the modified text

  end_marker = "</think>" # the marker that conclude the reasoning

  end_index = text.find(end_marker) # returns the index (position) of the first character of </think> if found, or -1 if it’s not found
  if end_index != -1:
      # Extract the text after the end marker
      cleaned_text = text[end_index + len(end_marker):] # from the <\think> to the end (end_index + len(end_marker) because end_index is the first char of the marker)
      return cleaned_text.strip() # strip remove all whitespaces, newlines and tabs
  else:
     return text
  

def call_translation_api(api_key, model_name, system_prompt_template, user_prompt_template, temperature) -> Optional[str]:

# This function sends a Prompt to a Groq-hosted API waiting for the response (the translated sentence)
# Args: - api_key: the Groq API key you need to authorization
#       - model_name: the name of the LLM model (ex. "llama-3.1-8b-instant")
#       - system_prompt_template: <str> it is a prompt containing the instructions for the model (ex. "You are a translator...")
#       - user_prompt_template: <str> it is a prompt containing the sentence to translate
#       - temperature: <float> it is a float number to set the temperature of the model
# Output: - translation: <str> the translated sentence returned by the model, or None if an error occurred

    url = "https://api.groq.com/openai/v1/chat/completions" # this is the url of the Groq API (same structure of OpenAI message!)
    
    # The header of the message contain the "Content-Type" (it say that the message structure will be json, a dict) and the "Authorization"
    # It is a string "Bearer gsk...." with the api_key in a Bearer Token type (Bearer means that the authorization is give with the api_key directly after)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # This is the message content itself. It contain the "model_name", its "temperature"
    # the messages contain two dictionaries: the first one is the system prompt (the instructions for the model, like "You are a translator...") 
    # and the second one is the user prompt (the input sentence to translate)
    data = {
        "model": model_name,
        "temperature": temperature,
        "messages": [
            {
                "role":"system",
                "content": system_prompt_template
            },
            {
                "role": "user",
                "content": user_prompt_template
            }
        ]
    }

    while True:
        try:
            response = requests.post(url, headers=headers, json=data) # send the request to the Groq API
            # Check for rate limiting (HTTP 429), wait 5 seconds and retry
            if response.status_code in [429,500]:
                print(f"Received {response.status_code}. Retrying in 5s...")                
                time.sleep(5)
                continue  # Retry after wait
            
            response.raise_for_status() # Raise an exception for other HTTP errors like 400 or 500 (if one occurred)
            translation = response.json() # return the response in json format (a dict)
            
            # Translation is a dict with 'id' (a unique identifier for the request),  'created' (the timestamp of the request) ... 
            # inside 'choices' there are different generated responses in general, we take the first one
            # inside 'choices there is the 'message' (translation) with the 'role' (user or assistant) and the 'content' (the translated sentence with the reasoning)
            return translation["choices"][0]["message"]["content"].strip()
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed for: '{user_prompt_template}'\nError: {e}")
            return None
        

def LLM_translation(api_key, input_file, model_name, prompt_builder, temperature, req_per_min, save_path):

# This function takes a pandas dataset in input and trannslate all the sentences saving a new dataset with the translations
# Args: - api_key: <str> the Groq API key you need to authorization
#       - input_file: <str> the path to the csv file containing the dataset to translate
#       - model_name: <str> the name of the LLM model (ex. "llama-3.1-8b-instant") 
#       - prompt_builder: <PromptBuilder> the PromptBuilder object to build the prompts
#       - temperature: <float> the temperature of the model (0.0 for deterministic output, 1.0 for more creative output)
#       - req_per_min: <int> the number of API requests per minute to respect the rate limit
#       - save_path: <str> the path where to save the translated dataset
#
# Output: - None, but the translated dataset is saved in a new csv file with the translations

    df = pd.read_csv(input_file) # Read the input file (CSV format) transforming into a pandas DataFrame
    
    seconds_per_request = 60 / req_per_min # calculate the seconds to wait between requests

    if 'Translation' not in df.columns:
        df['Translation'] = None # Add a new column 'Translation' to the DataFrame if it does not exist with None values

    last_request_time = 0 # Initialize the last request time to 0 at start

    # Iterate over each row in the DataFrame (idx is the index, row is the row data)
    for idx, row in df.iterrows():
        # Skip rows that already have a translation
        # If the 'Translation' column is not NaN, (or None), it means that has the translation, then skip the row
        if pd.notna(row.get('Translation')):
            continue  # Skip current iteration for sentence already translated

        sentence = row['Sentence'] # takes the sentence to translate from the 'Sentence' column
        if not sentence or pd.isna(sentence):
            continue # Skip empty or NaN sentences

        # Enforce rate limit: measure the current time and check if we need to sleep
        elapsed = time.time() - last_request_time # Time from the last request
        if elapsed < seconds_per_request:
            # If the elapsed time is less than the seconds per request, we need to sleep
            sleep_time = seconds_per_request - elapsed # Calculate how much time to sleep to respect the rate limit
            time.sleep(sleep_time)

        # Build the prompt using the PromptBuilder
        system_prompt_template = prompt_builder.getSystemPrompt() # Get the system prompt from the PromptBuilder    
        user_prompt_template = prompt_builder.build_prompt(sentence) # Build the user prompt using the PromptBuilder with the sentence to translate
        lang = prompt_builder.getLang() # Get the language of the prompt from the PromptBuilder
        mode = prompt_builder.getMode() # Get the mode of the prompt from the PromptBuilder
        k = prompt_builder.getK() # Get the number of examples to write in the prompt in case of few-shot from the PromptBuilder

        # Translate the sentence using the Groq API and cleaning the reasoning
        translation = clean_reasoning(call_translation_api(api_key=api_key, 
                                                           model_name=model_name, 
                                                           system_prompt_template = system_prompt_template,
                                                           user_prompt_template = user_prompt_template,
                                                           temperature=temperature
                                                           ))
        last_request_time = time.time() # Measure the time of the last request

        # save the translation in the 'Translation' column of the DataFrame at row index = idx
        df.at[idx, 'Translation'] = translation 
        print(f"Translated [{idx+1}/{len(df)}]: {sentence} -> {translation}")

    # Name as "CaponataLovers-hw2_transl-{model_name}_{mode}_{lang}.jsonl"
    # If k is 0, it means that we are in zero-shot mode, so we save the name do not have k
    if k == 0:
        output_name = input_file.split('/')[-1].replace('dataset.csv', f'CaponataLovers-hw2_transl-{model_name}_{mode}_{lang}_temp-{temperature}.jsonl')
    else:
        output_name = input_file.split('/')[-1].replace('dataset.csv', f'CaponataLovers-hw2_transl-{model_name}_{mode}_k-{k}_{lang}_temp-{temperature}.jsonl')
    output_file = save_path + '/' + output_name # Add the save path to the output file name
    df.to_json(output_file, orient="records", lines=True, force_ascii=False) # Save the DataFrame to a JSONL file 
    print(f"Translated dataset saved to {output_file}")