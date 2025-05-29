import requests
import time
from typing import Optional

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
            if response.status_code == 429:
                print("\n - Received 429 Too Many Requests. Waiting 5 seconds before retrying...")
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