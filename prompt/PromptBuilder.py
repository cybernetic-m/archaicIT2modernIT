class PromptBuilder:

  # This class takes a sentence in input and transform it into a prompt to send to an LLM.
  # Args:
  #       - mode: "zero-shot", "few-shot" (set the mode to write [few-shot] or not [zero-shot] examples of translation in the prompt)
  #       - examples: a list of tuples (archaic, modern) of examples to write in the prompt for "few-shot" mode
  #       - prompt_template: a tuple of strings (system_prompt, user_prompt)
  #       - k: the number of examples to write in the prompt in case of few-shot
  #       - lang: the language of the prompt ('it' or 'en')

  def __init__(self, prompt_template, mode="zero-shot", k=1, examples=None, lang='it'):

    # Initialize all the args
    self.mode = mode
    self.examples = examples
    self.system_prompt = prompt_template[0]  # the system prompt is the first element of the tuple
    self.user_prompt = prompt_template[1]  # the user prompt is the second element of the tuple
    if self.mode == 'zero-shot':
      self.k = 0  # in zero-shot mode, there are no examples to write in the prompt
    else:
      self.k = k
    self.lang = lang

  def getSystemPrompt(self):
    # This method returns the system prompt
    return self.system_prompt

  def getLang(self):
    # This method returns the language of the prompt
    return self.lang

  def getMode(self):
    # This method returns the mode of the prompt
    return self.mode

  def getK(self):
    # This method returns the number of examples to write in the prompt in case of few-shot
    return self.k

  def build_prompt(self, old_sentence):

    # This method build the prompt template taking the "old_sentence" to be translated

    if self.mode == "zero-shot":
      # in this case substitute the {old_sentence} in the user prompt with the actual "old_sentence"
      # there are no examples in "zero_shot"
      return self.user_prompt.format(old_sentence=old_sentence, examples='')

    elif self.mode == "few-shot":

      # Take the examples list and select the first "k" tuples (archaic,modern)
      k_examples = self.examples[0:self.k]

      if self.lang == 'it':
        # build the first part of the {examples} part to subs in the prompt_template
        examples_text = "Ti fornisco alcune traduzioni d'esempio: \n\n"
        for archaic, modern in k_examples:
          # iterate over all the (archaic, modern) pairs and add to the examples_text
          examples_text += f"Antico: '{archaic}'\nModerno: '{modern}'\n\n"

      # The same of previous but for english text
      elif self.lang == 'en':
        examples_text = "These are some examples of translations: \n\n"
        for archaic, modern in k_examples:
          examples_text += f"Archaic: '{archaic}'\nModern: '{modern}'\n\n"

      return self.user_prompt.format(old_sentence=old_sentence,
                                     examples=examples_text.strip())  # subs also {examples} in the prompt_template with examples_text

    else:
      raise ValueError(f'Mode {self.mode} is not valid. Choose "zero-shot" or "few-shot" instead!')

  def build_judge_prompt(self, tournament=True, old_sentence="", A="", B="", translation=""):

    if tournament:
      return self.user_prompt.format(old_sentence=old_sentence, examples='')

    else:
      return self.user_prompt.format(old_sentence=old_sentence, translation=translation)