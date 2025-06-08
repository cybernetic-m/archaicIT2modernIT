# archaicIT2modernIT
The task of this homework consists in translating a dataset from archaic italian to modern italian using LLMs and evaluating them using prometheus. For this task we compared 4 models: deepseek-llama, gemma2, NLLB-200 Transformer 600M non fine tuned and fine tuned in the translation task.
They are evaluated using rubric scores on five different metrics: Meaning Preservation, Grammar, Modern Structural Effectiveness, Completeness and Lexical Modernization.

# ⚠️TAs Instructions⚠️

1. **Clone the repository**:  
 ```sh 
git clone "https://github.com/cybernetic-m/archaicIT2modernIT.git"
 ```

2. **Make a groq api key**
It is needed to run the LLMs used for the translations.

3. **Run the LLM based approach**
You can try to re-translate the dataset using the zero-shot or the few-shot by opening the LLM based approach in the notebook.

4. **Run the transformer based approach**
You can try the fine tuning of the transformer and the translation using the non fine tuned and fine tuned transformers in the transformer based aproach section.

<img src="./images/sft.png" alt="transformer" width="500" height = "400" />

6. **Run the LLM as a judge**
You can try our tournament selection and absolute evaluations in the LLM as a judge section.


# ☁️ TAs GDrive Shared Folder ☁️

On the [Caponata_Lovers_hw2_shared_folder ](https://drive.google.com/drive/folders/1an6QsdK0kBZE63KZJgOVvfnzcqpunQCD?usp=drive_link)  you can see different folders where we uploaded the files of the translations, we actually made also the translations using english prompts but we cut them for time reason and GPU limits of colab for the prometheus approach.
- "*fewShot1_temp0_it*": this folder contains the translations of deepseek-lama and gemma with 1-shot, temperature 0, language of the prompt in italian.
- "*fewShot2_temp0_it*": this folder contains the translations of deepseek-lama and gemma with 2-shot and temperature 0, language of the prompt in italian.
- "*fewShot3_temp0_it*": this folder contains the translations of deepseek-lama and gemma with 3-shot and temperature 0, language of the prompt in italian.
- "*fewShot4_temp0_it*": this folder contains the translations of deepseek-lama and gemma with 4-shot and temperature 0, language of the prompt in italian.
- "*fewShot5_temp0_it*": this folder contains the translations of deepseek-lama and gemma with 5-shot and temperature 0, language of the prompt in italian.
- "*zeroShot_temp0_it*": this folder contains the translations of deepseek-lama and gemma with 0-shot and temperature 0, language of the prompt in italian.
- "*transformers*": this folder contains the translations of the transformer fine tuned and non fine tuned.


# Repository Structure





