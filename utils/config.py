import yaml

def load_config(config_path):
  
# This function loads a YAML configuration file from the specified path
# Args: - config_path: <str> the path to the YAML configuration file
# Output: - config: <dict> the configuration loaded from the YAML file


  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  return config