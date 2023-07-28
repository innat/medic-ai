
import yaml

def yaml_load(file='default.yaml', append_filename=False):
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read() 
        return {
            **yaml.safe_load(s), 
            'yaml_file': str(file)
        } if append_filename else yaml.safe_load(s)