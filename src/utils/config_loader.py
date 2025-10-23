import yaml
import os

def load_config(config_path='config/config.yaml'):
    """
    加载并解析 YAML 配置文件。

    Args:
        config_path (str): YAML 配置文件的路径。

    Returns:
        dict: 解析后的配置字典。
    """
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration file loaded successfully from '{config_path}'.")
        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error: Configuration file not found at '{config_path}'. "
            "Please ensure the file exists and the path is correct."
        )
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file '{config_path}': {e}")

config = load_config()