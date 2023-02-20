import configparser


def save_baseline(baseline_name:str, value:str):
    config = configparser.ConfigParser()
    config.read("config.ini")

    config.set('BASELINES', baseline_name, value)
    
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

def get_baseline_uri(baseline_name):
    config = configparser.ConfigParser()
    config.read("config.ini")

    return config['BASELINES'][baseline_name]


def get_aws_profile_name():
    config = configparser.ConfigParser()
    config.read("config.ini")

    return config['aws-settings']['profileName']


def get_aws_iam_role():
    config = configparser.ConfigParser()
    config.read("config.ini")

    return config['aws-settings']['executionRole']


def save_dataset(dateset_name:str, value:str):
    config = configparser.ConfigParser()
    config.read("config.ini")

    config.set('DATASETS', dateset_name, value)
    
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def get_dataset_uri(dataset_name):
    config = configparser.ConfigParser()
    config.read("config.ini")

    return config['DATASETS'][dataset_name]


def save_trial_name(scenario_name:str, value:str):
    config = configparser.ConfigParser()
    config.read("config.ini")

    config.set('EXPERIMENT_TRIALS', scenario_name, value)
    
    with open('config.ini', 'w') as configfile:
        config.write(configfile)


def get_trial_name(scenario_name):
    config = configparser.ConfigParser()
    config.read("config.ini")

    return config['EXPERIMENT_TRIALS'][scenario_name]