import yaml

url_config = {
        'vgg_seq2seq':'vgg_seq2seq.yml',
        'base':'base.yml',
        }

class Cfg(dict):
    def __init__(self, config_dict):
        super(Cfg, self).__init__(**config_dict)
        self.__dict__ = self

    @staticmethod
    def load_config_from_name(folder, name):
        path_base = folder + url_config['base']
        path_new_config = folder + url_config[name]
    
        with open(path_base, 'r') as stream:
            base_config = yaml.safe_load(stream)
        with open(path_new_config, 'r') as stream:
            new_config = yaml.safe_load(stream)
            
        base_config.update(new_config)
        return Cfg(base_config)
        
       