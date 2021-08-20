
class ConfCreator(object):

    def __init__(self):
        self.conf_template = {
            'use_case': ["Structured_Fodors-Zagats", "Structured_DBLP-GoogleScholar", "Structured_DBLP-ACM",
                         "Structured_Amazon-Google", "Structured_Walmart-Amazon", "Structured_Beer",
                         "Structured_iTunes-Amazon", "Textual_Abt-Buy", "Dirty_iTunes-Amazon", "Dirty_DBLP-ACM",
                         "Dirty_DBLP-GoogleScholar", "Dirty_Walmart-Amazon"],
            'data_type': ['train', 'test', 'valid'],
            'permute': [True, False],
            'model_name': ['bert-base-uncased'],
            'tok': ['attr_pair', 'sent_pair'],
            'size': [None],
            'fine_tune_method': ['simple', None], # ['advanced', 'simple', None],
            'extractor': ['attr_extractor'],
            'tester': ['attr_tester'],
        }
        self.use_case_map = {
            "Structured_Fodors-Zagats": 'S-FZ',
            "Structured_DBLP-GoogleScholar": 'S-DG',
            "Structured_DBLP-ACM": 'S-DA',
            "Structured_Amazon-Google": 'S-AG',
            "Structured_Walmart-Amazon": 'S-WA',
            "Structured_Beer": 'S-BR',
            "Structured_iTunes-Amazon": 'S-IA',
            "Textual_Abt-Buy": 'T-AB',
            "Dirty_iTunes-Amazon": 'D-IA',
            "Dirty_DBLP-ACM": 'D-DA',
            "Dirty_DBLP-GoogleScholar": 'D-DG',
            "Dirty_Walmart-Amazon": 'D-WA'
        }

    def validate_conf(self, conf: dict):
        assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
        assert all([p in self.conf_template for p in conf]), "Wrong data format for parameter 'conf'."
        wrong_format = False
        for (k, v) in conf.items():
            if v not in self.conf_template[k] and v != self.conf_template[k]:
                wrong_format = True
                break
        if wrong_format:
            raise ValueError("Wrong data format for parameter 'conf'.")

        return conf

    def get_confs(self, conf: dict, params: list):
        conf = self.validate_conf(conf)
        assert isinstance(params, list), "Wrong data type for parameter 'params'."
        assert all([isinstance(p, str) for p in params]), "Wrong data type for parameter 'params'."
        assert all([p in self.conf_template for p in params]), "Wrong value for parameter 'params'."

        confs = []
        for param in params:
            for val in self.conf_template[param]:
                out_conf = conf.copy()
                out_conf[param] = val
                confs.append(out_conf)

        return confs

    def get_param_values(self, param: str):
        assert isinstance(param, str), "Wrong data type for parameter 'param'."
        assert param in self.conf_template, "Wrong value for parameter 'param'."

        return self.conf_template[param]
