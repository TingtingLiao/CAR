from yacs.config import CfgNode as CN


def get_cfg_defaults(yaml_file='../configs/default.yaml'):
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(yaml_file)
    return cfg.clone()


# General config
def load_config(path, default_path=None):
    cfg = CN(new_allowed=True)
    if default_path is not None:
        cfg.merge_from_file(default_path)
    cfg.merge_from_file(path)

    return cfg
