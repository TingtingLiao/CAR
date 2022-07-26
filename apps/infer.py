import os
from lib.data.TestDataset import TestDataset
import argparse

if __name__ == '__main__':
    # loading cfg file
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu_device', type=int, default=2)
    parser.add_argument('-in_dir', '--in_dir', type=str, default="./examples")
    parser.add_argument('-out_dir', '--out_dir', type=str, default="./examples/results")
    parser.add_argument('-cfg', '--config', type=str, default="configs/geofeat/sdf-bbox.yaml")

    args = parser.parse_args()
    from lib.common.config import load_config
    from .train import Trainer

    # cfg read and merge
    cfg = load_config(args.config, 'configs/default.yaml')
    cfg.training.merge_from_list(['gpus', [args.gpu_device]])
    cfg.freeze()

    dataset = TestDataset(cfg, data_dir=args.in_dir)
    print('DATA LEN ', len(dataset))

    trainer = Trainer(cfg)
    for i in range(len(dataset)):
        data = dataset.get_item(i)
        save_obj_path = os.path.join(args.out_dir, data['im_name']+'.obj')
        if os.path.exists(save_obj_path):
            continue
        pr_canon_mesh, pr_posed_mesh = trainer.visualize(data, save_obj_path)
        pr_posed_mesh.export(save_obj_path[:-4]+'_posed.obj')
        print('saving to ', save_obj_path)

