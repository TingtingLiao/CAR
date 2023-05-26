import os
import argparse
import torch
from lib.data.TestDataset import TestDataset
from lib.common.config import load_config
from .train import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--gpu_device', type=int, default=0)
    parser.add_argument('-in_dir', '--in_dir', type=str, default="./examples/net")
    parser.add_argument('-cfg', '--config', type=str, default="configs/arwild.yaml")

    args = parser.parse_args()

    # cfg read and merge
    cfg = load_config(args.config, 'configs/default.yaml')
    cfg.training.merge_from_list(['gpus', [args.gpu_device]])
    cfg.freeze()

    dataset = TestDataset(cfg, image_dir=f'{args.in_dir}/image', out_dir=args.in_dir)
    print('DATA LEN ', len(dataset))

    model_name = args.config.split('/')[-1][:-5]
    save_dir = os.path.join(args.in_dir, 'results', model_name)
    os.makedirs(save_dir, exist_ok=True)

    trainer = Trainer(cfg)

    for i in range(len(dataset)):
        data = dataset.get_item(i)

        save_obj_path = os.path.join(save_dir, data['im_name'] + '.obj')

        if not os.path.exists(save_obj_path):
            with torch.no_grad():
                pr_canon_mesh, pr_posed_mesh = trainer.visualize(data, save_obj_path)
            print('saving to ', save_obj_path)

