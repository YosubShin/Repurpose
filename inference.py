from dataset.RepurposeClip import RepurposeClipTest
from dataset.RepurposeClip import collate_fn_test
from models.MMCTransformer import MMCTransformer
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import random
from tqdm import tqdm
from utils.metrics import *

def load_config(config_file):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    return config

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    # load config
    cfg = load_config(args.config_path)

    test_dataset = RepurposeClipTest(**cfg['test_dataset'])

    model = MMCTransformer(**cfg['model']).to(device='cuda')

    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_test, num_workers=24)

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    with torch.no_grad():
        total_tIoU = []
        for batch in tqdm(test_data_loader):
            batch['visual_feats'] = batch['visual_feats'].to(device='cuda')
            batch['audio_feats'] = batch['audio_feats'].to(device='cuda')
            batch['text_feats'] = batch['text_feats'].to(device='cuda')
            batch['masks'] = batch['masks'].to(device='cuda')
            preds = model.inference_(batch, cfg['test_cfg'])
            for i in range(len(preds)):
                thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
                precision_per_threshold = calculate_tiou(batch['gt_segments'][i], preds[i]['segments'].tolist(), thresholds)
                total_tIoU.append(precision_per_threshold)
        tIoU = {}
        for threshold in thresholds:
            tIoU[threshold] = sum([item[threshold] for item in total_tIoU]) / len(total_tIoU)

        average = sum(item for item in tIoU.values()) / len(tIoU)
        print(tIoU)
        print(f"average tIoU: {average}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    main(args)