import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
import yaml
import torch
import os
from utils.datasets import create_dataloader
from utils.general import check_dataset, colorstr
import argparse
from tqdm import tqdm

def kmeans_anchors(data='data.yaml', n_anchors=4, img_size=640, batch_size=16, gen_num=1000):
    """
    Generate 4 anchors for 2 feature layers (2 anchors per layer) using K-means clustering
    
    Args:
        data (str): Path to dataset config file
        n_anchors (int): Fixed to 4 (2 per feature layer)
        img_size (int): Input image size
        batch_size (int): Batch size for dataloader
        gen_num (int): Max boxes to use for clustering
    
    Returns:
        anchors (np.array): Array of 4 anchors in format [w, h] normalized to img_size
    """
    # Validate input
    if n_anchors != 4:
        print(f"⚠️ Overriding n_anchors to 4 (2 anchors per feature layer)")
        n_anchors = 4

    # Path handling
    data = str(Path(data).absolute())
    if not os.path.exists(data):
        raise FileNotFoundError(f"Dataset config not found: {data}")

    # Load dataset config
    with open(data) as f:
        data_dict = yaml.safe_load(f)
    check_dataset(data_dict)

    # Create a minimal opt object required by create_dataloader
    class Opt:
        def __init__(self):
            self.single_cls = False  # Default to multi-class
            self.rect = True        # Rectangular training
            self.augment = False    # No augmentation for anchor generation
            self.cache = False      # No image caching
            
    opt = Opt()

    # Create dataloader (aligned with training)
    dataloader, _ = create_dataloader(
        path=data_dict['train'],
        imgsz=img_size,
        batch_size=batch_size,
        stride=max(round(img_size / 640 * 32), 32),
        opt=opt,  # Pass the minimal opt object
        augment=False,
        cache=False,
        pad=0.5,
        rect=True,
        workers=8,
        prefix=colorstr('Anchors: ')
    )

    # Collect box dimensions
    shapes = []
    for _, targets, _, _ in tqdm(dataloader, desc='Processing boxes'):
        if targets.shape[0]:
            wh = targets[:, 3:5] * img_size  # Normalized to pixels
            valid = (wh > 5).all(1)  # Filter tiny boxes (5px threshold)
            shapes.append(wh[valid])
    
    if not shapes:
        raise ValueError("No valid boxes found!")

    shapes = torch.cat(shapes).float().numpy()
    
    # Downsample if needed
    if shapes.shape[0] > gen_num:
        shapes = shapes[np.random.choice(shapes.shape[0], gen_num, replace=False)]

    # K-means clustering with 2 separate groups
    areas = shapes.prod(1)
    split_idx = np.percentile(areas, 50)  # Median split
    
    # Cluster small and large boxes separately
    small_boxes = shapes[areas <= split_idx]
    large_boxes = shapes[areas > split_idx]
    
    anchors = []
    for boxes in [small_boxes, large_boxes]:
        if len(boxes) > 1:
            kmeans = KMeans(n_clusters=2, n_init=10).fit(boxes)
            anchors.extend(kmeans.cluster_centers_)
        else:
            anchors.extend([[img_size*0.1, img_size*0.1], [img_size*0.2, img_size*0.2]])

    anchors = np.array(anchors)
    anchors = anchors[np.argsort(anchors.prod(1))] / img_size  # Sort and normalize

    # Organize for 2 feature layers
    print("\nOptimized Anchors for 2 Feature Layers:")
    print(f"Layer 1 (Higher resolution): {anchors[:2].round(4).tolist()}")
    print(f"Layer 2 (Lower resolution): {anchors[2:].round(4).tolist()}")

    # Save to hyp.yaml
    hyp_path = Path('data/hyp.finetune.yaml')
    if hyp_path.exists():
        with open(hyp_path) as f:
            hyp = yaml.safe_load(f) or {}
        hyp['anchors'] = anchors.round(4).tolist()
        with open(hyp_path, 'w') as f:
            yaml.safe_dump(hyp, f)
        print(f"\nSaved to {hyp_path}")

    return anchors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.yaml', help='dataset.yaml path')
    parser.add_argument('--img-size', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    opt = parser.parse_args()

    kmeans_anchors(
        data=opt.data,
        img_size=opt.img_size,
        batch_size=opt.batch_size
    )