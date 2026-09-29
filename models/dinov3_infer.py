#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


EXTS = ('.png', '.jpg', '.jpeg')
DINOV3_MODEL_NAME = 'dinov3_vitl16_pretrain_lvd1689m'
PATCH_SIZE = 16
EMBED_DIM = 1024
SAVE_DTYPE = np.float32


class RGBTarSource:
    def __init__(self):
        self.tars = {}

    def close(self):
        for tf in self.tars.values():
            tf.close()
        self.tars.clear()

    def load_rgb(self, tar_path, member_name):
        tar_path = str(tar_path)
        tf = self.tars.get(tar_path)
        if tf is None:
            tf = tarfile.open(tar_path, 'r')
            self.tars[tar_path] = tf
        stem = str(Path(member_name).with_suffix(''))
        candidates = [member_name] + [stem + ext for ext in EXTS if stem + ext != member_name]
        for candidate in candidates:
            try:
                f = tf.extractfile(candidate)
            except KeyError:
                continue
            if f is not None:
                return Image.open(io.BytesIO(f.read())).convert('RGB')
        raise FileNotFoundError(
            f'None of {candidates} found in {tar_path}'
        )


class RGBTarDataset(Dataset):
    def __init__(self, frame_records, indices, size):
        self.frame_records = frame_records
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = build_transform(size)
        self.source = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        if self.source is None:
            self.source = RGBTarSource()
        idx = int(self.indices[item])
        rec = self.frame_records[idx]
        return idx, self.transform(self.source.load_rgb(rec['tar_path'], rec['member_name']))


def build_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_dinov3_vitl16(repo, weights, device):
    repo = str(Path(repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from dinov3.hub.backbones import dinov3_vitl16

    model = dinov3_vitl16(pretrained=True, weights=weights)
    return model.eval().to(device)


def build_rgb_cache_records(cache_dir):
    cache_dir = Path(cache_dir)
    with open(cache_dir / 'meta.json', encoding='utf-8') as f:
        meta = json.load(f)

    total_frames = int(meta['total_frames'])
    records = [None] * total_frames
    for motion_idx, motion in enumerate(meta['motions']):
        offset = int(motion['offset'])
        frames = int(motion['frames'])
        clip_name = motion['clip_name']
        tar_path = cache_dir / motion['image']['tar']
        frame_offset_in_tar = int(motion['image'].get('frame_offset_in_tar', 0))
        image_glob = motion.get('images', {}).get('image_glob', '*.jpg')
        image_ext = Path(image_glob).suffix.lower()
        if image_ext not in EXTS:
            image_ext = '.jpg'
        for frame in range(frames):
            idx = offset + frame
            records[idx] = {
                'skeleton_id': cache_dir.name,
                'motion_index': motion_idx,
                'clip_name': clip_name,
                'frame': frame,
                'tar_path': str(tar_path),
                'frame_offset_in_tar': frame_offset_in_tar + frame,
                'member_name': f'{clip_name}.{frame:06d}{image_ext}',
                'image_id': f'{cache_dir.name}/{clip_name}.{frame:06d}{image_ext}',
            }
    missing = [i for i, rec in enumerate(records) if rec is None]
    if missing:
        raise RuntimeError(f'{cache_dir} has {len(missing)} frames missing from meta motions; first missing index={missing[0]}')
    return meta, records


def build_source_fingerprint(cache_dir, meta):
    """Fingerprint labels and RGB shards so stale features cannot be resumed."""
    cache_dir = Path(cache_dir)
    digest = hashlib.sha256()
    for name in ('meta.json', 'static.npy', 'world_pos.npy'):
        path = cache_dir / name
        stat = path.stat()
        digest.update(f'{name}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())

    tar_paths = sorted({motion['image']['tar'] for motion in meta['motions']})
    for rel_path in tar_paths:
        path = cache_dir / rel_path
        stat = path.stat()
        digest.update(f'{rel_path}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


def validate_resume_metadata(out_dir, expected):
    metadata_path = Path(out_dir) / 'metadata.json'
    if not metadata_path.is_file():
        return
    with metadata_path.open(encoding='utf-8') as f:
        existing = json.load(f)
    old_fingerprint = existing.get('source_fingerprint')
    new_fingerprint = expected['source_fingerprint']
    if old_fingerprint != new_fingerprint:
        raise RuntimeError(
            f'Stale DINO cache at {out_dir}: source fingerprint changed '
            f'({old_fingerprint} != {new_fingerprint}). Remove the output directory '
            'and regenerate all features.'
        )


def open_outputs(out_dir, n, num_patches, dtype, resume):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_path = out_dir / 'patch_tokens.npy'
    cls_path = out_dir / 'cls_tokens.npy'
    done_path = out_dir / 'done.npy'

    mode = 'r+' if resume and patch_path.exists() and cls_path.exists() and done_path.exists() else 'w+'
    patch_tokens = np.lib.format.open_memmap(patch_path, mode=mode, dtype=dtype, shape=(n, num_patches, EMBED_DIM))
    cls_tokens = np.lib.format.open_memmap(cls_path, mode=mode, dtype=dtype, shape=(n, EMBED_DIM))
    if patch_tokens.dtype != dtype or cls_tokens.dtype != dtype:
        raise RuntimeError(
            f'Existing DINOv3 memmaps in {out_dir} are not float32 '
            f'(patch_tokens={patch_tokens.dtype}, cls_tokens={cls_tokens.dtype}). '
            'Delete/regenerate the output files or run without --resume.'
        )
    if patch_tokens.shape != (n, num_patches, EMBED_DIM) or cls_tokens.shape != (n, EMBED_DIM):
        raise RuntimeError(
            f'Existing DINOv3 memmap shape mismatch in {out_dir}: '
            f'patch_tokens={patch_tokens.shape}, cls_tokens={cls_tokens.shape}, expected n={n}, patches={num_patches}.'
        )
    done = np.lib.format.open_memmap(done_path, mode=mode, dtype=np.bool_, shape=(n,))
    if done.shape != (n,):
        raise RuntimeError(f'Existing done.npy shape mismatch in {out_dir}: {done.shape}, expected {(n,)}')
    if mode == 'w+':
        done[:] = False
    return mode, patch_tokens, cls_tokens, done


def save_metadata(out_dir, metadata, image_ids, mode):
    out_dir = Path(out_dir)
    if mode != 'w+':
        return
    np.save(out_dir / 'image_paths.npy', np.asarray(image_ids))
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def extract_to_memmaps(args, out_dir, n, image_ids, make_iterator, metadata):
    num_patches = (args.size // PATCH_SIZE) ** 2
    if args.dtype != 'float32':
        print('--dtype is deprecated for DINOv3 feature export; saving float32.', flush=True)
    dtype = SAVE_DTYPE

    mode, patch_tokens, cls_tokens, done = open_outputs(out_dir, n, num_patches, dtype, args.resume)
    save_metadata(out_dir, metadata, image_ids, mode)
    if args.init_only:
        print(f'Initialized {out_dir} for {n} images', flush=True)
        return

    model = load_dinov3_vitl16(args.repo, args.weights, args.device)
    pending = np.flatnonzero(~done)
    if args.num_shards > 1:
        pending = pending[args.shard_index::args.num_shards]
        print(f'Shard {args.shard_index}/{args.num_shards}: {len(pending)} pending images for {out_dir}', flush=True)

    iterator = make_iterator(pending)
    for batch_idx, imgs in iterator:
        if torch.is_tensor(batch_idx):
            batch_idx_np = batch_idx.numpy()
        else:
            batch_idx_np = np.asarray(batch_idx, dtype=np.int64)
        x = imgs.to(args.device, non_blocking=True)
        with torch.inference_mode():
            feats = model.forward_features(x)
            patches = feats['x_norm_patchtokens'].detach().cpu().numpy().astype(dtype, copy=False)
            cls = feats['x_norm_clstoken'].detach().cpu().numpy().astype(dtype, copy=False)
        patch_tokens[batch_idx_np] = patches
        cls_tokens[batch_idx_np] = cls
        done[batch_idx_np] = True
        patch_tokens.flush(); cls_tokens.flush(); done.flush()
        print(f'[{done.sum()}/{n}] saved {out_dir}', flush=True)


def run_rgb_cache_mode(args):
    dataset_root = Path(args.dataset_root)
    if args.skeleton_id:
        cache_dirs = [dataset_root / skeleton_id for skeleton_id in args.skeleton_id]
    else:
        cache_dirs = sorted(p for p in dataset_root.iterdir() if (p / 'meta.json').is_file())
    if not cache_dirs:
        raise FileNotFoundError(f'No cache meta.json files found under {dataset_root}')

    for cache_dir in cache_dirs:
        if not (cache_dir / 'meta.json').is_file():
            raise FileNotFoundError(f'Missing meta.json for {cache_dir}')
        meta, frame_records = build_rgb_cache_records(cache_dir)
        n = len(frame_records)
        image_ids = [rec['image_id'] for rec in frame_records]
        out_dir = Path(args.output_dir) / cache_dir.name

        def make_iterator(pending, records=frame_records):
            if args.num_workers > 0:
                dataset = RGBTarDataset(records, pending, args.size)
                loader_kwargs = dict(
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=args.device.startswith('cuda'),
                    persistent_workers=True,
                )
                loader_kwargs['prefetch_factor'] = args.prefetch_factor
                return DataLoader(dataset, **loader_kwargs)

            transform = build_transform(args.size)
            source = RGBTarSource()

            def serial_iterator():
                try:
                    for offset in range(0, len(pending), args.batch_size):
                        batch_idx = pending[offset:offset + args.batch_size]
                        imgs = []
                        for idx in batch_idx:
                            rec = records[int(idx)]
                            imgs.append(transform(source.load_rgb(rec['tar_path'], rec['member_name'])))
                        yield torch.as_tensor(batch_idx, dtype=torch.long), torch.stack(imgs, dim=0)
                finally:
                    source.close()

            return serial_iterator()

        metadata = {
            'format': 'dinov3_memmap',
            'source_format': 'rgb_cache',
            'model': DINOV3_MODEL_NAME,
            'size': args.size,
            'patch_size': PATCH_SIZE,
            'num_patches': (args.size // PATCH_SIZE) ** 2,
            'embed_dim': EMBED_DIM,
            'dtype': 'float32',
            'num_images': n,
            'dataset_root': str(dataset_root),
            'skeleton_id': cache_dir.name,
            'cache_total_frames': int(meta['total_frames']),
            'image_shard_count': int(meta.get('image_shard_count', 0)),
            'motions': len(meta.get('motions', [])),
            'skeleton_hash': meta.get('skeleton_hash'),
            'source_fingerprint': build_source_fingerprint(cache_dir, meta),
        }
        if args.resume:
            validate_resume_metadata(out_dir, metadata)
        extract_to_memmaps(args, out_dir, n, image_ids, make_iterator, metadata)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess RGB cache tar entries into DINOv3 ViT-L/16 memmaps.'
    )
    parser.add_argument('--dataset-root', required=True, help='Cache root containing <skeleton_id>/meta.json and images/shard-*.tar')
    parser.add_argument('--skeleton-id', action='append', help='Skeleton id under --dataset-root. Can be repeated; defaults to all caches.')
    parser.add_argument('--output-dir', required=True, help='Output dir, preferably under /dev/shm')
    parser.add_argument('--repo', default='/absolute/path/to/dinov3')
    parser.add_argument('--weights', default='/absolute/path/to/dinov3_weights/dinov3-vit-large/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float32', help='Deprecated; DINOv3 features are always saved as float32.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--init-only', action='store_true')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    args = parser.parse_args()

    if args.size % PATCH_SIZE != 0:
        raise ValueError(f'--size must be divisible by {PATCH_SIZE}')
    if args.num_shards < 1:
        raise ValueError('--num-shards must be >= 1')
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError('--shard-index must be in [0, num_shards)')
    run_rgb_cache_mode(args)


if __name__ == '__main__':
    main()
