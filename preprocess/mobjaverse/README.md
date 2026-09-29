# Mobjaverse preprocessing

This pipeline builds the [CAST cache](../README.md) from the
[Mobjaverse dataset on Hugging Face](https://huggingface.co/datasets/duckduckplz/Mobjaverse).

The dataset root passed to the script must contain `render.py`, `src/`,
`datalist/`, and `mobjaverse/<asset-id>/raw_data.npz`. The required source code
and dataset lists are included under `third_party/`; download the
`mobjaverse/` asset directory from Hugging Face. Put it under `third_party/`
or pass a complete Mobjaverse download as the dataset root.

## Requirements

Use Linux with Python 3.10 and an EGL-capable GPU driver. Install the Python
packages from this directory:

```bash
python3 -m pip install -r requirements.txt
```

## Build

Run from `preprocess/mobjaverse`, passing the dataset root and a new working
directory:

```bash
bash build_cache.sh /path/to/Mobjaverse /path/to/new-work-directory
```

The script renders every frame of the listed assets and builds the cache. The included
[`quality_rejections.json`](scripts/quality_rejections.json) determines which
source assets are excluded from the final cache. Intermediate renders are in
`<work-directory>/render/`; the finished cache is in `<work-directory>/cache/`.
Completed asset renders are reused on rerun while `cache/` does not exist.

Set `MOBJ_EGL_DEVICE_ID` to select the GPU and `MOBJ_CACHE_WORKERS` to control
cache-building parallelism.

## License

The Mobjaverse dataset and the source code bundled under `third_party/` are
released by the Mobjaverse authors under ODC-BY. See the
[dataset page](https://huggingface.co/datasets/duckduckplz/Mobjaverse) for its
license terms, and [LICENSE-3RD-PARTY](../../LICENSE-3RD-PARTY) for the attribution.

## Acknowledgments

We thank the authors of [TopoCap](https://huggingface.co/papers/2606.12153)
for releasing [Mobjaverse](https://huggingface.co/datasets/duckduckplz/Mobjaverse)
and its accompanying source code, from which the bundled `third_party/` files
are taken at revision `e635998f906e8a710fb5b8b7accad61548fc311f`. Mobjaverse is
derived from [Objaverse-XL](https://huggingface.co/datasets/allenai/objaverse-xl).
