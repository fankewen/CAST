#!/usr/bin/env bash
# Fetch the frozen backbone sources used by CAST, and the DINOv2 weights.
#
#   bash scripts/setup_backbones.sh              # sources + DINOv2 weights
#   bash scripts/setup_backbones.sh --no-weights # sources only
#
# DINOv3 weights are gated on Hugging Face and are deliberately NOT downloaded
# here; see the backbone section of docs/installation.md.
set -euo pipefail

# Revisions the released checkpoints were trained and verified with.
DINOV2_COMMIT="7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV3_COMMIT="346f38fee679c56a6888f91c51670fae61d364e0"

DINOV2_REPO="https://github.com/facebookresearch/dinov2.git"
DINOV3_REPO="https://github.com/facebookresearch/dinov3.git"
DINOV2_WEIGHTS_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
DINOV2_WEIGHTS_SHA256="d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428"

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
third_party="${repo_root}/third_party"
weights_dir="${third_party}/weights"
fetch_weights=1

usage() {
    sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

for argument in "$@"; do
    case "${argument}" in
        --no-weights) fetch_weights=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: ${argument}" >&2; usage >&2; exit 2 ;;
    esac
done

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

download() {
    if command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$2" "$1"
    else
        echo "need curl or wget to download ${1}" >&2
        return 1
    fi
}

checkout_source() {
    local url="$1" directory="$2" commit="$3"
    if [ -d "${directory}/.git" ]; then
        echo "  $(basename "${directory}"): present"
    else
        echo "  $(basename "${directory}"): cloning"
        git clone --quiet "${url}" "${directory}"
    fi
    git -C "${directory}" checkout --quiet "${commit}"
    echo "    revision $(git -C "${directory}" rev-parse --short HEAD)"
}

mkdir -p "${third_party}" "${weights_dir}"

echo "Backbone sources -> ${third_party}"
checkout_source "${DINOV2_REPO}" "${third_party}/dinov2" "${DINOV2_COMMIT}"
checkout_source "${DINOV3_REPO}" "${third_party}/dinov3" "${DINOV3_COMMIT}"

dinov2_weights="${weights_dir}/dinov2_vitl14_pretrain.pth"
if [ "${fetch_weights}" -eq 1 ]; then
    if [ -f "${dinov2_weights}" ] &&
        [ "$(sha256_of "${dinov2_weights}")" = "${DINOV2_WEIGHTS_SHA256}" ]; then
        echo "Weights: dinov2_vitl14_pretrain.pth already present and verified"
    else
        echo "Weights: downloading dinov2_vitl14_pretrain.pth (~1.2 GB)"
        download "${DINOV2_WEIGHTS_URL}" "${dinov2_weights}.part"
        if [ "$(sha256_of "${dinov2_weights}.part")" != "${DINOV2_WEIGHTS_SHA256}" ]; then
            rm -f "${dinov2_weights}.part"
            echo "sha256 mismatch; the download was removed" >&2
            exit 1
        fi
        mv "${dinov2_weights}.part" "${dinov2_weights}"
        echo "  verified sha256"
    fi
fi
