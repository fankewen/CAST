"""Allowlisted source release; never zip a live deployment directory recursively.

Only approved text formats/locations are included. External repos, runtime state,
weights, credentials and binaries are deliberately excluded. Optional private
token file (one literal token per line) extends the scan, never copies its contents.
This is a packaging guard, not a complete secret-detection or license audit.
"""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[3]
REMOTE_PREFIX = "blender/remote"
ROOT_FILES = {
    'Dockerfile.dockerignore', '.env.example', '.gitignore', 'Dockerfile', 'LICENSE',
    'README.md', 'THIRD_PARTY_NOTICES.md', 'compose.yaml',
    'requirements.txt', 'constraints.txt', 'run_service.sh',
    'settings.py', 'model_selection.py', 'service.py', 'ordered_pipeline.py',
    'origin_http.py', 'run_origin.py',
    'person_detector.py', 'human_tracking.py', 'full_step_graph.py', 'live_history.py',
}
ROOT_FILES = {REMOTE_PREFIX + '/' + name for name in ROOT_FILES} | {
    'LICENSE', 'LICENSE-3RD-PARTY', 'blender/rig_adapter.py',
    'blender/cast_live.py', 'blender/cast_engine_server.py',
    'blender/setup_panel_scene.py', 'blender/build_addon.py', 'blender/README.md',
}
SOURCE_DIRS = {REMOTE_PREFIX + '/' + name for name in
               ('admin', 'client', 'configs', 'docs', 'licenses', 'tests', 'tools')}
SOURCE_DIRS |= {'models', 'data', 'inference', 'utils', 'visualization', 'ops'}
SUFFIXES = {'.py', '.cu', '.cpp', '.h', '.cuh', '.sh', '.html', '.css', '.js', '.md', '.txt', '.yaml', '.json'}
SKIP_DIRS = {'__pycache__', '.git', '.venv', 'build', 'dist', 'node_modules', 'var', 'assets'}
BAD_SUFFIXES = {'.pyc', '.so', '.o', '.a', '.pt', '.pth', '.npy', '.npz', '.fbx', '.blend',
                '.png', '.jpg', '.jpeg', '.mp4', '.sqlite3', '.log', '.jsonl', '.zip', '.tar'}
REQUIRED = {'LICENSE', 'LICENSE-3RD-PARTY', 'blender/rig_adapter.py',
            'blender/remote/README.md', 'blender/remote/docs/VALIDATION.md',
            'blender/remote/third_party/dependencies.json',
            'blender/remote/licenses/PyTorch3D-BSD.txt',
            'blender/remote/tools/install.sh', 'blender/remote/tools/prepare_checkpoint.py',
            'blender/remote/configs/CAST_B.yaml', 'blender/remote/configs/CAST_L.yaml',
            'models/base_model/model.py', 'models/encoder/model.py',
            'models/decoder/model.py', 'models/grace/observability_rotation_correction.py',
            'inference/deploy_realtime_video2motion.py', 'ops/setup.py'}


def candidate_files(root=ROOT):
    selected = []
    for name in sorted(ROOT_FILES):
        p = root/name
        if p.is_symlink() or any(parent.is_symlink() for parent in p.parents if parent != root.parent):
            raise ValueError('Symlink source: ' + name)
        if p.is_file():
            selected.append(p)
    for dirname in sorted(SOURCE_DIRS):
        folder = root/dirname
        if not folder.exists():
            continue
        if folder.is_symlink():
            raise ValueError('Symlink source directory: ' + dirname)
        def walk(directory):
            for p in sorted(directory.iterdir()):
                if p.name in SKIP_DIRS or p.name.endswith('.egg-info'):
                    continue
                if p.is_symlink():
                    raise ValueError('Symlink in source tree: ' + str(p.relative_to(root)))
                if p.is_dir():
                    walk(p)
                elif p.suffix in BAD_SUFFIXES:
                    # Locally built extension files are not source artifacts.
                    continue
                elif p.suffix in SUFFIXES and p.name != '.env':
                    selected.append(p)
                else:
                    raise ValueError('Unapproved source artifact: ' + str(p.relative_to(root)))
        walk(folder)
    dep = root/REMOTE_PREFIX/'third_party/dependencies.json'
    if dep.is_symlink() or dep.parent.is_symlink():
        raise ValueError('External dependency manifest must not be a symlink')
    if dep.is_file():
        selected.append(dep)
    missing = REQUIRED - {p.relative_to(root).as_posix() for p in selected}
    if missing:
        raise ValueError('Required files missing: ' + ', '.join(sorted(missing)))
    return sorted(selected)


def check_text(name, raw, private_tokens=()):
    if b'\0' in raw:
        raise ValueError('Binary/NUL data in text source: ' + name)
    content = raw.decode('utf-8-sig')
    for token in private_tokens:
        if token and token in content:
            raise ValueError('Private token found (value redacted): ' + name)
    # Any non-loopback concrete IP in public source deserves manual review.
    for value in re.findall(r'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])', content):
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            continue
        if not address.is_loopback and not address.is_unspecified:
            # Numeric dotted package versions are not network destinations.
            if re.search(r'(?:==|version[=: ]+)["\x27]?' + re.escape(value), content, re.I):
                continue
            raise ValueError('Concrete network address needs review: ' + name)
    if re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', content):
        raise ValueError('Private key found: ' + name)
    if re.search(r'https?://[^/\s:@]+:[^/\s@]+@', content):
        raise ValueError('Credential-bearing URL found: ' + name)
    if re.search(r'(?m)^\s*(?:export[ \t]+)?(?:CAST_API_KEY|CAST_ADMIN_PASSWORD)[ \t]*=[ \t]*[^\s#]', content):
        raise ValueError('Literal service secret assignment: ' + name)
    return content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private-tokens-file', type=Path,
                        help='Optional private literals to reject; not included in archive')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error('Write archive outside the source directory')
    if output.exists() or output.with_suffix(output.suffix+'.sha256').exists():
        parser.error('Output already exists; choose another name')
    tokens = args.private_tokens_file.read_text().splitlines() if args.private_tokens_file else ()
    records = []
    for path in candidate_files():
        name = path.relative_to(ROOT).as_posix()
        raw = path.read_bytes()
        check_text(name, raw, tokens)
        records.append((name, raw))
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in records}
    manifest = {'format_version': 2, 'root': 'cast-deploy', 'sha256': hashes,
                'excluded': ['model weights', 'customer data', 'credentials', 'runtime state',
                             'compiled binaries', 'external repositories', 'git history'],
                'note': 'Manifest excludes itself; release ZIP has a separate SHA256 sidecar.'}
    records.append(('SOURCE_MANIFEST.json', (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(records):
            info = zipfile.ZipInfo('cast-deploy/'+name, date_time=(2026, 10, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if name.endswith('.sh') else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, raw)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix+'.sha256').write_text(digest+'  '+output.name+'\n')
    print(json.dumps({'archive': str(output), 'files': len(records), 'bytes': output.stat().st_size,
                      'sha256': digest}, indent=2))


if __name__ == '__main__':
    main()
