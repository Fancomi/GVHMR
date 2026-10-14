"""Fetch a CUDA-compatible MSVC toolset from Microsoft's VS 2022 feed.

Files stay in .build-tools; the installed Visual Studio is not modified.
The existing Windows SDK and VS developer shell are still required.
"""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path


def main():
    destination = Path(__file__).resolve().parents[1] / '.build-tools' / 'msvc'
    if (destination / 'provenance.json').is_file() and list(destination.glob('VC/Tools/MSVC/14.38.*/bin/Hostx64/x64/cl.exe')):
        print('Local MSVC 14.38 is available', flush=True)
        return
    channel = json.load(urllib.request.urlopen('https://aka.ms/vs/17/release/channel', timeout=60))
    manifest_url = next(p['payloads'][0]['url'] for p in channel['channelItems']
                        if p['id'] == 'Microsoft.VisualStudio.Manifests.VisualStudio')
    manifest = json.load(urllib.request.urlopen(manifest_url, timeout=60))
    package_ids = [
        'Microsoft.VC.14.38.17.8.CRT.Headers.base',
        'Microsoft.VC.14.38.17.8.CRT.x64.Desktop.base',
        'Microsoft.VC.14.38.17.8.Tools.HostX64.TargetX64.base',
        'Microsoft.VC.14.38.17.8.Tools.HostX64.TargetX64.Res.base',
    ]
    records = []
    for package_id in package_ids:
        package = next(p for p in manifest['packages'] if p['id'] == package_id
                       and p.get('language', 'en-US') == 'en-US')
        for payload in package['payloads']:
            print(f"Downloading {payload['fileName']} ({payload['size'] // 1024**2} MiB)", flush=True)
            data = urllib.request.urlopen(payload['url'], timeout=120).read()
            digest = hashlib.sha256(data).hexdigest()
            if digest.lower() != payload['sha256'].lower():
                raise RuntimeError(f"Hash mismatch: {payload['fileName']}")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    name = member.filename.replace('\\', '/')
                    if not name.startswith('Contents/') or member.is_dir():
                        continue
                    target = destination / name.removeprefix('Contents/')
                    if not target.resolve().is_relative_to(destination.resolve()):
                        raise RuntimeError(f'Unsafe archive path: {name}')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.read(member))
            records.append({'id': package_id, 'version': package['version'], 'sha256': digest, 'url': payload['url']})
    (destination / 'provenance.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    if not list(destination.glob('VC/Tools/MSVC/14.38.*/bin/Hostx64/x64/cl.exe')):
        raise RuntimeError('MSVC archive layout did not contain cl.exe')
    print('Local MSVC 14.38 is available', flush=True)


if __name__ == '__main__':
    main()
