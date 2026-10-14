"""Download the upstream Windows torch wheel in parallel, verifying SHA-256."""

import concurrent.futures
import hashlib
import html
import re
import threading
import time
import urllib.request
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    filename = 'torch-2.3.0+cu121-cp310-cp310-win_amd64.whl'
    destination = Path(__file__).resolve().parents[1] / '.build-tools/wheels' / filename
    index = urllib.request.urlopen('https://download.pytorch.org/whl/cu121/torch/', timeout=60).read().decode()
    link = next(html.unescape(url) for url in re.findall(r'href="([^"]+)"', index)
                if 'torch-2.3.0%2Bcu121-cp310-cp310-win_amd64.whl#' in url)
    url, expected_hash = link.split('#sha256=')
    if destination.exists():
        if sha256_file(destination) == expected_hash:
            print('Verified torch wheel is available', flush=True)
            return
        raise RuntimeError('Existing torch wheel has an unexpected SHA-256')
    request = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.status != 206:
            raise RuntimeError('Upstream server does not support range downloads')
        size = int(response.headers['Content-Range'].split('/')[-1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.whl.part')
    with temporary.open('wb') as stream:
        stream.truncate(size)
    block_size = 16 * 1024**2
    completed = 0
    lock = threading.Lock()
    start_time = time.monotonic()

    def fetch(start):
        nonlocal completed
        end = min(start + block_size, size) - 1
        for attempt in range(3):
            try:
                req = urllib.request.Request(url + f'?gvhmr_range={start}',
                                             headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                        raise RuntimeError('Unexpected upstream range response')
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError('Incomplete range response')
                with temporary.open('r+b') as stream:
                    stream.seek(start)
                    stream.write(data)
                with lock:
                    completed += len(data)
                    if completed // (128 * 1024**2) != (completed - len(data)) // (128 * 1024**2):
                        elapsed = time.monotonic() - start_time
                        print(f'Torch: {completed / size:.0%}, {completed / 1024**2 / elapsed:.1f} MiB/s', flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(fetch, range(0, size, block_size)))
    digest = sha256_file(temporary)
    if digest != expected_hash:
        raise RuntimeError('Downloaded torch wheel has an unexpected SHA-256')
    temporary.rename(destination)
    print('Verified torch wheel is available', flush=True)


if __name__ == '__main__':
    main()
