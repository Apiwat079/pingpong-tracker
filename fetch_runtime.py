"""Fetch the pinned web runtime used as the original-host fallback."""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/'
FILES = {
    'ort-wasm-simd-threaded.asyncify.mjs': '7236653b8565da4046e459cd0e274123419a1d9f1f8f18fd36c28058346ca655',
    'ort-wasm-simd-threaded.asyncify.wasm': '7e83cd6cee77e478bc96a7e91b198144fb5e4126287daf1f9b54bb195ebcd55a',
    'ort.webgpu.min.mjs': '46988a5a025f49449850f39f95eb0d21e40e67b3beb13a0b54efd3ab5d83f60e',
}


def main():
    target = Path(__file__).resolve().parent / 'static' / 'ort'
    target.mkdir(parents=True, exist_ok=True)
    for name, wanted in FILES.items():
        path = target / name
        if path.exists() and sha256(path.read_bytes()).hexdigest() == wanted:
            print(f'{name}: already verified')
            continue
        with urlopen(BASE + name, timeout=90) as response:
            data = response.read(26 * 1024 * 1024)
        if sha256(data).hexdigest() != wanted:
            raise RuntimeError(f'Unexpected checksum for {name}')
        pending = path.with_suffix(path.suffix + '.part')
        pending.write_bytes(data)
        pending.replace(path)
        print(f'{name}: downloaded and verified')


if __name__ == '__main__':
    main()
