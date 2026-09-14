"""Fetch a pinned Microsoft runtime package and extract hash-verified DLLs."""
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
from urllib.request import urlopen

PACKAGE_URL = 'https://download.visualstudio.microsoft.com/download/pr/ebdab8e5-1d7b-4d9f-a11b-cbb1720c3b12/843068991DAAA1F73AD9F6239BCE4D0F6A07A51F18C37EA2A867E9BECA71295C/VC_redist.x64.exe'
PACKAGE_SHA = '843068991daaa1f73ad9f6239bce4d0f6a07a51f18c37ea2a867e9beca71295c'
DLLS = {
    'MSVCP140.dll': '7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca',
    'VCRUNTIME140.dll': 'd1f4225df2cd877dbf130d5668a021dce3f94118455ff5ec952061c30afc9ce7',
    'VCRUNTIME140_1.dll': 'a7146c08f89fe5b04541ab507cdb59ff7b44534d4ba3c668a426c6450a03434e',
}
LICENSE_URL = 'https://visualstudio.microsoft.com/wp-content/uploads/2025/10/Visual-C-V14-License-Redistributable_and_Runtime_ENU.docx'
LICENSE_SHA = '08651651a7602fc7c0e2763de0fde1ff9f868df2780597cd1775ee9d6441c783'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, target, expected):
    if not target.exists():
        with urlopen(url, timeout=120) as source, target.open('xb') as destination:
            shutil.copyfileobj(source, destination, 1024 * 1024)
    if sha(target) != expected:
        raise ValueError('Build input checksum mismatch')


def fetch(output, scratch, seven_zip='7z'):
    output.mkdir(parents=True, exist_ok=False)
    scratch.mkdir(parents=True, exist_ok=True)
    package = scratch / 'vc_redist.x64.exe'
    download(PACKAGE_URL, package, PACKAGE_SHA)
    seven_zip = shutil.which(seven_zip) or str(Path('C:/Program Files/7-Zip/7z.exe'))
    # Burn bundles contain multiple attached CABs. Opening the EXE directly with
    # 7-Zip exposes only its bootstrapper UI, not the runtime payload container.
    blob = package.read_bytes()
    cabinets = []
    for offset in range(len(blob) - 36):
        if blob[offset:offset + 8] != b'MSCF\0\0\0\0':
            continue
        size = struct.unpack_from('<I', blob, offset + 8)[0]
        if 36 <= size <= len(blob) - offset:
            path = scratch / f'container-{offset}.cab'
            path.write_bytes(blob[offset:offset + size])
            cabinets.append(path)
    index = 0
    while index < len(cabinets):
        if len(cabinets) > 32:
            raise ValueError('Unexpected runtime cabinet nesting')
        destination = scratch / f'extracted-{index}'
        subprocess.run([seven_zip, 'x', str(cabinets[index]), '-o' + str(destination), '-y'], check=True, stdout=subprocess.DEVNULL)
        for path in destination.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    if stream.read(4) == b'MSCF':
                        cabinets.append(path)
        index += 1
    by_hash = {}
    for path in scratch.rglob('*'):
        if path.is_file() and 10000 <= path.stat().st_size <= 2 * 1024**2:
            by_hash[sha(path)] = path
    for name, checksum in DLLS.items():
        if checksum not in by_hash:
            raise ValueError('Pinned runtime DLL was not found in extracted package')
        shutil.copyfile(by_hash[checksum], output / name)
    license_path = scratch / 'runtime-license.docx'
    download(LICENSE_URL, license_path, LICENSE_SHA)
    import zipfile

    from defusedxml import ElementTree
    with zipfile.ZipFile(license_path) as archive:
        xml = ElementTree.fromstring(archive.read('word/document.xml'))
    ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    text = '\n'.join(''.join(n.itertext()) for n in xml.iter(ns + 'p'))
    (output / 'LICENSE.txt').write_text(text, encoding='utf-8')
    (output / 'runtime-provenance.json').write_text(json.dumps({
        'package_url': PACKAGE_URL, 'sha256': PACKAGE_SHA, 'version': '14.51.36247.0',
        'files': DLLS, 'license_url': LICENSE_URL, 'license_sha256': LICENSE_SHA,
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--seven-zip', default='7z')
    args = parser.parse_args()
    fetch(args.output.resolve(), args.scratch.resolve(), args.seven_zip)
