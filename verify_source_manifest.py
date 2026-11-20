"""Check the preserved original V5 artifacts against their SHA-256 manifest."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest_sha256.json").read_text())
failed = []
for name, expected in sorted(manifest.items()):
    path = root / ("README_V5_ORIGINAL.md" if name == "README.md" else name)
    if not path.is_file():
        print(f"MISSING {name}")
        failed.append(name)
        continue
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    status = "OK" if actual == expected else "MISMATCH"
    print(f"{status:8} {name}")
    if actual != expected:
        failed.append(name)
if failed:
    raise SystemExit(f"Verification failed for {len(failed)} file(s)")
print("All preserved original V5 files match their hashes.")
