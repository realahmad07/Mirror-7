"""
Mirror-7 Google Colab -> Complete Master Archive + GitHub Sync ZIP Generator
============================================================================
Paste and run this single cell in Google Colab!

It automatically builds TWO archives from your latest Colab & Google Drive files:
1) COMPLETE MASTER VAULT (Saved directly inside your Google Drive):
   `/content/drive/MyDrive/Mirror7_MASTER_VAULT_ALL_DATA_AND_DATASETS.zip`
   -> Saves ALL datasets (.jsonl), ALL latest Python code (.py), ALL Colab notebooks (.ipynb),
      ALL logs/JSONs, and your compact BF16 master checkpoint in one permanent archive
      (uses fast ZIP_STORED for binary .pt weights so archiving finishes in seconds!).

2) GITHUB-READY SYNC BUNDLE (Auto-downloaded to your Windows PC `Downloads` folder):
   `/content/Mirror7_Colab_GitHub_Sync.zip`
   -> Packs all latest .py modules, standalone inference scripts, Colab .ipynb notebooks,
      logs, JSON benchmarks, and a cryptographic SHA-256 manifest of your Drive datasets
      & 708 MB .pt weights — ready for `push_mirror7_to_github.ps1` on your local PC!
"""
import hashlib
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

try:
    from google.colab import drive, files
    IN_COLAB = True
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except ImportError:
    IN_COLAB = False

DRIVE_ROOT = Path("/content/drive/MyDrive")
COLAB_CONTENT = Path("/content")

CANDIDATE_DIRS = [
    DRIVE_ROOT / "Mirror7_Ultra_Backup",
    DRIVE_ROOT / "Mirror-7-main",
    DRIVE_ROOT / "Mirror-7",
    DRIVE_ROOT / "Mirror7",
    COLAB_CONTENT / "Mirror-7-main",
    COLAB_CONTENT / "Mirror-7",
]
COLAB_NB_DIR = DRIVE_ROOT / "Colab Notebooks"

CODE_AND_DOC_EXTS = {
    ".py", ".ipynb", ".md", ".json", ".yaml", ".yml",
    ".toml", ".txt", ".log", ".sh", ".ps1", ".cfg", ".ini",
}
SKIP_DIR_NAMES = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".ipynb_checkpoints"}
GITHUB_MAX_SINGLE_MB = 45.0


def sha256_fast_prefix(fpath: Path, max_bytes: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    size = fpath.stat().st_size
    h.update(f"size:{size}:".encode("utf-8"))
    with open(fpath, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


print("=" * 80)
print(" MIRROR-7 GOOGLE COLAB & DRIVE ARCHIVER (FULL VAULT + GITHUB SYNC BUNDLE)")
print("=" * 80)

found_dirs = [d for d in CANDIDATE_DIRS if d.exists()]
for d in found_dirs:
    print(f"  [Found Directory] {d}")

manifest = {
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "author": "Ahmad Hassan",
    "repository": "https://github.com/realahmad07/Mirror-7",
    "checkpoints_on_drive": [],
    "datasets_on_drive": [],
    "synced_code_and_notebooks": [],
}

master_vault_zip = DRIVE_ROOT / "Mirror7_MASTER_VAULT_ALL_DATA_AND_DATASETS.zip" if DRIVE_ROOT.exists() else Path("/content/Mirror7_MASTER_VAULT_ALL_DATA_AND_DATASETS.zip")
github_sync_zip = Path("/content/Mirror7_Colab_GitHub_Sync.zip")
drive_github_sync_copy = DRIVE_ROOT / "Mirror7_Colab_GitHub_Sync.zip"

vault_arcnames = set()
sync_arcnames = set()

with zipfile.ZipFile(master_vault_zip, "w", allowZip64=True) as z_vault, \
     zipfile.ZipFile(github_sync_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z_sync:

    # 1. Walk all Mirror directories on Drive & /content
    for base_dir in found_dirs:
        is_backup = (base_dir.name == "Mirror7_Ultra_Backup")
        for root, dirs, filenames in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]
            for fname in filenames:
                fpath = Path(root) / fname
                # Avoid zipping the output zip into itself
                if fpath in {master_vault_zip, github_sync_zip, drive_github_sync_copy}:
                    continue
                try:
                    st = fpath.stat()
                except OSError:
                    continue
                size_mb = round(st.st_size / (1024 * 1024), 2)
                ext = fpath.suffix.lower()
                rel = fpath.relative_to(base_dir).as_posix()
                arcname = f"colab_backup/{rel}" if is_backup else rel

                # A) Checkpoints (.pt): record SHA-256 manifest + store compact/final master in Drive Vault
                if ext in {".pt", ".pth", ".safetensors", ".bin"}:
                    manifest["checkpoints_on_drive"].append({
                        "name": fname,
                        "drive_path": str(fpath),
                        "size_mb": size_mb,
                        "fingerprint_sha256_16mb": sha256_fast_prefix(fpath),
                    })
                    # Include compact BF16 master or <=400MB checkpoints in Master Vault using fast ZIP_STORED
                    if ("COMPACT" in fname.upper() or size_mb <= 400.0) and arcname not in vault_arcnames:
                        z_vault.write(fpath, arcname, compress_type=zipfile.ZIP_STORED)
                        vault_arcnames.add(arcname)
                        print(f"  [Vault + Indexed]  {fname} ({size_mb} MB)")
                    else:
                        print(f"  [Checkpoint Index] {fname} ({size_mb} MB — kept directly in Drive)")
                    continue

                # B) Datasets (.jsonl): save ALL datasets into Master Vault + index in manifest
                if ext == ".jsonl":
                    manifest["datasets_on_drive"].append({
                        "name": fname,
                        "drive_path": str(fpath),
                        "size_mb": size_mb,
                        "fingerprint_sha256_16mb": sha256_fast_prefix(fpath),
                    })
                    if arcname not in vault_arcnames:
                        z_vault.write(fpath, arcname, compress_type=zipfile.ZIP_DEFLATED)
                        vault_arcnames.add(arcname)
                        print(f"  [Dataset Vaulted]  {fname} ({size_mb} MB)")
                    continue

                # C) Code, notebooks, configs, standalone inference, JSON/MD logs -> BOTH Vault & GitHub Sync ZIP
                if (ext in CODE_AND_DOC_EXTS or fname in {"Dockerfile", "LICENSE", ".gitignore"}):
                    if arcname not in vault_arcnames:
                        z_vault.write(fpath, arcname, compress_type=zipfile.ZIP_DEFLATED)
                        vault_arcnames.add(arcname)
                    if size_mb <= GITHUB_MAX_SINGLE_MB and arcname not in sync_arcnames:
                        z_sync.write(fpath, arcname)
                        sync_arcnames.add(arcname)
                        manifest["synced_code_and_notebooks"].append({"arcname": arcname, "size_mb": size_mb})

    # 2. Capture Mirror Colab Notebooks (.ipynb) from `/content/drive/MyDrive/Colab Notebooks`
    if COLAB_NB_DIR.exists():
        for nb_file in COLAB_NB_DIR.glob("*.ipynb"):
            low = nb_file.name.lower()
            if any(k in low for k in ("mirror", "stage", "ultra", "quantum", "v5", "v6")):
                size_mb = round(nb_file.stat().st_size / (1024 * 1024), 2)
                arcname = f"training/colab/{nb_file.name}"
                if arcname not in vault_arcnames:
                    z_vault.write(nb_file, arcname, compress_type=zipfile.ZIP_DEFLATED)
                    vault_arcnames.add(arcname)
                if size_mb <= GITHUB_MAX_SINGLE_MB and arcname not in sync_arcnames:
                    z_sync.write(nb_file, arcname)
                    sync_arcnames.add(arcname)
                    manifest["synced_code_and_notebooks"].append({"arcname": arcname, "size_mb": size_mb})
                    print(f"  [Colab Notebook]   {nb_file.name} -> {arcname} ({size_mb} MB)")

    # 3. Capture any loose *mirror*.py scripts in /content
    for py_file in COLAB_CONTENT.glob("*mirror*.py"):
        size_mb = round(py_file.stat().st_size / (1024 * 1024), 2)
        arcname = f"training/colab/{py_file.name}"
        if arcname not in vault_arcnames:
            z_vault.write(py_file, arcname, compress_type=zipfile.ZIP_DEFLATED)
            vault_arcnames.add(arcname)
        if size_mb <= GITHUB_MAX_SINGLE_MB and arcname not in sync_arcnames:
            z_sync.write(py_file, arcname)
            sync_arcnames.add(arcname)
            manifest["synced_code_and_notebooks"].append({"arcname": arcname, "size_mb": size_mb})

    # 4. Write cryptographic SHA-256 manifest into BOTH archives
    manifest_str = json.dumps(manifest, indent=2)
    z_vault.writestr("artifacts/google_drive_checkpoints_and_datasets_manifest.json", manifest_str)
    z_sync.writestr("artifacts/google_drive_checkpoints_and_datasets_manifest.json", manifest_str)

if DRIVE_ROOT.exists():
    shutil.copy2(github_sync_zip, drive_github_sync_copy)

vault_mb = round(master_vault_zip.stat().st_size / (1024 * 1024), 2)
sync_mb = round(github_sync_zip.stat().st_size / (1024 * 1024), 2)

print("\n" + "=" * 80)
print(f"[1] FULL MASTER VAULT SAVED ON DRIVE : {master_vault_zip} ({vault_mb} MB)")
print(f"[2] GITHUB SYNC ZIP READY FOR PC     : {github_sync_zip} ({sync_mb} MB | {len(sync_arcnames) + 1} files)")
print(f"    - Checkpoints SHA-256 Indexed    : {len(manifest['checkpoints_on_drive'])}")
print(f"    - Datasets Archived & Indexed    : {len(manifest['datasets_on_drive'])}")
print("=" * 80)

if IN_COLAB:
    print("Downloading Mirror7_Colab_GitHub_Sync.zip to your PC Downloads folder now...")
    files.download(str(github_sync_zip))
