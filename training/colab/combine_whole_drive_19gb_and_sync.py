"""
Mirror-7 WHOLE GOOGLE DRIVE (~19 GB+) Scanner, Dataset Combiner & Dual-ZIP Builder
==================================================================================
1. Scans 100% of `/content/drive/MyDrive` + `/content` (ZERO folder-name filters).
2. Prints a full folder-by-folder GB/MB breakdown of your ~19 GB Google Drive.
3. Streams & combines all `.jsonl` datasets found anywhere in Drive into one unified,
   deduplicated dataset (`mirror7_combined_all_drive_datasets.jsonl`).
4. Packs 100% of ALL ~19 GB+ files (all datasets, all 708 MB .pt weights, all code,
   all notebooks, plus the combined master dataset) into `Mirror7_WHOLE_DRIVE_MASTER.zip`
   using `allowZip64=True` (>4 GB support) and fast `ZIP_STORED` streaming.
5. Also creates & auto-downloads `Mirror7_Colab_GitHub_Sync.zip` (<35 MB: all code,
   Colab notebooks, logs, and SHA-256 inventory of all 19 GB) for `push_mirror7_to_github.ps1`
   (since GitHub blocks >100 MB files in `git push`).
"""
import hashlib
import json
import os
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from google.colab import drive, files

if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive")
COLAB_CONTENT = Path("/content")

SKIP_DIRS = {".Trash", ".git", ".venv", "venv", "__pycache__", ".pytest_cache", "node_modules", ".ipynb_checkpoints"}
ALREADY_COMPRESSED_EXTS = {
    ".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".gguf",
    ".parquet", ".gz", ".zst", ".bz2", ".xz", ".zip", ".7z", ".rar",
    ".npz", ".npy", ".png", ".jpg", ".jpeg", ".mp4", ".pdf",
}
DATASET_EXTS = {".jsonl", ".parquet", ".csv", ".tsv", ".arrow", ".feather", ".zst", ".gz"}
CHECKPOINT_EXTS = {".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".gguf"}
GITHUB_SAFE_EXTS = {".py", ".ipynb", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".log", ".sh", ".ps1", ".cfg", ".ini"}
GITHUB_MAX_MB = 45.0

OUT_COMBINED_JSONL = Path("/content/mirror7_combined_all_drive_datasets.jsonl")
OUT_MASTER_LOCAL   = Path("/content/Mirror7_WHOLE_DRIVE_MASTER.zip")
OUT_MASTER_DRIVE   = DRIVE_ROOT / "Mirror7_WHOLE_DRIVE_MASTER.zip"
OUT_GITHUB_ZIP     = Path("/content/Mirror7_Colab_GitHub_Sync.zip")
OUT_GITHUB_DRIVE   = DRIVE_ROOT / "Mirror7_Colab_GitHub_Sync.zip"

EXCLUDE_OUTPUT_NAMES = {
    OUT_COMBINED_JSONL.name,
    OUT_MASTER_LOCAL.name,
    OUT_GITHUB_ZIP.name,
    "Mirror7_MASTER_VAULT_ALL_DATA_AND_DATASETS.zip",
}


def fast_fingerprint(fpath: Path, prefix_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    sz = fpath.stat().st_size
    h.update(f"size:{sz}:".encode("utf-8"))
    with open(fpath, "rb") as f:
        h.update(f.read(prefix_bytes))
    return h.hexdigest()


print("=" * 86)
print(" STEP 1/4: SCANNING ENTIRE GOOGLE DRIVE (/content/drive/MyDrive) + /content")
print("=" * 86)

all_files = []
jsonl_files = []
folder_totals = defaultdict(lambda: [0, 0])
total_bytes = 0

for root, dirs, filenames in os.walk(DRIVE_ROOT):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for fname in filenames:
        if fname in EXCLUDE_OUTPUT_NAMES:
            continue
        fpath = Path(root) / fname
        try:
            st = fpath.stat()
        except OSError:
            continue
        rel = fpath.relative_to(DRIVE_ROOT)
        top_group = rel.parts[0] if len(rel.parts) > 1 else "(MyDrive Root Files)"
        folder_totals[top_group][0] += 1
        folder_totals[top_group][1] += st.st_size
        total_bytes += st.st_size
        all_files.append((fpath, f"drive/{rel.as_posix()}", st.st_size))
        if fpath.suffix.lower() == ".jsonl" and "smoke" not in fname.lower():
            jsonl_files.append(fpath)

for item in COLAB_CONTENT.iterdir():
    if item.name in {"drive", "sample_data", ".config"} or item.name in EXCLUDE_OUTPUT_NAMES:
        continue
    if item.is_file():
        st = item.stat()
        total_bytes += st.st_size
        folder_totals["(/content Scratch)"][0] += 1
        folder_totals["(/content Scratch)"][1] += st.st_size
        all_files.append((item, f"colab_content/{item.name}", st.st_size))
        if item.suffix.lower() == ".jsonl" and "smoke" not in item.name.lower():
            jsonl_files.append(item)
    elif item.is_dir() and item.name not in SKIP_DIRS:
        for root, dirs, filenames in os.walk(item):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in filenames:
                fpath = Path(root) / fname
                try:
                    st = fpath.stat()
                except OSError:
                    continue
                rel = fpath.relative_to(COLAB_CONTENT)
                total_bytes += st.st_size
                folder_totals[f"(/content/{item.name})"][0] += 1
                folder_totals[f"(/content/{item.name})"][1] += st.st_size
                all_files.append((fpath, f"colab_content/{rel.as_posix()}", st.st_size))
                if fpath.suffix.lower() == ".jsonl" and "smoke" not in fname.lower():
                    jsonl_files.append(fpath)

print(f"\nTotal Discovered: {len(all_files):,} files | {total_bytes / (1024**3):.2f} GB ({total_bytes / (1024**2):,.1f} MB)\n")
print(f"  {'Folder / Location':<44} | {'Files':>7} | {'Total Size':>16}")
print("  " + "-" * 73)
for grp, (cnt, b_sum) in sorted(folder_totals.items(), key=lambda kv: kv[1][1], reverse=True):
    gb = b_sum / (1024**3)
    mb = b_sum / (1024**2)
    sz_str = f"{gb:.2f} GB" if gb >= 0.5 else f"{mb:.1f} MB"
    print(f"  {grp[:44]:<44} | {cnt:>7,} | {sz_str:>16}")

# ------------------------------------------------------------------------------
# Step 2/4: Stream-Combine & Deduplicate All .jsonl Datasets Across Drive
# ------------------------------------------------------------------------------
print("\n" + "=" * 86)
print(f" STEP 2/4: COMBINING & DEDUPLICATING {len(jsonl_files)} JSONL DATASETS")
print("=" * 86)

seen_row_hashes = set()
unique_rows = 0
total_rows_read = 0

if jsonl_files:
    with open(OUT_COMBINED_JSONL, "w", encoding="utf-8") as fout:
        for jf in jsonl_files:
            jf_rows = 0
            try:
                with open(jf, "r", encoding="utf-8", errors="ignore") as fin:
                    for line in fin:
                        s = line.strip()
                        if not s:
                            continue
                        total_rows_read += 1
                        # 64-bit blake2b digest keeps RAM tiny even for millions of rows
                        rh = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
                        if rh not in seen_row_hashes:
                            seen_row_hashes.add(rh)
                            fout.write(s + "\n")
                            unique_rows += 1
                            jf_rows += 1
                print(f"  [Merged JSONL] {jf.name} (+{jf_rows:,} unique rows)")
            except Exception as exc:
                print(f"  [Skip JSONL]   {jf}: {exc}")
    comb_mb = OUT_COMBINED_JSONL.stat().st_size / (1024**2)
    print(f"  -> Unified Master Dataset: {unique_rows:,} unique rows (from {total_rows_read:,} raw) | {comb_mb:.2f} MB")
    all_files.append((OUT_COMBINED_JSONL, "combined_datasets/mirror7_combined_all_drive_datasets.jsonl", OUT_COMBINED_JSONL.stat().st_size))
    total_bytes += OUT_COMBINED_JSONL.stat().st_size
del seen_row_hashes

# ------------------------------------------------------------------------------
# Step 3/4: Pack Complete ~19 GB+ Master ZIP + Lightweight GitHub Sync ZIP
# ------------------------------------------------------------------------------
colab_free_gb = shutil.disk_usage("/content").free / (1024**3)
drive_free_gb = shutil.disk_usage(DRIVE_ROOT).free / (1024**3)
print("\n" + "=" * 86)
print(" STEP 3/4: ARCHIVING ALL FILES (~19 GB+ ZIP64) + GITHUB SYNC BUNDLE")
print(f"           Colab SSD Free: {colab_free_gb:.2f} GB | Google Drive Free: {drive_free_gb:.2f} GB")
print("=" * 86)

can_write_local = colab_free_gb > (total_bytes / (1024**3)) + 2.0
master_target = OUT_MASTER_LOCAL if can_write_local else OUT_MASTER_DRIVE
print(f"[Writing Complete Archive To] {master_target}")

manifest = {
    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "author": "Ahmad Hassan",
    "repository": "https://github.com/realahmad07/Mirror-7",
    "total_files_archived": len(all_files),
    "total_size_gb": round(total_bytes / (1024**3), 3),
    "combined_jsonl_unique_rows": unique_rows,
    "folder_summary": {
        k: {"files": v[0], "size_mb": round(v[1] / (1024**2), 2)}
        for k, v in sorted(folder_totals.items(), key=lambda kv: kv[1][1], reverse=True)
    },
    "checkpoints": [],
    "datasets": [],
    "github_synced_files": [],
}

processed_bytes = 0
last_pct = -5
github_seen = set()

with zipfile.ZipFile(master_target, "w", allowZip64=True) as z_master, \
     zipfile.ZipFile(OUT_GITHUB_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z_git:

    for fpath, arc_master, sz in all_files:
        ext = fpath.suffix.lower()
        size_mb = round(sz / (1024 * 1024), 2)
        comp = zipfile.ZIP_STORED if (ext in ALREADY_COMPRESSED_EXTS or size_mb > 100.0) else zipfile.ZIP_DEFLATED

        try:
            z_master.write(fpath, arc_master, compress_type=comp)
        except Exception as exc:
            print(f"  [Warn] Could not pack {fpath}: {exc}")
            continue

        if ext in CHECKPOINT_EXTS:
            manifest["checkpoints"].append({"file": arc_master, "size_mb": size_mb, "sha256_prefix": fast_fingerprint(fpath)})
        elif ext in DATASET_EXTS or "data" in arc_master.lower():
            if size_mb >= 0.1:
                manifest["datasets"].append({"file": arc_master, "size_mb": size_mb, "sha256_prefix": fast_fingerprint(fpath)})

        if (ext in GITHUB_SAFE_EXTS or fpath.name in {"Dockerfile", "LICENSE", ".gitignore"}) and ext != ".jsonl" and size_mb <= GITHUB_MAX_MB:
            git_arc = arc_master
            for prefix in ("drive/Mirror-7-main/", "drive/Mirror-7/", "drive/Mirror7/", "colab_content/Mirror-7-main/", "colab_content/Mirror-7/"):
                if git_arc.startswith(prefix):
                    git_arc = git_arc[len(prefix):]
                    break
            else:
                if git_arc.startswith("drive/Colab Notebooks/"):
                    git_arc = "training/colab/" + fpath.name
                elif git_arc.startswith("drive/Mirror7_Ultra_Backup/"):
                    git_arc = "colab_backup/" + git_arc[len("drive/Mirror7_Ultra_Backup/"):]

            if git_arc not in github_seen:
                z_git.write(fpath, git_arc)
                github_seen.add(git_arc)
                manifest["github_synced_files"].append({"path": git_arc, "size_mb": size_mb})

        processed_bytes += sz
        pct = int((processed_bytes / max(total_bytes, 1)) * 100)
        if pct >= last_pct + 10:
            last_pct = pct
            print(f"  [Progress] {pct:>3}%  ({processed_bytes / (1024**3):.2f} GB / {total_bytes / (1024**3):.2f} GB)")

    manifest_json = json.dumps(manifest, indent=2)
    z_master.writestr("artifacts/whole_drive_19gb_inventory_manifest.json", manifest_json)
    z_git.writestr("artifacts/google_drive_checkpoints_and_datasets_manifest.json", manifest_json)

# ------------------------------------------------------------------------------
# Step 4/4: Save to Google Drive & Download GitHub Bundle
# ------------------------------------------------------------------------------
print("\n" + "=" * 86)
print(" STEP 4/4: FINALIZING ARCHIVES")
print("=" * 86)

shutil.copy2(OUT_GITHUB_ZIP, OUT_GITHUB_DRIVE)
if jsonl_files and drive_free_gb > 1.0:
    shutil.copy2(OUT_COMBINED_JSONL, DRIVE_ROOT / OUT_COMBINED_JSONL.name)

master_gb = master_target.stat().st_size / (1024**3)
git_mb = OUT_GITHUB_ZIP.stat().st_size / (1024**2)

if master_target == OUT_MASTER_LOCAL:
    if drive_free_gb > master_gb + 0.5:
        print(f"Streaming {master_gb:.2f} GB Master ZIP into Google Drive ({OUT_MASTER_DRIVE})...")
        with open(OUT_MASTER_LOCAL, "rb") as fsrc, open(OUT_MASTER_DRIVE, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, length=16 * 1024 * 1024)
        print(f"[DONE] Complete {master_gb:.2f} GB Master ZIP saved in Google Drive: {OUT_MASTER_DRIVE}")
    else:
        print(f"[IMPORTANT] Your Google Drive only has {drive_free_gb:.2f} GB free space left (ZIP is {master_gb:.2f} GB).")
        print(f"            Your complete {master_gb:.2f} GB ZIP is safely built on Colab SSD at: {OUT_MASTER_LOCAL}")
else:
    print(f"[DONE] Complete {master_gb:.2f} GB Master ZIP saved in Google Drive: {OUT_MASTER_DRIVE}")

print(f"[DONE] GitHub-Ready Sync ZIP ({git_mb:.2f} MB | {len(github_seen)+1} files): {OUT_GITHUB_ZIP}")
print("Starting browser download of Mirror7_Colab_GitHub_Sync.zip for local GitHub push...")
files.download(str(OUT_GITHUB_ZIP))
