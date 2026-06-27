import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

import hanlp.datasets.parsing.loaders._ctb_utils as ctb_utils
import hanlp.utils.io_util as io_util


ARCHIVE_PATH = Path(r"e:\BaseNP\LDC2013T21.tgz")
OUTPUT_ROOT = Path(r"e:\BaseNP\CTB")


def decode_output(raw: bytes) -> str:
    for encoding in ('utf-8', 'gbk', 'mbcs'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def patch_hanlp_windows_decoding() -> None:
    # HanLP 2.1.3 hardcodes utf-8 when decoding subprocess output, which breaks on
    # Chinese Windows when Java tools print GBK-encoded messages.
    def safe_get_exitcode_stdout_stderr(cmd: str) -> tuple[int, str, str]:
        process = subprocess.Popen(
            io_util.shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        return process.returncode, decode_output(stdout), decode_output(stderr)

    io_util.get_exitcode_stdout_stderr = safe_get_exitcode_stdout_stderr
    ctb_utils.get_exitcode_stdout_stderr = safe_get_exitcode_stdout_stderr


def find_bracketed_dir(root: Path) -> Path:
    for current_root, dirnames, _ in os.walk(root):
        if 'bracketed' not in dirnames:
            continue
        candidate = Path(current_root) / 'bracketed'
        if any(child.name.startswith('chtb_') for child in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f'未在 {root} 中找到 CTB 的 bracketed 目录')


def list_cleaned_files(cleaned_root: Path) -> list[str]:
    return sorted(
        file.name
        for file in cleaned_root.iterdir()
        if file.is_file() and file.name.startswith('chtb')
    )


def reset_generated_dirs(output_root: Path) -> None:
    for name in ['cleaned_bracket', 'cws', 'pos', 'par', 'dep']:
        target = output_root / name
        if target.exists():
            shutil.rmtree(target)


def build_splits(cleaned_files: list[str]) -> Iterable[tuple[str, list[str]]]:
    train, dev, test = ctb_utils.split_chtb(cleaned_files)
    print(
        '3/4 划分数据集: '
        f'train={len(train)} dev={len(dev)} test={len(test)}'
    )
    return (
        ('train', train),
        ('dev', dev),
        ('test', test),
    )


def main() -> None:
    if not ARCHIVE_PATH.is_file():
        raise FileNotFoundError(f'找不到数据包: {ARCHIVE_PATH}')

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    reset_generated_dirs(OUTPUT_ROOT)
    patch_hanlp_windows_decoding()

    with tempfile.TemporaryDirectory(prefix='ctb8_', dir=OUTPUT_ROOT.parent) as temp_dir:
        temp_root = Path(temp_dir)
        print(f'1/4 解压数据包: {ARCHIVE_PATH}')
        with tarfile.open(ARCHIVE_PATH, 'r:gz') as archive:
            archive.extractall(temp_root)

        bracketed_root = find_bracketed_dir(temp_root)
        extracted_ctb_root = bracketed_root.parent

        print(f'2/4 清洗 bracketed 文件 -> {OUTPUT_ROOT / "cleaned_bracket"}')
        ctb_utils.clean_ctb_bracketed(str(extracted_ctb_root), str(OUTPUT_ROOT / 'cleaned_bracket'))

        cleaned_files = list_cleaned_files(OUTPUT_ROOT / 'cleaned_bracket')
        for part_name, files in build_splits(cleaned_files):
            file_paths = [str(OUTPUT_ROOT / 'cleaned_bracket' / name) for name in files]
            print(f'4/4 生成 {part_name} 集任务文件')
            ctb_utils.make_ctb_tasks(file_paths, str(OUTPUT_ROOT), part_name)

    print(f'预处理完成，输出目录: {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
