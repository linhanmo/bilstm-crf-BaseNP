import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable

import hanlp.datasets.parsing.loaders._ctb_utils as ctb_utils
import hanlp.utils.io_util as io_util


ARCHIVE_PATH = Path(r"LDC2013T21.tgz")
OUTPUT_ROOT = Path(r"CTB")


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
        if os.name == 'nt':
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
        else:
            process = subprocess.Popen(
                io_util.shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        stdout, stderr = process.communicate()
        return process.returncode, decode_output(stdout), decode_output(stderr)

    io_util.get_exitcode_stdout_stderr = safe_get_exitcode_stdout_stderr
    ctb_utils.get_exitcode_stdout_stderr = safe_get_exitcode_stdout_stderr

    def safe_remove_all_ec(path: str) -> None:
        script = io_util.get_resource('https://file.hankcs.com/bin/remove_ec.zip')
        classpath = ';'.join([
            'elit-ddr-0.0.5-SNAPSHOT.jar',
            'elit-sdk-0.0.5-SNAPSHOT.jar',
            'hanlp-1.7.8.jar',
            'fastutil-8.1.1.jar',
            '.',
        ])
        with io_util.pushd(script):
            io_util.run_cmd(
                f'java -cp {classpath} demo.RemoveEmptyCategoriesTreebank "{Path(path).as_posix()}"'
            )

    def safe_convert_to_dependency(src: str, dst: str, language: str = 'zh', version: str = '3.3.0',
                                   conllx: bool = True, ud: bool = False) -> None:
        ctb_utils.cprint(
            f'Converting {os.path.basename(src)} to {os.path.basename(dst)} using Stanford Parser Version {version}. '
            f'It might take a while [blink][yellow]...[/yellow][/blink]'
        )
        if version == '3.3.0':
            sp_home = 'https://nlp.stanford.edu/software/stanford-parser-full-2013-11-12.zip'
        elif version == '4.2.0':
            sp_home = 'https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip'
        else:
            raise ValueError(f'Unsupported version {version}')
        sp_home = io_util.get_resource(sp_home)
        if ud:
            jclass = (
                'edu.stanford.nlp.trees.international.pennchinese.UniversalChineseGrammaticalStructure'
                if language == 'zh'
                else 'edu.stanford.nlp.trees.ud.UniversalDependenciesConverter'
            )
        else:
            jclass = (
                'edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure'
                if language == 'zh'
                else 'edu.stanford.nlp.trees.EnglishGrammaticalStructure'
            )
        cmd = f'java -cp {Path(sp_home).as_posix()}/* {jclass} -treeFile "{Path(src).as_posix()}"'
        if conllx:
            cmd += ' -conllx'
        if not ud:
            cmd += ' -basic -keepPunct'
        code, out, err = safe_get_exitcode_stdout_stderr(cmd)
        with open(dst, 'w', encoding='utf-8', newline='\n') as file:
            file.write(out)
        if code:
            raise RuntimeError(
                f'Conversion failed with code {code} for {src}. The err message is:\n{err}\n'
                f'Do you have java installed? Do you have enough memory?'
            )

    def safe_make_ctb_tasks(chtbs: list[str], out_root: str, part: str) -> None:
        for task in ['cws', 'pos', 'par', 'dep']:
            os.makedirs(Path(out_root) / task, exist_ok=True)
        timer = ctb_utils.CountdownTimer(len(chtbs))
        par_path = Path(out_root) / 'par' / f'{part}.txt'
        with open(Path(out_root) / 'cws' / f'{part}.txt', 'w', encoding='utf-8', newline='\n') as cws, \
                open(Path(out_root) / 'pos' / f'{part}.tsv', 'w', encoding='utf-8', newline='\n') as pos, \
                open(par_path, 'w', encoding='utf-8', newline='\n') as par:
            for file_path in chtbs:
                with open(file_path, encoding='utf-8') as src:
                    content = src.read()
                    trees = ctb_utils.split_str_to_trees(content)
                    for tree in trees:
                        try:
                            tree = ctb_utils.Tree.fromstring(tree)
                        except ValueError:
                            print(tree)
                            raise
                        words = []
                        for word, tag in tree.pos():
                            if tag == '-NONE-' or not tag:
                                continue
                            tag = tag.split('-')[0]
                            if tag == 'X':
                                tag = 'FW'
                            pos.write(f'{word}\t{tag}\n')
                            words.append(word)
                        cws.write(' '.join(words))
                        par.write(tree.pformat(margin=1_000_000))
                        for fp in (cws, pos, par):
                            fp.write('\n')
                timer.log(
                    f'Preprocesing the [blue]{part}[/blue] set of CTB [blink][yellow]...[/yellow][/blink]',
                    erase=False,
                )
        safe_remove_all_ec(str(par_path))
        dep_path = Path(out_root) / 'dep' / f'{part}.conllx'
        safe_convert_to_dependency(str(par_path), str(dep_path))
        sents = list(ctb_utils.read_conll(str(dep_path)))
        with open(dep_path, 'w', encoding='utf-8', newline='\n') as out:
            for sent in sents:
                for cells in sent:
                    tag = cells[3].split('-')[0]
                    if tag == 'X':
                        tag = 'FW'
                    cells[3] = cells[4] = tag
                    out.write('\t'.join(str(x) for x in cells))
                    out.write('\n')
                out.write('\n')

    ctb_utils.remove_all_ec = safe_remove_all_ec
    ctb_utils.convert_to_dependency = safe_convert_to_dependency
    ctb_utils.make_ctb_tasks = safe_make_ctb_tasks


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
