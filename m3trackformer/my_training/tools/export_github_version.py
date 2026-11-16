"""Prepare reviewed public GitHub payloads; no credentials or network access."""
import argparse
import base64
import json
import re
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str) and re.search(r'[A-Za-z]:[\\/]', value):
        return '<LOCAL_ASSET>/' + value.replace('\\', '/').rsplit('/', 1)[-1]
    return value


def export(version):
    root = BASE/'experiments'/version
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    entries = {}
    with zipfile.ZipFile(root/'source.zip') as archive:
        for name in archive.namelist():
            entries['m3trackformer/'+name] = archive.read(name)
    prefix = f'm3trackformer/versions/{version}/'
    entries[prefix+'source.zip'] = (root/'source.zip').read_bytes()
    for sub in ('results', 'figures'):
        for path in (root/sub).rglob('*'):
            if not path.is_file() or path.name in {'tracking_events.csv', 'beam_scans.csv'}:
                continue
            if path.suffix not in {'.png', '.json', '.csv', '.txt'}:
                continue
            data = path.read_bytes()
            if path.suffix == '.json':
                data = json.dumps(portable(json.loads(data)), ensure_ascii=False, indent=2).encode('utf-8')
            elif path.suffix == '.csv':
                # Charts' source_run column uses local paths; strip those before publication.
                import csv, io
                reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
                buffer = io.StringIO(newline='')
                writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows({k: portable(v) for k, v in row.items()} for row in reader)
                data = buffer.getvalue().encode('utf-8-sig')
            entries[prefix+path.relative_to(root).as_posix()] = data
    public_manifest = portable(manifest)
    public_manifest['public_export_note'] = '原始飞行数据、权重、逐帧大文件及本机路径配置不上传；本地保留完整结果。source.zip保留源代码快照，external_assets用SHA256核对。'
    entries[prefix+'manifest.json'] = json.dumps(public_manifest, ensure_ascii=False, indent=2).encode('utf-8')
    entries['m3trackformer/AGENTS.md'] = (
        '# m³TrackFormer experiment instructions\n\n'
        'This subproject is independent of the repository root camera project. '
        'Use Python 3.13 and dependencies in requirements.txt. '
        'Keep numbered steps 1–10; internal modules live in my_training/core. '
        'Every new experiment version must preserve source and results via core.versions.save_version, '
        'then create a separate GitHub commit/version branch. Never overwrite frozen versions. '
        'Do not publish original flight datasets, weights, machine-specific path configurations or caches. '
        'Run checks/test_multi_target.py and relevant new checks. '
        'Use Chinese chart labels; report group-stop and detection-only RMSE definitions.\n'
    ).encode('utf-8')
    entries['m3trackformer/README.md'] = (
        '# m³TrackFormer 多目标实验\n\n'
        f'当前归档版本：**{version}**。所有旧版本在 `versions/` 中保留完整源代码 ZIP、指标及图片。\n\n'
        '本项目使用雷达距离/角度测量和 Transformer 波束跟踪。3目标共享12束，51条轨迹组成17场景，'
        '每场景49.8秒；任一目标连续漏检5次则整组停止。检测帧位置RMSE的样本覆盖不同，请结合成功率和时长判断。\n\n'
        '原始GNSS数据、训练权重和本机绝对路径配置未上传。下载后需准备已有正式版N-Net/R-Net权重及'
        '4个分区NPZ，使用 `--rnet-checkpoint`、`--data` 指定；R-Net checkpoint中引用的N-Net路径需指向本地权重。'
        '项目环境位于仓库工作目录的 `.venv` 时会自动选择。\n\n'
        '原始：`python my_training/step10_single_target_tracking.py --no-show`\n\n'
        '扰动未优化：同一命令加 `--measurement-regime stress`；鲁棒版再加 `--quality-mode robust`（V3起）。\n\n'
        '当前是轨迹驱动的代理测量仿真，理想身份关联和波束命中反馈仍保留，未声称复现论文全部物理回波或数值。'
        'V2/V3固定同一噪声种子及模型；鲁棒版只使用已命中的测量和接收机名义协方差，不使用异常真值标签。\n'
    ).encode('utf-8')
    out = BASE.parent/'_doc_work'/'github_exports'/version
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, (name, data) in enumerate(sorted(entries.items())):
        binary = name.endswith(('.zip', '.png'))
        row = {'path': name, 'encoding': 'base64' if binary else 'utf-8',
               'content': base64.b64encode(data).decode('ascii') if binary else data.decode('utf-8-sig')}
        payload = out/f'{i:03}.json'
        payload.write_text(json.dumps(row, ensure_ascii=True), encoding='ascii')
        rows.append({'index': i, 'path': name, 'payload': str(payload), 'bytes': len(data), 'binary': binary})
    (out/'index.json').write_text(json.dumps(rows, ensure_ascii=True), encoding='ascii')
    print(json.dumps({'directory': str(out), 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('version')
    export(parser.parse_args().version)
