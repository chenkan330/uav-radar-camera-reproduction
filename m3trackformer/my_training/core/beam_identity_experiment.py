"""V5: beam-contribution identity audit on an unchanged Huber N-Net/R-Net baseline."""
import argparse
import hashlib
from pathlib import Path
import shutil

import torch

from .beam_identity import IdentityAudit, METHODS, DEFAULT_GATE, pool_summaries
from .beam_identity_charts import render_budget, render_comparison
from .common import BASE, load_nnet, new_run, read_json, state_digest, write_json, write_rows
from .maneuver_nnet_rnet import ORIGINAL_MODELS_SHA256, run_original_nr
from .models import JournalRNet
from .multi_target import summarize
from .optimization_experiment import TEST_SEEDS, aggregate, prepare_scenes, sha, load_tracking_choice
from .real_data_audit import audit_real_reference
from .trajectory_figure import save_trajectory_figure
from .versions import save_version


def report(root, records):
    percent = lambda value: 'N/A' if value is None else f'{100*value:.2f}%'
    lines = ['# V5：按波束评分贡献判别身份', '',
        '当前主线采用用户指定的Huber基准：阈值3、加速度标准差12，原Transformer N-Net/R-Net结构、权重和选束公式不变。', '',
        '## 你的方案和改进版', '',
        '1. 原网络逐目标输出波束概率 p_i(b)，总评分 S(b)=sum_i p_i(b)，选总评分最高的B束。',
        '2. 你的规则：匿名回波来自波束b，判给argmax_i p_i(b)。同束多个回波会被判给同一目标。',
        '3. 改进规则：保留相对贡献q_i(b)=p_i(b)/sum_j p_j(b)，结合过去Huber位置/速度及协方差预测的三维创新距离；门控后进行全局一对一匹配，允许拒配。',
        f'4. 参数在测试前固定：门控NIS≤{DEFAULT_GATE:.8f}；融合代价=0.5×NIS−log(q)；拒配代价20。名义门控对应高斯三维99.9%，不是含异常值数据的实测覆盖率。',
        '5. 融合代价是工程评分，q不是经过身份标签校准的后验概率；没有加入数据集不存在的多普勒、外观或目标发射编号。', '',
        '## 结果口径', '',
        '本版是条件身份判别审计。旧Huber闭环依然使用理想身份关联，维持共同的过去历史；两种判别规则仅获得相同的已检出匿名点、当前N/R概率和过去Huber状态。',
        '真实目标索引只保留在模拟传感器/事后评价适配器，不传入associate匹配函数。身份判断不反馈Huber或网络，不声称完成未知身份闭环或解决身份错误累积。',
        '跟踪指标属于Huber共同参照，不属于两种身份规则各自的闭环跟踪效果。原N/R位置曲线仍是实际位置头输出，未替换为滤波位置。', '',
        '每个预算分别使用相同17组真实GNSS轨迹组合、3组固定噪声；不是51个独立场景。真实GNSS是参考估计，雷达测量和异常值为模拟，多目标由不同实测片段组合。',
        '当前网络只负责雷达波束跟踪目标；数据来源是无人机飞行记录，不代表同时实现多个感知无人机的协作系统。', '',
        '|共享波束数|规则|全部点正确归属率（正确/全部）|已分配精度|分配覆盖率|同束混叠点正确率|归属切换次数|',
        '|---:|---|---:|---:|---:|---:|---:|']
    for record in records:
        for method, name in METHODS.items():
            m = record['identity_metrics'][method]
            lines.append(f'|{record["total_beams"]}|{name}|{percent(m["accuracy_all_returns"])} '
                f'({m["correct"]}/{m["total"]})|{percent(m["precision_assigned"])}|{percent(m["assignment_coverage"])}|'
                f'{percent(m["accuracy_same_beam"])}|{m["identity_switches"]}|')
    lines += ['', '拒配在“全部点正确归属率”的分母中保留，不因少分配而抬高该指标。',
        '归属切换从初始化正确身份开始，统计同一真实目标连续已分配判断的标签变化，跨拒配间隙保留上次标签；只是本审计流的指标，不是完整未知身份轨迹管理的IDSW。', '',
        '|共享波束数|Huber共同参照完整跟踪成功率|平均同时跟踪时间/s|重捕率|同束混叠检出点数/全部|',
        '|---:|---:|---:|---:|---:|']
    for record in records:
        m = record['tracking_reference']; identity = record['identity_metrics']['beam_argmax']
        lines.append(f'|{record["total_beams"]}|{percent(m["success_rate"])}|{m["mean_all_targets_duration_s"]:.2f}|'
            f'{percent(m["recovery_rate"])}|{identity["collision_total"]}/{identity["total"]}|')
    lines += ['', '不同预算的检出点数和场景存活时间不同，不可把条件分类精度直接当成总体跟踪优劣。',
        '同束多回波仍假设距离/角度检测可分辨；没有模拟真实回波峰合并、虚警或额外随机漏检。异常值是已有点偏移，并非新增杂波点。',
        '固定首场景首目标的N-Net轨迹图、逐帧身份时间图只展示首个噪声种子；汇总表包括三种子。',
        'beam_contributions.png优先展示首个同束多回波时刻（纯展示），未根据身份正确率挑选；若无此时刻展示首帧。', '',
        '## 下一步', '',
        '先审查身份误判、拒配和同束重叠事件，再将实际关联测量接入Huber和原N/R历史，验证错误累积。',
        '接回闭环前必须定义“确认未检出”和“归属不确定”两种反馈；身份不确定不能直接作为R-Net失败波束证据。',
        '随后才考虑短窗口多假设、目标出现/消失和新增杂波，不同时改动这些条件。', '',
        '## 复现', '', '```powershell',
        r'.\.venv\Scripts\python.exe .\my_training\step10_single_target_tracking.py --beam-identity',
        '```', '', '--total-beams 4 或 --total-beams 12可只跑一个预算；默认两者都跑。--no-show用于后台。',
        '数据来源、权重、网络调用审计、固定协议和逐检测原始数值都保存在本次版本。旧版本不覆盖。', '',
        '方法参考：[全局关联与门控](https://www.mathworks.com/help/fusion/ug/introduction-to-assignment-methods-in-tracking-systems.html)、'
        '[线性和分配求解器](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html)。', '']
    (root/'identity_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--total-beams', type=int, nargs='+', default=[4, 12])
    parser.add_argument('--output-root', type=Path, default=BASE/'outputs/step10_beam_identity')
    args = parser.parse_args(argv)
    if any(b <= 3 or b > 1024 for b in args.total_beams) or len(set(args.total_beams)) != len(args.total_beams):
        parser.error('三目标共享预算须大于3且不超过1024，不可重复')
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    else:
        import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    current = read_json(BASE/'outputs/current_journal_experiment.json')
    choice, choice_lock = load_tracking_choice()
    if choice['configuration']['id'] != 'huber_base':
        raise ValueError('本次身份审计固定使用用户指定的当前Huber基准')
    hashes = {k: sha(current[k]) for k in ('nnet_checkpoint', 'rnet_checkpoint')}
    if hashes != choice['checkpoint_sha256'] or sha(Path(__file__).with_name('models.py')) != ORIGINAL_MODELS_SHA256:
        raise RuntimeError('原N/R模型或权重发生变化')
    device = torch.device(args.device)
    nnet, checkpoint = load_nnet(current['nnet_checkpoint'], device)
    rnet = JournalRNet(nnet.config).to(device).eval().requires_grad_(False)
    rnet.load_state_dict(torch.load(current['rnet_checkpoint'], map_location=device, weights_only=False)['model'])
    before = state_digest(nnet), state_digest(rnet)
    root = new_run(args.output_root)
    protocol = dict(version='beam_identity_audit', targets=3, total_beam_budgets=args.total_beams,
        noise_seeds=TEST_SEEDS, scene_seed=20261115, huber_threshold=3., acceleration_std=12.,
        external_temperature=1., gate=DEFAULT_GATE, beam_weight=1., unmatched_cost=20.,
        parameter_selection='fixed_before_test_without_test_tuning',
        scope='conditional_identity_decisions_on_oracle_associated_Huber_histories',
        association_decisions_feed_history=False, models_source_sha256=ORIGINAL_MODELS_SHA256,
        checkpoint_sha256=hashes, operating_choice_lock_sha256=sha(choice_lock))
    write_json(root/'protocol.json', protocol)
    data_root = Path(current['data'])
    audit_real_reference(data_root, 'test', root/'test_reference')
    variants, unused = prepare_scenes(data_root, 'test', TEST_SEEDS)
    records, paths = [], []
    for budget in args.total_beams:
        out = root/f'beams_{budget}'; out.mkdir()
        audits, repetitions, primary = [], [], None
        for seed, (scenes, noise_metadata) in variants.items():
            observer = IdentityAudit(scenes['trajectory_id'], seed, nnet.config.history, .05)
            plot_id = int(scenes['trajectory_id'][0,0])
            result, trace, _, timing, _, calls = run_original_nr(nnet, rnet, checkpoint, scenes, device,
                quality_mode='robust', dt=.05, budget=budget, plot_id=plot_id,
                record_events=False, collect_probability=False, collect_updates=False,
                detection_observer=observer)
            identity = observer.summary()
            audits.append(identity)
            repetitions.append(dict(noise_seed=seed, scene_rows=result[0], target_rows=result[1],
                filter_timing=timing, network_calls=calls, identity_metrics=identity,
                measurement_sha256=hashlib.sha256(scenes['measurement'].tobytes()).hexdigest()))
            write_rows(out/f'identity_detection_samples_{seed}.csv', observer.rows)
            write_rows(out/f'scenes_{seed}.csv', result[0]); write_rows(out/f'targets_{seed}.csv', result[1])
            write_json(out/f'beam_contribution_snapshot_{seed}.json', observer.collision_snapshot or observer.snapshot)
            if primary is None:
                primary = observer, result, trace, scenes, plot_id
            print(f'{budget} beams seed={seed}: '+', '.join(f'{k}={100*v["accuracy_all_returns"]:.2f}% ({v["correct"]}/{v["total"]})'
                for k,v in identity.items()), flush=True)
        observer, result, trace, scenes, plot_id = primary
        maximum = scenes['beam'].shape[-1]-nnet.config.history
        m = aggregate(repetitions, maximum)
        record = dict(version='beam_identity_audit', protocol=protocol, total_beams=budget,
            source=str((data_root/'test.npz').resolve()), nnet_checkpoint=current['nnet_checkpoint'],
            rnet_checkpoint=current['rnet_checkpoint'], tracking_reference=m,
            identity_metrics=pool_summaries(audits), noise_repeats=repetitions,
            primary_plot_noise_seed=TEST_SEEDS[0], primary_plot_trajectory_id=plot_id,
            scene_membership=scenes['trajectory_id'].tolist(), unused_trajectory_ids=unused,
            models_source_sha256=ORIGINAL_MODELS_SHA256, networks_and_weights_unchanged=True,
            association_results_used_for_tracking=False,
            trajectory_figure='figures/single_trajectory_tracking.png')
        save_trajectory_figure(out, scenes, trace, result[1],
            summarize(result[0], result[1], result[4], .05, budget, 3, 'joint'),
            dt=.05, history=nnet.config.history, trajectory_id=plot_id,
            title=f'Huber基准：{budget}束；身份判别暂不反馈网络',
            selection_note='固定首场景首目标、首组噪声；本图是共同Huber参照，不是未知身份关联后的跟踪结果。')
        render_budget(out, record, observer, maximum, nnet.config.history, .05)
        for name in ('real_trajectory_sources.csv', 'real_trajectory_audit.json'):
            shutil.copy2(root/'test_reference'/name, out/name)
        shutil.copy2(choice_lock, out/'operating_choice_lock.json')
        write_json(out/'summary.json', record)
        records.append(record); paths.append(out)
    if (before != (state_digest(nnet), state_digest(rnet))
            or hashes != {k: sha(current[k]) for k in hashes}
            or sha(Path(__file__).with_name('models.py')) != ORIGINAL_MODELS_SHA256):
        raise RuntimeError('审计过程中原N/R结构或权重变化')
    # The passive observer must reproduce the previous 12-beam Huber reference.
    previous = read_json(Path(choice['experiment_root'])/'test/huber_base/summary.json')
    for r in records:
        if r['total_beams'] == 12:
            for new, old in zip(r['noise_repeats'], previous['noise_repeats']):
                for key in ('noise_seed', 'scene_rows', 'target_rows', 'network_calls', 'measurement_sha256'):
                    if new[key] != old[key]:
                        raise RuntimeError(f'被动审计改变了Huber基准：{key}')
            r['tracking_reference_exactly_matches_prior_huber'] = True
    render_comparison(root, records); report(root, records)
    for record, out in zip(records, paths):
        write_json(out/'summary.json', record)
        for name in ('identity_report.md','identity_comparison.png','protocol.json'):
            shutil.copy2(root/name, out/name)
        prefix = 'v5_'+root.name.removeprefix('run_').removesuffix('_UTC')
        archive = save_version(f'{prefix}_beam_identity_b{record["total_beams"]}', out,
            '用户波束评分贡献身份方案与评分/几何门控全局匹配；原Huber历史条件审计，不反馈网络。')
        record['archive_path'] = str(archive.resolve())
        print(f'已保存：{archive.resolve()}', flush=True)
    write_json(root/'summary.json', dict(protocol=protocol, results=[dict(total_beams=r['total_beams'],
        identity_metrics=r['identity_metrics'], tracking_reference=r['tracking_reference'],
        archive_path=r['archive_path']) for r in records], all_networks_and_weights_unchanged=True))
    (args.output_root/'latest_complete_run.txt').write_text(root.name, encoding='utf-8')
    if not args.no_show:
        from .optimization_charts import show_images
        display = next((p for p,r in zip(paths,records) if r['total_beams']==12),paths[0])
        show_images([display/'figures/single_trajectory_tracking.png',root/'identity_comparison.png'])
    print(f'身份判别结果：{root.resolve()}', flush=True)
    return root
