"""V6: validation-selected anonymous identity tracking with the original N/R."""
import argparse
import hashlib
from pathlib import Path
import shutil

import numpy as np
import torch

from .beam_identity import DEFAULT_GATE
from .common import BASE,load_nnet,new_run,read_json,state_digest,write_json,write_rows
from .identity_closed_loop import (RecordedRadar,simulate_anonymous,evaluate,aggregate_association,
                                  physical_reception_metrics,aggregate_physical)
from .identity_closed_loop_charts import trial_figures,comparison
from .maneuver_nnet_rnet import ORIGINAL_MODELS_SHA256
from .models import JournalRNet
from .optimization_experiment import VALIDATION_SEEDS,TEST_SEEDS,prepare_scenes,aggregate,load_tracking_choice,sha
from .real_data_audit import audit_real_reference
from .trajectory_figure import save_trajectory_figure
from .versions import save_version

CANDIDATES=[dict(id='v5_gate',label='闭环：原V5门限',parameters=dict(gate=DEFAULT_GATE,beam_weight=1.,unmatched_cost=20.)),
    dict(id='gate36',label='闭环：门限36',parameters=dict(gate=36.,beam_weight=1.,unmatched_cost=32.)),
    dict(id='gate64',label='闭环：门限64',parameters=dict(gate=64.,beam_weight=1.,unmatched_cost=48.))]
CRITERIA=dict(max_success_loss=.05,min_duration_retention=.95,min_assigned_precision=.995)


def selection_key(record):
    a,m=record['association'],record['identity_tracking']
    return ((a['precision_assigned'] or 0)>=CRITERIA['min_assigned_precision'],m['success_rate'],
            m['mean_all_targets_duration_s'],a['precision_assigned'] or 0,a['correct_update_coverage'])


def acceptance(metrics,association,reference):
    checks=dict(success_retention=metrics['success_rate']>=reference['success_rate']-CRITERIA['max_success_loss']-1e-12,
        duration_retention=metrics['mean_all_targets_duration_s']>=reference['mean_all_targets_duration_s']*CRITERIA['min_duration_retention'],
        assigned_precision=(association['precision_assigned'] or 0)>=CRITERIA['min_assigned_precision'])
    return dict(passed=all(checks.values()),checks=checks,criteria=CRITERIA,
        label='project_stage_criteria_not_an_industry_or_paper_standard')


def run_trial(root,configuration,phase,variants,current,nnet,rnet,checkpoint,device,reference):
    out=root/phase/configuration['id'];out.mkdir(parents=True)
    records=[];primary=None
    for seed,(scenes,metadata) in variants.items():
        sensor=RecordedRadar(scenes['measurement'],scenes['measurement_sigma'],scenes['beam']-1,seed)
        h=nnet.config.history;steps=scenes['beam'].shape[-1]
        events,timing,calls=simulate_anonymous(nnet,rnet,checkpoint,scenes['measurement'][...,:h,:],
            scenes['measurement_sigma'][...,:h,:],sensor,steps,device,.05,12,configuration['parameters'])
        scores,events=evaluate(events,sensor,scenes,h,.05,nnet.config.failure_limit)
        rec=dict(noise_seed=seed,**scores,filter_timing=timing,network_calls=calls,
            measurement_sha256=hashlib.sha256(scenes['measurement'].tobytes()).hexdigest(),
            physical_reception=physical_reception_metrics(events,scenes,h))
        records.append(rec)
        write_rows(out/f'association_events_{seed}.csv',events)
        write_rows(out/f'identity_scenes_{seed}.csv',scores['scene_rows'])
        write_rows(out/f'identity_targets_{seed}.csv',scores['target_rows'])
        write_rows(out/f'controller_scenes_{seed}.csv',scores['controller_scene_rows'])
        if primary is None:primary=(scenes,events,scores)
        good=sum(r['success'] for r in scores['scene_rows'])
        assoc=scores['association'];precision=assoc['correct']/assoc['assigned'] if assoc['assigned'] else 0
        print(f'{phase} {configuration["id"]} seed={seed}: identity_complete={good}/{len(scores["scene_rows"])}; accepted_precision={precision:.6f}',flush=True)
    maximum=steps-h
    metrics=aggregate(records,maximum)
    controller=aggregate([{**r,'scene_rows':r['controller_scene_rows'],'target_rows':r['controller_target_rows']} for r in records],maximum)
    metrics['filter_ms_per_detected_frame']=controller['filter_ms_per_detected_frame']
    association=aggregate_association(records,maximum,3)
    record=dict(version='identity_closed_loop',phase=phase,configuration=configuration,
        identity_tracking=metrics,controller_tracking=controller,association=association,noise_repeats=records,
        physical_reception=aggregate_physical(records),
        acceptance=acceptance(metrics,association,reference),ideal_huber_reference=reference,
        source=str(Path(current['data'])/f'{phase}.npz'),nnet_checkpoint=current['nnet_checkpoint'],rnet_checkpoint=current['rnet_checkpoint'],
        total_beams=12,targets=3,dt=.05,history_frames=h,
        models_source_sha256=ORIGINAL_MODELS_SHA256,networks_and_weights_unchanged=True,
        association_results_used_for_tracking=True,initialization='first_five_correctly_grouped_acquired_measurements_only',
        evaluation_truth_controls_online_stopping=False,negative_evidence='empty_sensor_channels_only; occupied_channels_are_padding',
        trajectory_figure='figures/single_trajectory_tracking.png',primary_plot_noise_seed=next(iter(variants)),
        metric_scope='identity duration stops offline after any original identity has five consecutive noncorrect updates; controller stops only after five unavailable updates',
        scene_membership=primary[0]['trajectory_id'].tolist())
    scenes,events,scores=primary
    plot_id=int(scenes['trajectory_id'][0,0]);trace=[]
    from .measurement_quality import spherical_to_cartesian
    for k in range(h):
        xyz=spherical_to_cartesian(scenes['measurement'][0,0,k])
        trace.append(dict(scene_id=0,target_index=0,trajectory_id=plot_id,time_index=k,time_s=(k-h)*.05,
            phase='initialization',hit=True,recovered=False,identity_correct=True,
            assigned_source_evaluation_only=0,estimate_source='initial_measurement',
            **{f'estimated_{a}_m':float(xyz[j]) for j,a in enumerate('xyz')}))
    trace += [dict(e) for e in events if e['scene_id']==0 and e['target_index']==0]
    cm=aggregate([{**records[0],'scene_rows':scores['controller_scene_rows'],'target_rows':scores['controller_target_rows']}],maximum)
    cm['scenes']=len(scores['controller_scene_rows'])
    strict_time=scores['scene_rows'][0]['duration_s']
    save_trajectory_figure(out,scenes,trace,scores['controller_target_rows'],cm,dt=.05,history=h,trajectory_id=plot_id,
        title='匿名关联闭环：原 N-Net/R-Net实际输出',
        selection_note=f'固定首场景首目标、首组噪声；该场景严格身份有效时长{strict_time:.2f}秒。\n本图保留控制器实际输出，错配也如实绘出；总体身份指标见比较图。')
    trial_figures(out,record,events,steps,h,.05)
    for name in ('real_trajectory_sources.csv','real_trajectory_audit.json'):
        shutil.copy2(root/(phase+'_reference')/name,out/name)
    shutil.copy2(root/'protocol.json',out/'protocol.json')
    write_json(out/'summary.json',record)
    prefix='v6_'+root.name.removeprefix('run_').removesuffix('_UTC')
    archive=save_version(f'{prefix}_{phase}_{configuration["id"]}',out,
        '匿名关联真实接入Huber和原N/R；真值仅事后评价；只编码空回波波束负证据；保存严格身份与控制器两套指标。')
    return record,archive


def write_report(root,records,reference,selected,baseline_origin):
    fmt=lambda v:'N/A' if v is None else f'{100*v:.2f}%'
    lines=['# V6：匿名身份关联接入真实闭环','','原Transformer N-Net/R-Net的结构、权重、接口、N/R分工保持；Huber阈值3、加速度标准差12，逐目标Softmax求和Top-12保持。',
        '本版不再由真实身份维持历史。初始化5帧之后，只将匿名匹配得到的实际测量与其名义协方差送入对应轨迹Huber，再进入原N/R历史；不补造缺测。',
        '有回波但未关联与全扫描无回波分别记录。不可用历史仍触发原R-Net；失败波束仅编码本帧确认无任何回波的通道，含回波的波束一律不冒充失败证据。',
        '评价器在整个控制器运行结束后才读取真实来源/位置/异常标签。它不决定在线终止，也不修复在线身份。', '',
        '## 指标定义与验收','','主指标为“身份有效”：任一初始化身份连续5帧没有收到属于自己的正确关联，事后有效计时结束。错身份持续收点不能虚增此指标。',
        '另外保留可用测量控制器时长：任一轨迹连续5帧无可用测量才停止。轨迹图保留这期间全部实际N-Net输出，错配点用紫菱形，不隐去错误。',
        '身份有效重捕率统计没有正确关联的事件恢复到正确关联的比例；与只看可用更新的控制器重捕率分别报告。',
        '阶段门槛在测试前设定：完整成功率相对理想关联参照降低≤5个百分点，平均身份有效时长保留≥95%，已接受关联精度≥99.5%。这是本项目阶段门槛，不是论文或行业标准。',
        f'门限候选仅由验证集排序，测试前锁定。本报告汇总{len(records[0]["noise_repeats"][0]["scene_rows"])}组场景×{len(records[0]["noise_repeats"])}组噪声；噪声重复不是新增独立场景，记录此前已使用，不是新盲测。', '',
        '|方案|身份有效完整成功率|平均身份有效时长/s|物理漏检重捕率|已接受关联精度|正确更新覆盖率|跨缺测归属切换|阶段通过|',
        '|---|---:|---:|---:|---:|---:|---:|---|',
        f'|理想关联Huber参照|{fmt(reference["success_rate"])}|{reference["mean_all_targets_duration_s"]:.2f}|{fmt(reference["recovery_rate"])}|100%|{fmt(reference["position_evaluation_coverage"])}|0|参照|']
    for r in records:
        m,a=r['identity_tracking'],r['association']
        lines.append(f'|{r["configuration"]["label"]}{"（验证集选择）" if r["configuration"]["id"]==selected else ""}|'
            f'{fmt(m["success_rate"])}|{m["mean_all_targets_duration_s"]:.2f}|{fmt(r["physical_reception"]["recovery_rate"])}|'
            f'{fmt(a["precision_assigned"])}|{fmt(a["correct_update_coverage"])}|{a["identity_switches"]}|{r["acceptance"]["passed"]}|')
    lines += ['', '## 重捕统计与剩余问题','',
        '表中的物理漏检重捕沿用旧定义：实际扫描未得到某个物理目标的回波后，再次收到它的回波。来源只用于事后计分。',
        '身份有效重捕还包括几何拒配和短暂错配后的恢复，事件分母大很多；不能将此约99%的数值解读为R-Net相对84%的直接提升。']
    for r in records:
        m,a,p=r['identity_tracking'],r['association'],r['physical_reception']
        lines.append(f'{r["configuration"]["label"]}：物理恢复{p["recoveries"]}/{p["lost_events"]}={fmt(p["recovery_rate"])}；'
            f'身份有效恢复{m["recoveries"]}/{m["lost_events"]}={fmt(m["recovery_rate"])}；'
            f'实际错配更新{a["wrong"]}次；零错配且完整成功场景比例{fmt(a["zero_wrong_update_scene_success_rate"])}。')
    lines += ['', '阶段通过不等于全程零身份错误；平均时长接近95%保留门槛，不称效果完全无损。',
        '',f'理想关联参照来自相同数据/模型/噪声的既有Huber完整运行：{baseline_origin}。已核对测量数组SHA256、场景、模型和原始数据来源。',
        '真实GNSS参考轨迹与模拟雷达测量/SNR/异常值；仍假设同束多目标可分辨，没有加入新增杂波、随机漏检或出生消失。',
        '目前解决的是已初始化3目标的身份保持，不是未知目标数量的完整航迹起始与删除。',
        '身份切换给出跨缺测保持上次已分配标签、以及仅连续帧变化两种计数，不与早期条件审计的计数混称为同一闭环指标。',
        '正常N-Net定位头仍为唯一位置输出；滤波预测只供输入处理与几何关联。RMSE评价帧集合不同，须结合正确更新覆盖率理解。', '',
        '## 运行','','```powershell',r'.\.venv\Scripts\python.exe .\my_training\step10_single_target_tracking.py --identity-closed-loop --apply-selected',
        '```','','复用验证集锁定配置、首组噪声，保存全新版本，显示轨迹和身份指标图。完整重跑去掉--apply-selected；后台加--no-show。',
        '若未达到阶段门槛，下一步先分析连续拒配、错配或负证据不足，不直接增加新复杂条件；达标后才增加近距离交会/短时缺测，继续逐项验证。','']
    (root/'closed_loop_report.md').write_text('\n'.join(lines),encoding='utf-8')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true');parser.add_argument('--apply-selected',action='store_true')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-root',type=Path,default=BASE/'outputs/step10_identity_closed_loop')
    parser.add_argument('--review-run',type=Path,help='保留原仿真，另建版本补充物理重捕统计和图表')
    args=parser.parse_args(argv)
    import matplotlib
    if args.no_show:matplotlib.use('Agg')
    else:
        import matplotlib.pyplot as plt
    if args.review_run is not None:
        return review_saved(args.review_run,args.output_root)
    torch.set_num_threads(4)
    current=read_json(BASE/'outputs/current_journal_experiment.json');choice,_=load_tracking_choice()
    hashes={k:sha(current[k]) for k in ('nnet_checkpoint','rnet_checkpoint')}
    if choice['configuration']['id']!='huber_base' or hashes!=choice['checkpoint_sha256'] or sha(Path(__file__).with_name('models.py'))!=ORIGINAL_MODELS_SHA256:
        raise RuntimeError('须使用固定Huber及原N/R结构权重')
    device=torch.device(args.device);nnet,checkpoint=load_nnet(current['nnet_checkpoint'],device)
    rnet=JournalRNet(nnet.config).to(device).eval().requires_grad_(False)
    rnet.load_state_dict(torch.load(current['rnet_checkpoint'],map_location=device,weights_only=False)['model'])
    before=state_digest(nnet),state_digest(rnet)
    root=new_run(args.output_root);data_root=Path(current['data']);baseline_root=Path(choice['experiment_root'])
    protocol=dict(version='identity_closed_loop',candidates=CANDIDATES,criteria=CRITERIA,
        validation_seeds=VALIDATION_SEEDS,test_seeds=TEST_SEEDS,models_source_sha256=ORIGINAL_MODELS_SHA256,
        checkpoint_sha256=hashes,reference_root=str(baseline_root),targets=3,total_beams=12,
        missing_limit=5,initialization_frames=5,test_records_previously_used=True,
        negative_evidence='only selected channels with no sensor returns',no_oracle_identity_after_initialization=True,
        ranking_order=['validation_precision_at_least_99p5','identity_success_rate','identity_duration','precision','correct_coverage'])
    write_json(root/'protocol.json',protocol)
    pointer_file=BASE/'outputs/current_identity_closed_loop.json'
    if args.apply_selected:
        pointer=read_json(pointer_file);lock=Path(pointer['selection_record'])
        if sha(lock)!=pointer['selection_lock_sha256'] or hashes!=pointer['checkpoint_sha256']:
            raise RuntimeError('闭环选择记录或权重校验不匹配')
        configuration=read_json(lock)['configuration'];selected=configuration['id']
        phases=[('test',[TEST_SEEDS[0]],[configuration])];selection_path=lock
        selection_digest=sha(selection_path)
    else:
        phases=[('validation',VALIDATION_SEEDS,CANDIDATES)]
    validation=[];tests=[];archives=[];selected=None if not args.apply_selected else selected
    phase_index=0
    while phase_index<len(phases):
        phase,seeds,configurations=phases[phase_index]
        audit_real_reference(data_root,phase,root/(phase+'_reference'))
        variants,_=prepare_scenes(data_root,phase,seeds)
        reference_record=read_json(baseline_root/phase/'huber_base/summary.json')
        phase_reference=aggregate([r for r in reference_record['noise_repeats'] if r['noise_seed'] in seeds],
            next(iter(variants.values()))[0]['beam'].shape[-1]-nnet.config.history)
        for seed,(scenes,_) in variants.items():
            previous=next(r for r in reference_record['noise_repeats'] if r['noise_seed']==seed)
            if hashlib.sha256(scenes['measurement'].tobytes()).hexdigest()!=previous['measurement_sha256']:
                raise RuntimeError('与Huber参照的测量不相同')
        for configuration in configurations:
            record,archive=run_trial(root,configuration,phase,variants,current,nnet,rnet,checkpoint,device,phase_reference)
            archives.append((archive,phase))
            (validation if phase=='validation' else tests).append(record)
            write_json(root/(phase+'_progress.json'),[dict(configuration=r['configuration'],identity_tracking=r['identity_tracking'],association=r['association'])
                for r in (validation if phase=='validation' else tests)])
        if phase=='validation':
            winner=max(validation,key=selection_key);selected=winner['configuration']['id']
            selection_path=root/'selection_locked_before_test.json'
            write_json(selection_path,dict(configuration=winner['configuration'],validation_tracking=winner['identity_tracking'],
                validation_association=winner['association'],selected_before_test=True,protocol_sha256=sha(root/'protocol.json')))
            selection_digest=sha(selection_path)
            finalist={CANDIDATES[0]['id']:CANDIDATES[0],selected:winner['configuration']}
            phases.append(('test',TEST_SEEDS,list(finalist.values())))
            print(f'验证集锁定闭环配置：{selected}',flush=True)
        phase_index+=1
    if before!=(state_digest(nnet),state_digest(rnet)) or hashes!={k:sha(current[k]) for k in hashes} or sha(Path(__file__).with_name('models.py'))!=ORIGINAL_MODELS_SHA256:
        raise RuntimeError('原N/R结构或权重变化')
    if sha(selection_path)!=selection_digest:
        raise RuntimeError('测试阶段改变了验证集选择记录')
    reference=phase_reference
    comparison(root,tests,reference,selected);write_report(root,tests,reference,selected,str(baseline_root))
    chosen=next(r for r in tests if r['configuration']['id']==selected)
    result=dict(protocol=protocol,selected_configuration=chosen['configuration'],selection_record=str(selection_path.resolve()),
        selection_lock_sha256=sha(selection_path),test_results=[dict(configuration=r['configuration'],identity_tracking=r['identity_tracking'],
            controller_tracking=r['controller_tracking'],association=r['association'],physical_reception=r['physical_reception'],acceptance=r['acceptance']) for r in tests],
        selected_acceptance=chosen['acceptance'],networks_and_weights_unchanged=True,
        replay_single_seed=args.apply_selected)
    write_json(root/'summary.json',result)
    from tools.verify_versions import lock_figures
    for archive,phase in archives:
        if phase!='test':continue
        extra=archive/'figures';extra.mkdir()
        for name in ('closed_loop_comparison.png','closed_loop_report.md','summary.json'):
            shutil.copy2(root/name,extra/name)
        shutil.copy2(selection_path,extra/'selection_locked_before_test.json');lock_figures(archive)
    if not args.apply_selected:
        selected_archive=next(str(a.resolve()) for a,p in archives if p=='test' and a.name.endswith('_'+selected))
        write_json(pointer_file,dict(configuration=chosen['configuration'],selection_record=str(selection_path.resolve()),
            selection_lock_sha256=sha(selection_path),checkpoint_sha256=hashes,models_source_sha256=ORIGINAL_MODELS_SHA256,
            acceptance=chosen['acceptance'],experiment_root=str(root.resolve()),version_archive=selected_archive))
    (args.output_root/'latest_complete_run.txt').write_text(root.name,encoding='utf-8')
    print(f'闭环评估完成：{root.resolve()}\n阶段门槛通过：{chosen["acceptance"]["passed"]}',flush=True)
    if not args.no_show:
        from .optimization_charts import show_images
        show_images([root/'test'/selected/'figures/single_trajectory_tracking.png',root/'closed_loop_comparison.png'])
    return root


def review_saved(original_root,output_root):
    """Create a new review snapshot; retain old CSVs and all primary metrics."""
    import csv
    original_root=Path(original_root).resolve();original=read_json(original_root/'summary.json')
    root=new_run(output_root);records=[];archives=[]
    for name in ('protocol.json','selection_locked_before_test.json'):
        shutil.copy2(original_root/name,root/name)
    selected=original['selected_configuration']['id']
    for saved in original['test_results']:
        configuration=saved['configuration'];source=original_root/'test'/configuration['id']
        out=root/'test'/configuration['id'];shutil.copytree(source,out)
        record=read_json(source/'summary.json')
        variants,_=prepare_scenes(Path(record['source']).parent,'test',[r['noise_seed'] for r in record['noise_repeats']])
        primary_events=None
        for repeat in record['noise_repeats']:
            seed=repeat['noise_seed'];scenes=variants[seed][0]
            if hashlib.sha256(scenes['measurement'].tobytes()).hexdigest()!=repeat['measurement_sha256']:
                raise RuntimeError('重捕诊断与原仿真测量不相同')
            with (source/f'association_events_{seed}.csv').open(encoding='utf-8-sig',newline='') as stream:
                events=list(csv.DictReader(stream))
            for e in events:
                for key in ('scene_id','target_index','time_index','assigned_source_evaluation_only'):
                    e[key]=int(e[key])
                for key in ('hit','identity_correct'):e[key]=e[key].lower()=='true'
                e['time_s']=float(e['time_s'])
            repeat['physical_reception']=physical_reception_metrics(events,scenes,record['history_frames'])
            if primary_events is None:primary_events=events
        record['physical_reception']=aggregate_physical(record['noise_repeats'])
        record['review_revision']=dict(source_run=str(source),source_summary_sha256=sha(source/'summary.json'),
            controller_simulation_repeated=False,primary_tracking_and_association_metrics_unchanged=True,
            purpose='补充原定义物理漏检重捕，避免将拒配恢复与R-Net性能改善混淆')
        trial_figures(out,record,primary_events,variants[next(iter(variants))][0]['beam'].shape[-1],record['history_frames'],.05)
        records.append(record)
    reference=records[0]['ideal_huber_reference']
    comparison(root,records,reference,selected);write_report(root,records,reference,selected,original['protocol']['reference_root'])
    result={**original,'review_revision':True,'source_run':str(original_root),
        'test_results':[dict(configuration=r['configuration'],identity_tracking=r['identity_tracking'],
            controller_tracking=r['controller_tracking'],association=r['association'],physical_reception=r['physical_reception'],
            acceptance=r['acceptance']) for r in records]}
    write_json(root/'summary.json',result)
    for record in records:
        out=root/'test'/record['configuration']['id']
        write_json(out/'summary.json',record)
        for name in ('closed_loop_comparison.png','closed_loop_report.md'):
            shutil.copy2(root/name,out/name)
        prefix='v6_'+root.name.removeprefix('run_').removesuffix('_UTC')
        archive=save_version(f'{prefix}_test_{record["configuration"]["id"]}_review',out,
            '闭环结果复核版：不重算控制器、不改主指标和原始CSV；补充物理重捕与身份恢复的分母区别。')
        archives.append(archive)
        if record['configuration']['id']==selected:
            pointer=read_json(BASE/'outputs/current_identity_closed_loop.json')
            pointer['review_version_archive']=str(archive.resolve());pointer['review_experiment_root']=str(root.resolve())
            write_json(BASE/'outputs/current_identity_closed_loop.json',pointer)
        print(f'闭环复核版本：{archive.resolve()}',flush=True)
    (output_root/'latest_complete_run.txt').write_text(root.name,encoding='utf-8')
    return root
