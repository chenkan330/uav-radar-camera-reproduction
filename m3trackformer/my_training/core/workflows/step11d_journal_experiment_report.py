"""记录正式版核心对齐验证结果、参数/数据差异和相同失跟输入的离线对照。"""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from core.common import BASE, load_nnet, read_json, write_json
from core.models import JournalRNet


def paired_tracking(path):
    def rows(name):
        with (path/f'{name}.csv').open(encoding='utf-8-sig',newline='') as f:
            return {int(row['trajectory_id']):row for row in csv.DictReader(f)}
    a,b=rows('nnet_only'),rows('nnet_rnet')
    assert a.keys()==b.keys()
    rescued=[i for i in a if a[i]['success']=='False' and b[i]['success']=='True']
    harmed=[i for i in a if a[i]['success']=='True' and b[i]['success']=='False']
    result={'rescued_segment_ids':rescued,'newly_failed_segment_ids':harmed,
            'same_success_status':sum(a[i]['success']==b[i]['success'] for i in a)}
    events=path/'nnet_rnet_events.csv'
    if events.exists():
        with events.open(encoding='utf-8-sig',newline='') as f:
            trajectories={}
            for row in csv.DictReader(f):trajectories.setdefault(int(row['trajectory_id']),[]).append(row)
        recoveries=next_misses=0
        for seq in trajectories.values():
            seq.sort(key=lambda r:int(r['time_index']))
            for current,nxt in zip(seq,seq[1:]):
                if current['mode']=='R' and current['hit']=='True':
                    recoveries+=1
                    next_misses+=int(nxt['mode']=='N' and nxt['hit']=='False')
        result['recoveries_with_next_block_observed']=recoveries
        result['immediate_miss_after_return_to_nnet']=next_misses
    return result


def paired_lost_validation(nnet,rnet,path,device):
    with np.load(path) as d:
        data={k:d[k].copy() for k in ('history','valid','failed','count','label')}
    both=only_nnet=only_rnet=neither=0
    for offset in range(0,len(data['label']),256):
        h,v,f,c,y=[torch.from_numpy(data[k][offset:offset+256]).to(device) for k in ('history','valid','failed','count','label')]
        with torch.no_grad():
            a=nnet(h,v)['logits'][:,0]
            b=rnet(nnet,h,v,f,c)[:,0]
        ca=(a.topk(4,-1).indices==y[:,:1]).any(1)
        cb=(b.topk(4,-1).indices==y[:,:1]).any(1)
        both+=int((ca&cb).sum());only_nnet+=int((ca&~cb).sum());only_rnet+=int((~ca&cb).sum());neither+=int((~ca&~cb).sum())
    n=both+only_nnet+only_rnet+neither
    return {'count':n,'nnet_top4':(both+only_nnet)/n,'rnet_top4':(both+only_rnet)/n,
            'both_hit':both,'only_nnet_hit':only_nnet,'only_rnet_hit':only_rnet,'neither_hit':neither,
            'independent_test_evidence':False,'reason':'该失跟验证集也参与R-Net权重选择，仅作为诊断。'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-run',type=Path,required=True);p.add_argument('--test-run',type=Path,required=True)
    a=p.parse_args();test=read_json(a.test_run/'summary.json');val=read_json(a.validation_run/'summary.json')
    if test['split']!='test' or val['split']!='validation' or test['rnet_checkpoint']!=val['rnet_checkpoint']:
        p.error('必须比较同一组权重的验证/测试结果')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rpath=Path(test['rnet_checkpoint']);saved=torch.load(rpath,map_location=device,weights_only=False)
    npath=Path(saved['nnet_checkpoint']);nnet,checkpoint=load_nnet(npath,device)
    rnet=JournalRNet(nnet.config).to(device);rnet.load_state_dict(saved['model']);rnet.eval()
    lost=Path(saved['data']);collection=read_json(lost/'config.json')
    paired=paired_lost_validation(nnet,rnet,lost/'validation.npz',device)
    inventory={}
    for split in ('train','validation'):
        with np.load(lost/f'{split}.npz') as d:
            inventory[split]={'samples':len(d['label']),'unique_segments':len(np.unique(d['trajectory_id'])),
                              'unique_target_beams':len(np.unique(d['label'][d['label']>=0]))}
    source_files=('core/workflows/step05g_prepare_journal_dataset.py','core/models.py','core/common.py','core/plots.py',
                  'core/workflows/step07d_train_journal_nnet.py','core/workflows/step08d_collect_journal_lost.py','core/workflows/step09d_train_journal_rnet.py',
                  'step10d_journal_tracking.py','checks/test_journal_alignment.py')
    record={'target':'TWC_formal_single_target_core','fully_reproduced_all_paper_experiments':False,
            'nnet_training':read_json(npath.parent/'summary.json'),'nnet_best_epoch':checkpoint['epoch'],
            'nnet_best_validation':checkpoint['validation'],
            'rnet_training':read_json(rpath.parent/'summary.json'),'rnet_best_epoch':saved['epoch'],
            'model_config':nnet.config.dictionary(),'data':read_json(Path(checkpoint['data'])/'config.json'),
            'nnet_checkpoint':str(npath),'rnet_checkpoint':str(rpath),'collection':collection,'inventory':inventory,
            'paired_lost_validation':paired,'validation_closed_loop':val,'test_closed_loop':test,
            'paired_test_trajectories':paired_tracking(a.test_run),
            'alignment_audit':read_json(BASE/'outputs/journal_alignment_audit.json'),
            'source_sha256':{name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in source_files}}
    write_json(a.test_run/'experiment_record.json',record)
    write_json(BASE/'outputs/current_journal_experiment.json',
               {'status':'core_workflow_completed_NOT_full_paper_numerical_reproduction',
                'data':checkpoint['data'],'nnet_checkpoint':str(npath),'rnet_checkpoint':str(rpath),
                'validation_run':str(a.validation_run.resolve()),'test_run':str(a.test_run.resolve()),
                'report':str((a.test_run/'journal_training_report.txt').resolve()),
                'alignment_notes':str((BASE/'JOURNAL_ALIGNMENT.txt').resolve())})
    lines=['正式版单目标核心对齐验证记录',
           '注意：本次使用真实GPS参考轨迹+模拟雷达测量，不是原文训练集，不声称完整复现论文数值。',
           '论文及源码仍有未消歧项，详见my_training/JOURNAL_ALIGNMENT.txt。',
           f"N-Net：完成{record['nnet_training']['epochs_completed']}轮，最佳第{checkpoint['epoch']}轮，验证Top4={checkpoint['validation']['top4']:.2%}。",
           f"R-Net：完成{record['rnet_training']['epochs_completed']}轮，最佳第{saved['epoch']}轮，失跟验证Top4={saved['validation']['top4']:.2%}。",
           f"新失跟训练样本：{inventory['train']}；验证样本：{inventory['validation']}。",
           f"N-Net冻结校验：{record['rnet_training']['nnet_unchanged']}。",
           f"同一失跟验证输入：N-Net Top4={paired['nnet_top4']:.2%}，R-Net Top4={paired['rnet_top4']:.2%}；此验证集用于选权重，不是独立证据。",'']
    for label,results in [('验证',val),('测试',test)]:
        for mode,r in results['results'].items():
            lines.append(f"{label} {mode}：成功{r['successful_segments']}/{r['count']}={r['success_rate']:.2%}，平均跟踪{r['mean_duration_s']:.4f}秒，恢复{r['recoveries']}次，漏检{r['missed_blocks']}个时刻。")
    lines.extend(['',f"每段最大计分跟踪时间：{test['maximum_evaluated_duration_s']:.2f}秒；初始化窗口不计入。",
                  f"逐条测试对照：{record['paired_test_trajectories']}",
                  '恢复率的分母和失跟状态分布会随策略变化，不能只据恢复次数/恢复率增加认定长期跟踪改善。',
                  '每时刻固定K=4，一次漏检后下一时刻触发R-Net，连续5次失败才终止。',
                  '原记录组互斥；真实测试记录组在以前实验已使用，不是新盲测。',
                  '新数据20Hz、码本标签及网络/损失同时变化，不能将与旧结果的差值单归因于某一模块。'])
    (a.test_run/'journal_training_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print(a.test_run/'journal_training_report.txt')


if __name__=='__main__':main()
