"""正式版单目标因果闭环：一次漏检后切R-Net、连续5次失败终止、不补入未知测量。"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# VS Code可能仍选择系统Python；在导入第三方依赖前切换到项目环境。
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parent.parent
    project_python = project_root/'.venv/Scripts/python.exe'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if not project_python.is_file():
        raise SystemExit(f'找不到项目Python环境：{project_python}')
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('MPLCONFIGDIR', str(project_root/'.cache/matplotlib'))
    if Path(sys.executable).resolve() != project_python.resolve():
        print(f'自动使用项目环境：{project_python}', flush=True)
        raise SystemExit(subprocess.run([str(project_python), '-X', 'utf8',
                                        str(Path(__file__).resolve()), *sys.argv[1:]]).returncode)

import numpy as np
import torch

from core.common import BASE, load_nnet, load_split, new_run, normalized_history, read_json, write_json, write_rows
from core.plots import finish_tracking_plots, save_tracking_plots
from core.models import JournalRNet


def simulate(nnet, rnet, checkpoint, source, device, dt, k=4, record_events=False):
    data = load_split(source); measurement, beam = data['measurement'],data['beam']-1
    n,t = beam.shape; c=nnet.config; h=c.history
    available=np.zeros((n,t),bool); available[:,:h]=True
    active=np.ones(n,bool); count=np.zeros(n,dtype='int64'); failed=np.full((n,c.failure_limit-1,k),-1,dtype='int64')
    evaluated=np.zeros(n,dtype=int); misses=np.zeros(n,dtype=int); recoveries=np.zeros(n,dtype=int)
    recovery_delays=[[] for _ in range(n)]; events=[]
    for step in range(h,t):
        take=np.flatnonzero(active)
        if not len(take): break
        valid=available[take,step-h:step]
        history=normalized_history(measurement[take,step-h:step],valid,checkpoint['normalization'])
        x=torch.from_numpy(history).to(device); v=torch.from_numpy(valid).to(device)
        with torch.no_grad():
            logits=nnet(x,v)['logits'][:,0]
            lost=count[take]>0
            if rnet is not None and lost.any():
                mask=torch.from_numpy(lost).to(device)
                logits[mask]=rnet(nnet,x[mask],v[mask],torch.from_numpy(failed[take[lost]]).to(device),
                                  torch.from_numpy(count[take[lost]]).to(device))[:,0]
            chosen=logits.topk(k,-1).indices.cpu().numpy()
        # 当前真值只在动作产生之后用于环境判定，不传入模型。
        hit=(chosen==beam[take,step,None]).any(1); evaluated[take]+=1
        if record_events:
            for j,i in enumerate(take):
                events.append({'trajectory_id':int(data['trajectory_id'][i]),'time_index':step,
                               'mode':'R' if rnet is not None and count[i]>0 else 'N',
                               'prior_consecutive_misses':int(count[i]),'chosen_beams_1based':','.join(map(str,chosen[j]+1)),
                               'hit':bool(hit[j])})
        good=take[hit]; bad=take[~hit]
        for i in good[count[good]>0]: recovery_delays[i].append(int(count[i]))
        recoveries[good]+=(count[good]>0); available[good,step]=True; count[good]=0; failed[good]=-1
        if len(bad):
            misses[bad]+=1; room=count[bad]<c.failure_limit-1
            failed[bad[room],count[bad[room]]]=chosen[~hit][room]
            count[bad]+=1; active[bad[count[bad]>=c.failure_limit]]=False
    rows=[]
    for i in range(n):
        episodes=int(recoveries[i]+(count[i]>0))
        rows.append({'trajectory_id':int(data['trajectory_id'][i]),'success':bool(active[i]),
                     'duration_s':float(evaluated[i]*dt),'evaluated_blocks':int(evaluated[i]),'missed_blocks':int(misses[i]),
                     'lost_events':episodes,'recoveries':int(recoveries[i]),
                     'recovery_delay_blocks_sum':sum(recovery_delays[i])})
    return rows,events


def summarize(rows,dt):
    recoveries=sum(r['recoveries'] for r in rows); episodes=sum(r['lost_events'] for r in rows)
    total=sum(r['duration_s'] for r in rows)
    return {'count':len(rows),'successful_segments':sum(r['success'] for r in rows),
            'success_rate':float(np.mean([r['success'] for r in rows])),
            'mean_duration_s':float(np.mean([r['duration_s'] for r in rows])),
            'recoveries':recoveries,'lost_events':episodes,'missed_blocks':sum(r['missed_blocks'] for r in rows),
            'recovery_rate':recoveries/episodes if episodes else None,
            'mean_recovery_delay_s':sum(r['recovery_delay_blocks_sum'] for r in rows)*dt/recoveries if recoveries else None,
            'lost_events_per_minute':episodes/total*60,'total_evaluated_s':total}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,help='R-Net权重；默认读取现有实验记录')
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--test-data',type=Path);p.add_argument('--k',type=int,default=4)
    p.add_argument('--events',action='store_true',help='兼容旧命令；现在默认保存逐时刻记录以自动绘图')
    p.add_argument('--no-show',action='store_true',help='保存两张图，但不弹出窗口（用于后台运行）')
    p.add_argument('--plot-trajectory',type=int,help='指定状态图的轨迹ID；默认选择R-Net恢复次数最多的轨迹')
    p.add_argument('--output-root',type=Path,default=BASE/'outputs/step10g_journal');a=p.parse_args()
    if a.test_data and a.split!='test':p.error('--test-data只允许独立测试目录')
    if a.checkpoint is None:
        current=BASE/'outputs/current_journal_experiment.json'
        if not current.is_file():p.error('未找到现有实验记录，请用--checkpoint指定R-Net权重')
        a.checkpoint=Path(read_json(current)['rnet_checkpoint'])
    if not a.checkpoint.is_file():p.error(f'找不到R-Net权重：{a.checkpoint}')
    print(f'使用现有权重：{a.checkpoint.resolve()}',flush=True)
    print(f'评估数据集：{a.split}；测试结束后自动生成两张图。',flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved=torch.load(a.checkpoint,map_location=device,weights_only=False)
    nnet,checkpoint=load_nnet(saved['nnet_checkpoint'],device)
    rnet=JournalRNet(nnet.config).to(device);rnet.load_state_dict(saved['model']);rnet.eval()
    data=a.test_data or Path(checkpoint['data']);cfg=read_json(data/'config.json');dt=cfg['dt']
    if dt!=.05:p.error('正式版评估必须使用0.05秒数据')
    with np.load(data/f'{a.split}.npz') as d:
        steps=d['beam'].shape[1]
        if a.plot_trajectory is not None and a.plot_trajectory not in d['trajectory_id']:
            p.error(f'当前测试集不包含轨迹 {a.plot_trajectory}')
    out=new_run(a.output_root);results={};all_rows={};all_events={}
    for name,network in [('nnet_only',None),('nnet_rnet',rnet)]:
        print(f'正在评估 {name} ...',flush=True)
        rows,events=simulate(nnet,network,checkpoint,data/f'{a.split}.npz',device,dt,a.k,record_events=True)
        write_rows(out/f'{name}.csv',rows);write_rows(out/f'{name}_events.csv',events)
        all_rows[name]=rows;all_events[name]=events
        results[name]=summarize(rows,dt);print(name,results[name],flush=True)
    plots,figures=save_tracking_plots(out,all_rows,all_events,dt,nnet.config.history,
                                    (steps-nnet.config.history)*dt,
                                    show=not a.no_show,trajectory_id=a.plot_trajectory)
    write_json(out/'summary.json',{'split':a.split,'data':str(data.resolve()),'dataset_kind':cfg['dataset_kind'],
                                  'rnet_checkpoint':str(a.checkpoint.resolve()),'nnet_checkpoint':saved['nnet_checkpoint'],
                                  'k':a.k,'failure_limit':nnet.config.failure_limit,'dt':dt,
                                  'maximum_evaluated_duration_s':(steps-nnet.config.history)*dt,'results':results,
                                  'plots':plots,
                                  'test_is_new_blind_set':not cfg.get('test_records_used_previously',False),
                                  'model_inputs_are_causal':True,'known_data_limitations':cfg['limitations']})
    (a.output_root/f'latest_{a.split}_run.txt').write_text(out.name,encoding='utf-8');print(out,flush=True)
    print(f'状态时间图：{(out/plots["tracking_status"]).resolve()}',flush=True)
    print(f'持续跟踪比例曲线：{(out/plots["tracking_survival"]).resolve()}',flush=True)
    if not a.no_show:print('测试已完成，关闭两个图窗后程序退出。',flush=True)
    finish_tracking_plots(figures,show=not a.no_show)


if __name__=='__main__':main()
