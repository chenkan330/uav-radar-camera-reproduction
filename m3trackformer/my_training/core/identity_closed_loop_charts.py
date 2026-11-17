"""Figures from actual closed-loop association and neural position records."""
import numpy as np
from .metric_charts import _plotting


def trial_figures(out,record,primary_events,steps,history,dt):
    plt=_plotting(); folder=out/'figures';folder.mkdir(exist_ok=True)
    m,a=record['identity_tracking'],record['association']
    specs=[(m['success_rate']*100,'身份有效整段成功率','%'),
           (m['mean_all_targets_duration_s'],'平均身份有效时长','秒'),
           (100*m['recovery_rate'] if m['recovery_rate'] is not None else None,'身份有效重捕率','%'),
           (100*a['precision_assigned'] if a['precision_assigned'] is not None else None,'已接受关联正确率','%')]
    fig,axes=plt.subplots(1,4,figsize=(13.5,4.7))
    for ax,(value,title,unit) in zip(axes,specs):
        ax.bar([0],[value or 0],color='#287c69',width=.45)
        ax.text(0,value or 0,'N/A' if value is None else f'{value:.2f}',ha='center',va='bottom')
        ax.set(title=title,ylabel=unit,xticks=[],ylim=(0,115 if unit=='%' else max(1,value or 0)*1.3))
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('真实关联结果已接回 Huber 与原 Transformer N-Net/R-Net',fontsize=15)
    fig.text(.04,.02,'任一原始身份连续5帧没有正确关联，事后身份有效计时结束；控制器不读取此真值判据。身份错误不能充当跟踪成功。',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.9));fig.savefig(folder/'closed_loop_metrics.png',dpi=175);plt.close(fig)
    events=[e for e in primary_events if e['scene_id']==0]
    colors=['#2470b5','#e58824','#25925a']
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Patch
    grid=np.full((3,steps-history),-1)
    for e in events:grid[e['target_index'],e['time_index']-history]=e['assigned_source_evaluation_only']
    fig,ax=plt.subplots(figsize=(12,4.7))
    ax.imshow(grid,cmap=ListedColormap(['#e4e4e4',*colors]),
        norm=BoundaryNorm(np.arange(-1.5,3.5),4),aspect='auto',interpolation='nearest',
        extent=(0,(steps-history)*dt,2.5,-.5))
    wrong=[e for e in events if e['hit'] and not e['identity_correct']]
    if wrong:ax.scatter([e['time_s']+dt/2 for e in wrong],[e['target_index'] for e in wrong],c='#ca1641',marker='x',s=12)
    stop=record['noise_repeats'][0]['scene_rows'][0]['duration_s']
    ax.axvline(stop,color='#8c1c4a',ls='--',lw=1.2,label='身份有效计时结束')
    ax.set(title='固定首场景：各轨迹实际收到哪一个目标的测量',xlabel='闭环开始后的时间 / 秒',
        yticks=range(3),yticklabels=['轨迹A','轨迹B','轨迹C'])
    handles=[Patch(color=colors[i],label=f'实际来自{letter}') for i,letter in enumerate('ABC')]
    handles+=[Patch(color='#e4e4e4',label='无可用更新/已停止')]
    fig.legend(handles=handles,loc='lower center',ncol=4,bbox_to_anchor=(.5,.045),fontsize=9)
    fig.suptitle('闭环身份保持时间图：颜色来自事后来源核对，未参与关联',fontsize=14)
    fig.text(.04,.015,'每条轨迹保持其初始化身份；红×是错配更新。仅展示首组噪声，汇总指标使用全部固定种子。',fontsize=9)
    fig.tight_layout(rect=(0,.14,1,.90));fig.savefig(folder/'closed_loop_identity_timeline.png',dpi=175);plt.close(fig)
    counts=np.asarray(a['confusion']);fraction=counts/np.maximum(counts.sum(-1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(7,5))
    ax.imshow(fraction,vmin=0,vmax=1,cmap='Blues')
    for i in range(3):
        for j in range(4):
            ax.text(j,i,f'{counts[i,j]}\n{fraction[i,j]*100:.2f}%',ha='center',va='center',
                color='white' if fraction[i,j]>.6 else '#222222',fontsize=10)
    ax.set(xticks=range(4),xticklabels=['轨迹A','轨迹B','轨迹C','未关联'],
        yticks=range(3),yticklabels=['真实A','真实B','真实C'],xlabel='实际更新到的轨迹',title='闭环全部实际检出点：含拒配')
    fig.tight_layout();fig.savefig(folder/'closed_loop_confusion.png',dpi=175);plt.close(fig)


def comparison(root,records,reference,selected):
    plt=_plotting()
    names=['理想关联\nHuber参照']+[r['configuration']['label'] for r in records]
    colours=['#949ca8']+['#20835a' if r['configuration']['id']==selected else '#6f94b9' for r in records]
    fig,axes=plt.subplots(2,3,figsize=(13.5,9))
    specs=[('success_rate','身份有效整段成功率','%',100),
        ('mean_all_targets_duration_s','平均身份有效跟踪时长','秒',1),
        ('recovery_rate','身份有效重捕率','%',100),
        ('precision_assigned','已接受关联正确率','%',100),
        ('correct_update_coverage','正确更新覆盖完整时域的比例','%',100),
        ('position_rmse_m_on_detections','身份有效时段内N-Net位置RMSE','m',1)]
    for ax,(key,title,unit,scale) in zip(axes.flat,specs):
        ref=1. if key=='precision_assigned' else reference['position_evaluation_coverage'] if key=='correct_update_coverage' else reference[key]
        values=[ref]+[(r['association'] if key in ('precision_assigned','correct_update_coverage') else r['identity_tracking'])[key] for r in records]
        ax.bar(range(len(names)),[(v or 0)*scale for v in values],color=colours)
        for i,v in enumerate(values):ax.text(i,(v or 0)*scale,'N/A' if v is None else f'{v*scale:.2f}',ha='center',va='bottom',fontsize=10)
        ax.set_xticks(range(len(names)),names,fontsize=9)
        ax.set(title=title,ylabel=unit,ylim=(0,115 if unit=='%' else max([1]+[(v or 0)*scale for v in values])*1.28))
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('V6：关联结果真正进入闭环；原 N-Net/R-Net、Huber及12束预算不变',fontsize=16)
    fig.text(.04,.025,'绿色是验证集锁定的配置。相同17组场景×3组噪声；身份计时为事后评价，不控制在线算法。\n'
        '基准使用理想身份关联；未知身份版本仅向R-Net提供确认无回波的波束证据。RMSE帧集合不同，需结合覆盖率解读。',fontsize=9)
    fig.tight_layout(rect=(0,.10,1,.94),h_pad=2);fig.savefig(root/'closed_loop_comparison.png',dpi=175);plt.close(fig)
