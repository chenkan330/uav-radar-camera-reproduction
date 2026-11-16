"""极小数据的整链路检查；输出单独存放，绝不计作正式实验成绩。"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.common import BASE, new_run, read_json, write_json


def main():
    root=new_run(BASE/'outputs/journal_workflow_smoke')
    def run(script,*args):
        subprocess.run([sys.executable,str(BASE/script),*map(str,args)],cwd=BASE.parent,check=True)
    def latest(stage):
        folder=root/stage
        return folder/(folder/'latest_run.txt').read_text(encoding='utf-8').strip()
    run('core/workflows/step05g_prepare_journal_dataset.py','--count',20,'--validation-count',5,'--test-count',5,'--output-root',root/'data')
    data=latest('data')
    run('core/workflows/step07d_train_journal_nnet.py','--data',data,'--epochs',1,'--stride',5,'--output-root',root/'nnet')
    nnet=latest('nnet')/'best_nnet.pth'
    run('core/workflows/step08d_collect_journal_lost.py','--checkpoint',nnet,'--stride',5,'--output-root',root/'lost')
    lost=latest('lost')
    run('core/workflows/step09d_train_journal_rnet.py','--data',lost,'--epochs',1,'--output-root',root/'rnet')
    rnet=latest('rnet')/'best_rnet.pth'
    run('step10d_journal_tracking.py','--checkpoint',rnet,'--split','validation','--no-show','--output-root',root/'tracking')
    run('step10d_journal_tracking.py','--checkpoint',rnet,'--split','test','--no-show','--output-root',root/'tracking')
    assert read_json(nnet.parent/'summary.json')['epochs_completed']==1
    assert read_json(rnet.parent/'summary.json')['nnet_unchanged']
    result={'passed':True,'purpose':'small_end_to_end_integration_test_NOT_experiment_results',
            'data':str(data),'nnet_checkpoint':str(nnet),'rnet_checkpoint':str(rnet)}
    write_json(root/'smoke_result.json',result);print('End-to-end workflow smoke test PASSED:',root,flush=True)


if __name__=='__main__':main()
