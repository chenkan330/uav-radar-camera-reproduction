"""正式版改动的语义检查：缺失隔离、动态前缀、多token因果性、冻结、损失及闭环。"""
import io
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from core.common import BASE, state_digest
from core.workflows.step05g_prepare_journal_dataset import BS, centered_dft_beams
from core.models import JournalConfig, JournalNNet, JournalRNet, angular_distance_matrix, classification_loss
from step10d_journal_tracking import simulate
from core.workflows.step08d_collect_journal_lost import collect_windows


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.c=JournalConfig(d_model=32,heads=4,encoder_layers=2,decoder_layers=2,dropout=0)
        self.nnet=JournalNNet(self.c).eval().requires_grad_(False)
        self.rnet=JournalRNet(self.c).eval()
        self.h=torch.randn(2,5,3);self.valid=torch.tensor([[1,1,1,1,0],[1,0,0,0,0]],dtype=torch.bool)
        self.failed=torch.full((2,4,5),-1,dtype=torch.long)
        self.failed[0,0,:3]=torch.tensor([1,20,100])
        self.failed[1,:,:3]=torch.tensor([3,90,800])
        self.count=torch.tensor([1,4])

    def test_missing_measurements_cannot_change_predictions(self):
        altered=self.h.clone();altered[~self.valid]=1e8
        a=self.nnet(self.h,self.valid)['logits']
        b=self.nnet(altered,self.valid)['logits']
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

    def test_extra_future_tokens_cannot_change_first_prediction(self):
        a=self.nnet(self.h,self.valid,1)['logits'][:,0]
        b=self.nnet(self.h,self.valid,5)['logits'][:,0]
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        a=self.rnet(self.nnet,self.h,self.valid,self.failed,self.count,1)[:,0]
        b=self.rnet(self.nnet,self.h,self.valid,self.failed,self.count,5)[:,0]
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)

    def test_dynamic_prefix_batch_padding_is_invisible(self):
        a=self.rnet(self.nnet,self.h,self.valid,self.failed,self.count)
        one=self.rnet(self.nnet,self.h[:1],self.valid[:1],self.failed[:1],self.count[:1])
        torch.testing.assert_close(a[:1],one,atol=3e-6,rtol=3e-5)
        altered=self.failed.clone();altered[0,1:]=1023
        b=self.rnet(self.nnet,self.h,self.valid,altered,self.count)
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-5)

    def test_zero_feature_residual_recovers_nnet(self):
        class ZeroFusion(nn.Module):
            def forward(self,lost,motion,padding):return torch.zeros_like(lost)
        self.rnet.ff=nn.ModuleList([ZeroFusion()])
        a=self.rnet(self.nnet,self.h,self.valid,self.failed,self.count)
        b=self.nnet(self.h,self.valid)['logits']
        torch.testing.assert_close(a,b)

    def test_phase2_does_not_update_shared_nnet(self):
        before=state_digest(self.nnet);optimizer=torch.optim.AdamW(self.rnet.parameters(),lr=.001)
        loss=self.rnet(self.nnet,self.h,self.valid,self.failed,self.count).square().mean()
        optimizer.zero_grad();loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.rnet.parameters()))
        optimizer.step()
        self.assertEqual(before,state_digest(self.nnet))
        self.assertTrue(all(p.grad is None for p in self.nnet.parameters()))

    def test_classification_reduction_and_padding(self):
        logits=torch.tensor([[[1.,2.,3.],[3.,2.,1.]],[[2.,1.,3.],[1.,4.,2.]]],requires_grad=True)
        labels=torch.tensor([[2,0],[1,-1]])
        distance=torch.tensor([[0.,1.,2.],[1.,0.,3.],[2.,3.,0.]])
        loss=classification_loss(logits,labels,distance,.2)
        expected=0
        for i,j in [(0,0),(0,1),(1,0)]:
            expected+=-logits[i,j].log_softmax(-1)[labels[i,j]]+.2*torch.sqrt((logits[i,j].softmax(-1)*distance[labels[i,j]]).sum())
        torch.testing.assert_close(loss,expected/2)
        loss.backward();torch.testing.assert_close(logits.grad[1,1],torch.zeros(3))

    def test_angular_matrix_matches_upstream_formula(self):
        matrix=angular_distance_matrix()
        torch.testing.assert_close(matrix,matrix.T)
        torch.testing.assert_close(matrix.diag(),torch.zeros(1024))
        # 作者64个方位角、16个俯仰角索引；方位角首尾应环绕相邻。
        self.assertAlmostEqual(float(matrix[0,63]),(2*math.pi/64)**2,places=5)
        self.assertAlmostEqual(float(matrix[0,64]),(math.pi/32)**2,places=5)

    def test_dft_labels_match_released_data_with_its_origin(self):
        from scipy.io import loadmat
        path=BASE.parent/'Transformer-based-mmWave-tracking/code4tracking/data/testing_data_vel35_mt32_dft.mat'
        data=loadmat(path)
        # 原发布数据以z=0计算标签，论文场景z=10；这里只验证码本索引，绝不拿测试数据训练。
        label=centered_dft_beams(data['all_trajectories']+BS)
        np.testing.assert_array_equal(label,data['all_beam_numbers'][:,0])

    def test_closed_loop_misses_never_reveal_current_measurements(self):
        class ConstantNNet:
            config=JournalConfig(history=5)
            def __init__(self):self.histories=[]
            def __call__(self,history,valid):
                self.histories.append((history.cpu().numpy().copy(),valid.cpu().numpy().copy()))
                return {'logits':torch.tensor([6.,5.,4.,3.,0.,-1.]).view(1,1,6).expand(len(history),-1,-1)}
        model=ConstantNNet();memory=io.BytesIO()
        measured=np.arange(39,dtype='float32').reshape(1,13,3)
        # 内部0..3命中：先漏一次、恢复，再连续漏5次；最后时刻应完全不读取。
        np.savez(memory,measurement=measured,beam=np.array([[1]*5+[6,1]+[6]*5+[1]]),trajectory_id=np.array([9]))
        memory.seek(0)
        ck={'normalization':{'mean':[0,0,0],'std':[1,1,1]}}
        rows,_=simulate(model,None,ck,memory,torch.device('cpu'),.05,k=4)
        self.assertEqual(rows[0]['evaluated_blocks'],7)
        self.assertEqual(rows[0]['missed_blocks'],6)
        self.assertEqual(rows[0]['recoveries'],1)
        self.assertEqual(rows[0]['lost_events'],2)
        self.assertFalse(rows[0]['success'])
        # 第一次漏检之后，历史最后一条必须是零填充，而不是第5时刻的潜在测量。
        np.testing.assert_array_equal(model.histories[1][0][0,-1],np.zeros(3))
        self.assertFalse(model.histories[1][1][0,-1])

    def test_window_sampler_stores_only_past_failures_and_future_labels(self):
        class ConstantNNet:
            config=JournalConfig(history=5,future=5)
            def __call__(self,history,valid):
                z=torch.zeros(len(history),1,1024)
                z[:,:,:5]=torch.arange(10.,5.,-1)
                return {'logits':z}
        memory=io.BytesIO()
        m=np.arange(60,dtype='float32').reshape(1,20,3)
        np.savez(memory,measurement=m,beam=np.full((1,20),1024),trajectory_id=np.array([23]))
        memory.seek(0)
        ck={'normalization':{'mean':[0,0,0],'std':[1,1,1]}}
        samples,stats=collect_windows(ConstantNNet(),ck,memory,torch.device('cpu'),stride=1,batch_size=8)
        self.assertGreater(stats['unique_lost_states'],0)
        self.assertEqual(samples['label'].shape[1],5)
        self.assertTrue(((samples['label']==1023)|(samples['label']==-1)).all())
        self.assertTrue((samples['label'][:,0]==1023).all())
        for history,valid,count,failed in zip(samples['history'],samples['valid'],samples['count'],samples['failed']):
            self.assertTrue((history[-count:]==0).all())
            self.assertTrue((~valid[-count:]).all())
            self.assertTrue((failed[count:]==-1).all())


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
