"""审计论文版本、源码一致性、参数量、发布数据场景差异，保存可复查事实。"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from scipy.io import loadmat

from core.common import BASE, write_json
from core.workflows.step05g_prepare_journal_dataset import BS, centered_dft_beams
from core.models import JournalConfig, JournalNNet, JournalRNet


def git_blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def main():
    source=BASE.parent/'Transformer-based-mmWave-tracking/code4tracking'
    expected={'bert_model.py':'1df92f9395fc95d66ff8f2989ffd04ecee6838d5',
              'transformers.py':'3c81f66ad5f009eb4996826e5eafb9ab814615b2',
              'refine_Model.py':'9508842e23c6fac5ef5019ffc53d572a119ab0ef',
              'parameters.py':'4e3e7d6875c2892da6f9348819cfd9d792355835',
              'test_collect_lost_data.py':'f66f57dba337b9e76ee5e592081337b9f0ffefc2',
              'utils/tools.py':'b0d77c2bf27b68c2021dc6e1646715a933dc8f44'}
    source_check={}
    for name,blob in expected.items():
        raw=(source/name).read_bytes()
        source_check[name]={'expected_git_blob':blob,'matches_upstream':blob in (git_blob(raw),git_blob(raw.replace(b'\r\n',b'\n'))),
                            'local_sha256':hashlib.sha256(raw).hexdigest()}
    mat=loadmat(source/'data/testing_data_vel35_mt32_dft.mat')
    labels=mat['all_beam_numbers'][:,0]
    origin_zero_agreement=float(np.mean(centered_dft_beams(mat['all_trajectories']+BS)==labels))
    origin_ten_agreement=float(np.mean(centered_dft_beams(mat['all_trajectories'])==labels))
    c=JournalConfig()
    paper=next(BASE.parent.glob('m3TrackFormer_Transformer*.pdf'))
    record={'target':'local_TWC_formal_journal_version','doi':'10.1109/TWC.2026.3720890',
            'paper_pdf_sha256':hashlib.sha256(paper.read_bytes()).hexdigest(),
            'upstream_commit':'f9dbbc90da4b5b2355bf5c587e013a1841b26aeb','upstream_files':source_check,
            'formal_setup':{'dt':.05,'duration':50,'training_trajectories':10000,'fresh_test_trajectories':1000,
                            'encoder_layers':4,'decoder_layers':2,'d_model':256,'heads':8,'batch_size':256,
                            'optimizer':'AdamW','lr':.0003,'epochs':50,'failure_limit':5,'history':5,
                            'phase1_loss':'CE + eta*sqrt(expected_angular_distance_squared) + lambda*position_MSE',
                            'phase2_loss':'same_angular_classification_loss_only'},
            'arxiv_v1_differences':{'dt':.1,'encoder_layers':3,'raw_trajectories':20000,'angular_penalty':False},
            'parameter_count':{'implemented_nnet':sum(p.numel() for p in JournalNNet(c).parameters()),
                               'implemented_additional_le_ff':sum(p.numel() for p in JournalRNet(c).parameters()),
                               'implemented_rnet_with_entire_frozen_nnet':sum(p.numel() for p in JournalNNet(c).parameters())+sum(p.numel() for p in JournalRNet(c).parameters()),
                               'reported_paper_nnet_approx':3300000,'reported_paper_rnet_approx':4200000,
                               'paper_rnet_shared_parameter_counting_convention':'not_fully_specified',
                               'counts_match_reported_paper':False},
            'released_test_data':{'trajectories':len(labels)//1005,'samples_per_trajectory':1005,
                                  'beam_label_match_with_base_z0':origin_zero_agreement,
                                  'beam_label_match_with_paper_base_z10':origin_ten_agreement,
                                  'used_to_train_new_networks':False},
            'known_unresolved':['作者角距损失64x16角度索引与其32x32方向余弦DFT标签排序不同，未给转换表。',
                               '论文参数量与公开权重/按公式构造的网络不一致。',
                               '原始训练数据、完整训练入口、OFDM/Swerling I测量生成及估计器未公开。',
                               '多token/采样细节、lambda/eta、初始位置等具体设定未完整给出。'],
            'strict_full_numerical_reproduction_claimed':False,
            'source_code_left_unmodified':all(item['matches_upstream'] for item in source_check.values())}
    out=BASE/'outputs/journal_alignment_audit.json';out.parent.mkdir(exist_ok=True)
    write_json(out,record)
    print('Source files match pinned upstream:',record['source_code_left_unmodified'])
    print('Formal vs implemented parameter counts:',record['parameter_count'])
    print('Released test scene audit:',record['released_test_data'])
    print(out)


if __name__=='__main__':main()
