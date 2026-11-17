"""Closed-loop association must affect history without reading evaluation truth."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from core.identity_closed_loop import RecordedRadar,simulate_anonymous,evaluate,physical_reception_metrics
from core.identity_closed_loop_experiment import acceptance,selection_key
from core.measurement_quality import RobustMeasurementProcessor,spherical_to_cartesian
from core.models import JournalConfig
from test_trajectory_figure import NORMALIZATION


class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__();self.config=JournalConfig(classes=8)
    def forward(self,x,valid):
        logits=torch.arange(8.,0.,-1.)[None,None].expand(len(x),1,-1).clone()
        return {'logits':logits,'position':x[:,-1,:]}


class CheckEvidenceRNet(torch.nn.Module):
    def forward(self,nnet,x,valid,failed,count):
        if (failed==0).any():raise AssertionError('Occupied beam entered negative evidence')
        return nnet(x,valid)['logits']


def data_fixture(targets=2,steps=12):
    z=np.zeros((1,targets,steps,3),dtype=np.float32)
    for target in range(targets):z[0,target]=[100.+100*target,0.,np.pi/2]
    return dict(measurement=z,measurement_sigma=np.broadcast_to([.5,.001,.001],z.shape).copy(),
        beam=np.ones((1,targets,steps),dtype=int),position=spherical_to_cartesian(z),
        trajectory_id=np.arange(targets)[None]+100)


def run(data,matcher,rnet=None,processor=RobustMeasurementProcessor):
    sensor=RecordedRadar(data['measurement'],data['measurement_sigma'],data['beam']-1,20261116)
    events,timing,calls=simulate_anonymous(ToyNet().eval(),rnet,NORMALIZATION,
        data['measurement'][...,:5,:],data['measurement_sigma'][...,:5,:],sensor,data['beam'].shape[-1],
        torch.device('cpu'),.05,4,{},association_function=matcher,processor_factory=processor)
    return events,sensor,calls


def swap_match(probability,packet,predicted,covariance,**kwargs):
    assigned=np.where(packet.measurements[:,0]<150.,1,0)
    return {'beam_geometry':assigned},None


def no_match(probability,packet,predicted,covariance,**kwargs):
    return {'beam_geometry':np.full(len(packet.beams),-1,dtype=int)},None


class ClosedLoopTests(unittest.TestCase):
    def test_wrong_association_really_updates_the_other_track_huber_history(self):
        submitted=[]
        class Spy(RobustMeasurementProcessor):
            def step(self,index,observed,z,sigma):
                if index>=5:submitted.append((index,np.array(z),np.array(observed)))
                return super().step(index,observed,z,sigma)
        events,sensor,_=run(data_fixture(),swap_match,processor=Spy)
        self.assertEqual(submitted[0][1][0,0,0],200.)
        self.assertEqual(submitted[0][1][0,1,0],100.)
        self.assertTrue(submitted[0][2].all())
        self.assertTrue(all(e['hit'] for e in events))

    def test_evaluator_cannot_keep_a_wrong_identity_alive_or_stop_online_controller(self):
        data=data_fixture();events,sensor,_=run(data,swap_match)
        self.assertEqual(max(e['time_index'] for e in events),11)
        scores,_=evaluate(events,sensor,data,5,.05,5)
        self.assertTrue(scores['controller_scene_rows'][0]['success'])
        self.assertFalse(scores['scene_rows'][0]['success'])
        self.assertEqual(scores['scene_rows'][0]['duration_s'],.25)
        self.assertEqual(scores['association']['correct'],0)
        self.assertEqual(scores['association']['wrong'],14)

    def test_ambiguity_never_becomes_occupied_beam_negative_evidence(self):
        events,_,calls=run(data_fixture(),no_match,CheckEvidenceRNet().eval())
        self.assertEqual(max(e['post_misses'] for e in events),5)
        self.assertTrue(all(e['reception_status']=='unassigned_with_returns' for e in events))
        self.assertTrue(all(e['failed_beams_1based']=='2,3,4' for e in events))
        self.assertEqual(calls['rnet_target_frames'],8)
        self.assertEqual(calls['rnet_target_frames'],calls['expected_rnet_target_frames'])

    def test_unscanned_measurements_and_sigmas_cannot_affect_controller(self):
        data=data_fixture();data['beam'][...,5:]=8
        a,_,_=run(data,no_match)
        other={k:v.copy() for k,v in data.items()}
        other['measurement'][...,5:,:]=np.nan;other['measurement_sigma'][...,5:,:]=np.nan
        b,_,_=run(other,no_match)
        self.assertEqual(a,b)
        self.assertTrue(all(e['reception_status']=='confirmed_empty_scan' for e in a))

    def test_stage_acceptance_uses_absolute_success_loss_and_actual_precision(self):
        reference=dict(success_rate=.7,mean_all_targets_duration_s=40.)
        self.assertTrue(acceptance(dict(success_rate=.66,mean_all_targets_duration_s=39.),
            dict(precision_assigned=.999),reference)['passed'])
        self.assertFalse(acceptance(dict(success_rate=.64,mean_all_targets_duration_s=39.),
            dict(precision_assigned=1.),reference)['passed'])
        self.assertFalse(acceptance(dict(success_rate=.7,mean_all_targets_duration_s=37.),
            dict(precision_assigned=1.),reference)['passed'])
        self.assertFalse(acceptance(dict(success_rate=.7,mean_all_targets_duration_s=40.),
            dict(precision_assigned=.994),reference)['passed'])

    def test_validation_cannot_prefer_more_successes_bought_by_wrong_identities(self):
        a=dict(association=dict(precision_assigned=.999,correct_update_coverage=.7),
            identity_tracking=dict(success_rate=.7,mean_all_targets_duration_s=40.))
        b=dict(association=dict(precision_assigned=.98,correct_update_coverage=.9),
            identity_tracking=dict(success_rate=.95,mean_all_targets_duration_s=49.))
        self.assertGreater(selection_key(a),selection_key(b))

    def test_association_rejections_are_not_reported_as_physical_beam_misses(self):
        data=data_fixture();events,sensor,_=run(data,no_match)
        scores,events=evaluate(events,sensor,data,5,.05,5)
        self.assertGreater(sum(r['lost_events'] for r in scores['target_rows']),0)
        physical=physical_reception_metrics(events,data,5)
        self.assertEqual(physical['lost_events'],0)
        self.assertIsNone(physical['recovery_rate'])


if __name__=='__main__':unittest.main()
