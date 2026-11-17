"""Anonymous association drives the actual Huber and original N/R histories.

Controller inputs contain acquired measurements only. Source labels and current
truth positions enter the separate evaluator after the controller has finished.
"""
from collections import defaultdict
import time

import numpy as np
import torch

from .beam_identity import AnonymousDetections, associate
from .common import normalized_history
from .measurement_quality import RobustMeasurementProcessor
from .multi_target import _prediction, joint_beams


class RecordedRadar:
    """Simulation environment; public scan packets contain no source identity."""
    def __init__(self, measurement, sigma, beams_zero_based, seed):
        self.measurement, self.sigma = measurement, sigma
        self.beams, self.seed = beams_zero_based, seed
        self.receipts = {}

    def scan(self, step, live, selected):
        packets = []
        for local, group in enumerate(live):
            source = np.flatnonzero(np.isin(self.beams[group, :, step], selected[local]))
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(group), step]))
            source = source[rng.permutation(len(source))]
            packet = AnonymousDetections(self.beams[group, source, step],
                self.measurement[group, source, step], self.sigma[group, source, step])
            self.receipts[int(group), step] = packet, source.copy()
            packets.append(packet)
        return packets


def simulate_anonymous(nnet, rnet, checkpoint, initialization, initial_sigma,
                       sensor, steps, device, dt, budget, configuration, *,
                       association_function=associate, processor_factory=RobustMeasurementProcessor):
    """No current positions, beam labels, source IDs, or outlier flags in this API.

    Five consecutive unavailable updates still terminate the group. R-Net uses
    the original lost-history routing; its failed-beam tokens contain only beam
    channels with NO reported returns. Occupied/ambiguous beams remain -1.
    """
    groups, targets, history, _ = initialization.shape
    c = nnet.config
    if history != c.history or steps <= history or not 1 <= budget <= c.classes:
        raise ValueError('Invalid initialization, horizon, or beam budget')
    processor = processor_factory(initialization, initial_sigma, dt)
    measurements = np.zeros((groups, targets, steps, 3), dtype=np.float32)
    measurements[:, :, :history] = processor.filtered_initial_history
    valid = np.zeros((groups, targets, steps), dtype=bool); valid[:, :, :history] = True
    counts = np.zeros((groups, targets), dtype=np.int64)
    failed = np.full((groups, targets, c.failure_limit-1, budget), -1, dtype=np.int64)
    active = np.ones(groups, dtype=bool)
    events, timing = [], dict(total_ms=0., initialization_ms=0., detected_target_frames=0)
    calls = dict(nnet_forward_calls=0, nnet_target_frames=0, rnet_forward_calls=0,
                 rnet_target_frames=0, expected_rnet_target_frames=0)
    def count_n(module, inputs):
        calls['nnet_forward_calls'] += 1; calls['nnet_target_frames'] += len(inputs[0])
    def count_r(module, inputs):
        calls['rnet_forward_calls'] += 1; calls['rnet_target_frames'] += len(inputs[1])
    hooks = []
    if hasattr(nnet, 'register_forward_pre_hook'):
        hooks.append(nnet.register_forward_pre_hook(count_n))
    if rnet is not None and hasattr(rnet, 'register_forward_pre_hook'):
        hooks.append(rnet.register_forward_pre_hook(count_r))
    try:
        with torch.inference_mode():
            for step in range(history, steps):
                live = np.flatnonzero(active)
                if not len(live): break
                mask = valid[live, :, step-history:step]
                past = normalized_history(measurements[live, :, step-history:step], mask,
                                          checkpoint['normalization'])
                prior = counts[live].copy()
                calls['expected_rnet_target_frames'] += int((prior > 0).sum()) if rnet is not None else 0
                logits = _prediction(nnet, rnet, torch.from_numpy(past).to(device).flatten(0,1),
                    torch.from_numpy(mask).to(device).flatten(0,1),
                    torch.from_numpy(failed[live]).to(device).flatten(0,1),
                    torch.from_numpy(prior).to(device).flatten()).reshape(len(live), targets, -1)
                selected = joint_beams(logits, budget).cpu().numpy()
                probabilities = logits.double().softmax(-1).cpu().numpy()
                packets = sensor.scan(step, live, selected)
                predicted = processor.state @ processor.transition.T
                covariance = processor.transition @ processor.covariance @ processor.transition.T + processor.process_covariance
                observed = np.zeros((groups, targets), dtype=bool)
                assigned_z = np.full((groups, targets, 3), np.nan)
                assigned_sigma = np.full_like(assigned_z, np.nan)
                indices = np.full((len(live), targets), -1, dtype=int)
                feedback = np.full((len(live), budget), -1, dtype=int)
                for i, group in enumerate(live):
                    packet = packets[i]
                    decisions, _ = association_function(probabilities[i], packet,
                        predicted[group, :, :3], covariance[group, :, :3, :3], **configuration)
                    assigned = np.asarray(decisions['beam_geometry'])
                    if (assigned.shape != (len(packet.beams),) or (assigned < -1).any()
                            or (assigned >= targets).any()):
                        raise ValueError('Invalid anonymous assignment')
                    accepted = assigned[assigned >= 0]
                    if len(accepted) != len(np.unique(accepted)):
                        raise ValueError('Two detections cannot update the same track')
                    empty = ~np.isin(selected[i], packet.beams)
                    feedback[i, empty] = selected[i, empty]
                    assert not np.isin(feedback[i][feedback[i] >= 0], packet.beams).any()
                    for detection, target in enumerate(assigned):
                        if target >= 0:
                            observed[group,target] = True
                            assigned_z[group,target] = packet.measurements[detection]
                            assigned_sigma[group,target] = packet.sigmas[detection]
                            indices[i,target] = detection
                started = time.perf_counter()
                filtered, _ = processor.step(step, observed, assigned_z, assigned_sigma)
                timing['total_ms'] += (time.perf_counter()-started)*1000
                timing['detected_target_frames'] += int(observed.sum())
                measurements[:, :, step][observed] = filtered[observed]
                valid[:, :, step] = observed
                hit = observed[live]
                counts[live] = np.where(hit, 0, prior+1)
                active[live[(counts[live] >= c.failure_limit).any(-1)]] = False
                positions = np.full((len(live),targets,3),np.nan)
                good_i,good_t = np.nonzero(hit)
                if len(good_i):
                    g = live[good_i]
                    vm = valid[g,good_t,step-history+1:step+1]
                    hm = normalized_history(measurements[g,good_t,step-history+1:step+1],vm,checkpoint['normalization'])
                    positions[good_i,good_t] = nnet(torch.from_numpy(hm).to(device),
                                                  torch.from_numpy(vm).to(device))['position'].cpu().numpy()
                for i,group in enumerate(live):
                    for target in range(targets):
                        if hit[i,target]: failed[group,target] = -1
                        elif prior[i,target] < c.failure_limit-1:
                            failed[group,target,prior[i,target]] = feedback[i]
                        events.append(dict(scene_id=int(group),target_index=target,time_index=step,
                            time_s=(step-history)*dt,hit=bool(hit[i,target]),
                            assigned_detection_index=int(indices[i,target]),
                            prior_misses=int(prior[i,target]),post_misses=int(counts[group,target]),
                            recovered=bool(hit[i,target] and prior[i,target]>0),
                            mode='R' if prior[i,target] else 'N',
                            reception_status='updated' if hit[i,target] else
                                'unassigned_with_returns' if len(packets[i].beams) else 'confirmed_empty_scan',
                            failed_beams_1based=','.join(str(b+1) for b in feedback[i] if b>=0)
                                if not hit[i,target] else '',
                            occupied_beams_1based=','.join(str(b+1) for b in np.unique(packets[i].beams)),
                            **{f'estimated_{a}_m':float(positions[i,target,k]) if hit[i,target] else None
                               for k,a in enumerate('xyz')}))
    finally:
        for hook in hooks: hook.remove()
    if hooks and rnet is not None and calls['rnet_target_frames'] != calls['expected_rnet_target_frames']:
        raise RuntimeError('R-Net call count does not match missing-history routing')
    calls.update(original_nr_routing_verified=True,alternative_prediction_network_used=False,
        trajectory_estimator='nnet_position_head',occupied_beams_never_encoded_as_failures=True,
        measurement_history_is_actual_association_output=True)
    return events, timing, calls


def evaluate(events, sensor, scenes, history, dt, failure_limit):
    """Offline evaluation only; cannot affect decisions, histories, or stopping."""
    groups,targets,steps = scenes['beam'].shape
    indexed = defaultdict(lambda:defaultdict(list))
    for e in events:
        group,step,target = e['scene_id'],e['time_index'],e['target_index']
        j = e['assigned_detection_index']
        source = int(sensor.receipts[group,step][1][j]) if j>=0 else -1
        e['assigned_source_evaluation_only']=source
        e['identity_correct']=source==target
        e['trajectory_id']=int(scenes['trajectory_id'][group,target])
        e['estimate_source']='nnet_position_head'
        e['phase']='tracking'
        for k,a in enumerate('xyz'): e[f'true_{a}_m']=float(scenes['position'][group,target,step,k])
        e['position_squared_error']=sum((e[f'estimated_{a}_m']-e[f'true_{a}_m'])**2 for a in 'xyz') if e['hit'] else 0.
        indexed[group][step].append(e)
    controller_scenes,controller_targets,identity_scenes,identity_targets=[],[],[],[]
    for group in range(groups):
        all_rows=[e for step in sorted(indexed[group]) for e in indexed[group][step]]
        for identity_mode,scene_rows,target_rows in ((False,controller_scenes,controller_targets),
                                                   (True,identity_scenes,identity_targets)):
            gap=np.zeros(targets,dtype=int); missed=np.zeros_like(gap); lost=np.zeros_like(gap)
            recovered=np.zeros_like(gap); delays=np.zeros_like(gap); position_n=np.zeros_like(gap)
            sse=np.zeros(targets);blocks=0;wrong=0;terminal=False
            for step in sorted(indexed[group]):
                blocks+=1
                for e in indexed[group][step]:
                    target=e['target_index'];good=e['identity_correct'] if identity_mode else e['hit']
                    if good:
                        recovered[target]+=int(gap[target]>0);delays[target]+=gap[target];gap[target]=0
                    else:
                        lost[target]+=int(gap[target]==0);gap[target]+=1;missed[target]+=1
                    wrong+=int(e['hit'] and not e['identity_correct'])
                    if e['hit']:
                        sse[target]+=e['position_squared_error'];position_n[target]+=1
                if (gap>=failure_limit).any():terminal=True;break
            success=not terminal and blocks==steps-history
            scene_rows.append(dict(scene_id=group,success=bool(success),duration_s=blocks*dt,
                evaluated_blocks=blocks,recoveries=int(recovered.sum()),missed_target_blocks=int(missed.sum()),
                terminal_failed_targets=int((gap>=failure_limit).sum()),wrong_updates=wrong,
                zero_wrong_update_complete=bool(success and wrong==0)))
            for target in range(targets):
                failed=bool(gap[target]>=failure_limit)
                target_rows.append(dict(scene_id=group,target_index=target,
                    trajectory_id=int(scenes['trajectory_id'][group,target]),
                    terminal_failure=failed,censored_by_peer_failure=not success and not failed,
                    observed_duration_s=blocks*dt,missed_blocks=int(missed[target]),lost_events=int(lost[target]),
                    recoveries=int(recovered[target]),recovery_delay_blocks_sum=int(delays[target]),
                    position_count=int(position_n[target]),position_squared_error_sum=float(sse[target])))
    association=dict(total=0,assigned=0,correct=0,wrong=0,rejected=0,collision_total=0,
        collision_correct=0,identity_switches=0,identity_switches_consecutive=0,
        unassigned_with_returns=sum(e['reception_status']=='unassigned_with_returns' for e in events),
        confirmed_empty_scan=sum(e['reception_status']=='confirmed_empty_scan' for e in events),
        confusion=np.zeros((targets,targets+1),dtype=int))
    previous=np.broadcast_to(np.arange(targets),(groups,targets)).copy()
    last_step=np.full((groups,targets),history-1)
    for (group,step),(packet,sources) in sorted(sensor.receipts.items(),key=lambda row:(row[0][1],row[0][0])):
        assignments={e['assigned_detection_index']:e['target_index'] for e in indexed[group][step] if e['hit']}
        unique,count=np.unique(packet.beams,return_counts=True)
        for j,source in enumerate(sources):
            target=assignments.get(j,-1);correct=target==source
            collision=bool(packet.beams[j] in unique[count>1])
            association['total']+=1;association['assigned']+=int(target>=0)
            association['correct']+=int(correct);association['wrong']+=int(target>=0 and not correct)
            association['rejected']+=int(target<0);association['collision_total']+=int(collision)
            association['collision_correct']+=int(collision and correct)
            association['confusion'][source,target if target>=0 else targets]+=1
            if target>=0:
                changed=previous[group,source]!=target
                association['identity_switches']+=int(changed)
                association['identity_switches_consecutive']+=int(changed and last_step[group,source]==step-1)
                previous[group,source]=target;last_step[group,source]=step
    association['confusion']=association['confusion'].tolist()
    return dict(scene_rows=identity_scenes,target_rows=identity_targets,
        controller_scene_rows=controller_scenes,controller_target_rows=controller_targets,
        association=association),events


def aggregate_association(records, maximum, targets):
    result={key:sum(r['association'][key] for r in records) for key in records[0]['association'] if key!='confusion'}
    result['confusion']=np.sum([r['association']['confusion'] for r in records],axis=0).tolist()
    n,assigned=result['total'],result['assigned']
    result.update(accuracy_all_returns=result['correct']/n if n else None,
        precision_assigned=result['correct']/assigned if assigned else None,
        assignment_coverage=assigned/n if n else None,
        accuracy_same_beam=result['collision_correct']/result['collision_total'] if result['collision_total'] else None,
        correct_update_coverage=result['correct']/(len(records)*len(records[0]['scene_rows'])*targets*maximum),
        zero_wrong_update_scene_success_rate=float(np.mean([s['zero_wrong_update_complete'] for r in records for s in r['scene_rows']])))
    return result


def physical_reception_metrics(events,scenes,history):
    """Offline, original beam-miss recovery definition, excluding association rejection.

    In this simulator each illuminated target reports one return and there is no
    additional random sensor miss. Thus a physical target returns iff its true
    beam is among the occupied scan channels. This reconstruction is evaluation
    only and must never drive controller stopping, histories, or R-Net inputs.
    """
    counts=np.zeros(scenes['trajectory_id'].shape,dtype=int)
    result=dict(recoveries=0,lost_events=0,missed_target_frames=0,evaluated_target_frames=0)
    for e in events:
        g,t,k=e['scene_id'],e['target_index'],e['time_index']
        occupied={int(b) for b in e['occupied_beams_1based'].split(',') if b}
        hit=int(scenes['beam'][g,t,k]) in occupied
        result['recoveries']+=int(hit and counts[g,t]>0)
        result['lost_events']+=int(not hit and counts[g,t]==0)
        result['missed_target_frames']+=int(not hit);result['evaluated_target_frames']+=1
        counts[g,t]=0 if hit else counts[g,t]+1
    result['recovery_rate']=result['recoveries']/result['lost_events'] if result['lost_events'] else None
    return result


def aggregate_physical(records):
    result={key:sum(r['physical_reception'][key] for r in records)
            for key in ('recoveries','lost_events','missed_target_frames','evaluated_target_frames')}
    result['recovery_rate']=result['recoveries']/result['lost_events'] if result['lost_events'] else None
    return result
