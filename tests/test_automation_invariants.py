from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_rule_grid_is_prespecified():
    text=(ROOT/'configs/locked_rule_grid.yaml').read_text()
    assert 'absolute_threshold_px' in text and 'confidence' in text and 'iou' in text

def test_scale_records_have_both_domains():
    d=pd.read_parquet(ROOT/'outputs/scale/box_scale_records.parquet')
    assert {'aitod','uavdt','visdrone'} <= set(d.source)
    assert d[['max_side_px','normalized_side','normalized_area','normalized_diagonal']].notna().all().all()

def test_structural_pairs_are_not_quality_scores():
    d=pd.read_parquet(ROOT/'outputs/structural/pairwise_compatibility.parquet')
    n = d.source_a.nunique()
    assert len(d)==n*n and n >= 3 and 'comparison_boundary' in d

def test_claim_registry_has_reporting_scope():
    text = (ROOT/'configs/CLAIM_REGISTRY.yaml').read_text()
    assert text.count('reporting_scope:') == 8

def test_reference_results_are_scoped():
    d=pd.read_parquet(ROOT/'outputs/rule_grid_reference/metrics_long.parquet')
    assert set(d.source)=={'aitod','uavdt','visdrone'}
    assert set(d.confidence)=={0.25} and set(d.iou)=={0.25}

def test_reproduction_entrypoint_exists():
    assert (ROOT/'sourceaudit/reproduce.py').exists()

def test_declared_contract_path_is_in_manuscript():
    tex=(ROOT/'manuscript/final_grsl.tex').read_text()
    assert 'We declare four contracts as ordered pairs' in tex
    assert '$F$ filters the target' in tex
    assert '$R$ keeps the same filtered target' in tex
    summary=json.loads((ROOT/'outputs/coverage_corrected_grid/summary.json').read_text())
    assert summary['raw_multi_candidate_f1_headline_settings_iou025_iou050']==24
    assert summary['effective_multi_candidate_f1_policy_outcomes_iou025_iou050']==20

def test_contract_ledger_closes_path_identity():
    endpoints=pd.read_csv(ROOT/'outputs/contract_path/contract_endpoints.csv')
    ledger=pd.read_csv(ROOT/'outputs/contract_path/contract_transition_ledger.csv')
    for keys, rows in ledger.groupby(['source','mode','threshold']):
        end=endpoints[
            (endpoints.source==keys[0]) &
            (endpoints['mode']==keys[1]) &
            (endpoints.threshold==keys[2])
        ].set_index('contract')
        expected=float(end.loc['S','f1']-end.loc['A','f1'])
        assert abs(float(rows.delta_f1.sum())-expected) < 1e-12

def test_metric_rank_bootstrap_and_ap_crosscheck_are_prespecified():
    boot=pd.read_csv(ROOT/'outputs/metric_qualified_rank/paired_bootstrap.csv')
    assert set(boot.resampling_unit)=={'image','sequence'}
    assert set(boot.replicates)=={10000}
    seq=boot[boot.resampling_unit=='sequence']
    assert set(seq.sampling_units)=={76}
    cross=pd.read_csv(ROOT/'outputs/metric_qualified_rank/evaluator_crosscheck.csv')
    assert cross['pass'].all()
    assert cross.absolute_difference.max() < 1e-12

def test_metric_factorial_control_is_complete_and_closes_existing_endpoints():
    root=ROOT/'outputs/metric_factorial_control'
    points=pd.read_csv(root/'factorial_points.csv')
    pair=pd.read_csv(root/'factorial_pairwise_bootstrap.csv')
    cross=pd.read_csv(root/'factorial_ap_crosscheck.csv')
    assert set(points.iou)=={.25,.50}
    assert set(points.short_candidate)=={'Y11m','ASD'}
    assert len(points)==8
    assert set(pair.metric)=={'F1@conf=.25','max-F1','AP'}
    assert len(pair)==12 and set(pair.replicates)=={10000}
    assert set(pair.sampling_units)=={76}
    assert cross['pass'].all() and set(cross.iou)=={.50}
    existing=pd.read_csv(ROOT/'outputs/metric_qualified_rank/metric_rank_points.csv')
    for row in points.itertuples():
        old=existing[
            existing.candidate.eq(row.candidate) & existing['mode'].eq(row.mode) &
            np.isclose(existing.threshold,row.threshold)
        ].iloc[0]
        if np.isclose(row.iou,.25):
            assert np.isclose(row.f1_at_025,old.f1_at_025)
            assert np.isclose(row.max_f1,old.max_f1)
        else:
            assert np.isclose(row.ap,old.ap50)

def test_cross_configuration_contract_path_closes_and_repeats():
    root=ROOT/'outputs/cross_configuration_contract_paths'
    endpoints=pd.read_csv(root/'contract_endpoints.csv')
    ledger=pd.read_csv(root/'transition_ledger.csv')
    provenance=pd.read_csv(root/'artifact_provenance.csv')
    assert len(endpoints)==96
    assert len(ledger)==72
    assert provenance.groupby('source').candidate.nunique().to_dict()=={
        'AI-TOD':3, 'UAVDT':4, 'VisDrone':5,
    }
    removed=ledger[ledger.transition=='F->R']
    assert len(removed)==24
    assert (removed.delta_f1>0).all()
    assert len(removed[~removed.coverage_conditioned])==18

def test_scene_strata_and_operational_consequences_are_complete():
    root=ROOT/'outputs/visdrone_scene_consequences'
    summary=pd.read_csv(root/'stratified_path_summary.csv')
    pair=pd.read_csv(root/'stratified_pair_differences.csv')
    operational=pd.read_csv(root/'operational_metrics_overall.csv')
    assert set(summary.stratifier)=={'density','target_size','nearest_neighbor','occlusion'}
    assert len(pair)==24 and (pair.y11m_minus_asd_f1>0).all()
    assert len(operational)==10
    assert (operational.false_positives_per_image>=0).all()
    assert operational.false_negatives_per_100_scored_vehicles.between(0,100).all()
    assert operational.confirmed_vehicles_per_100_scored_alerts.between(0,100).all()
