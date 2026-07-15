from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_rule_grid_is_registered():
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

def test_claim_registry_has_forbidden_wording():
    assert 'forbidden_wording' in (ROOT/'configs/CLAIM_REGISTRY.yaml').read_text()

def test_reference_results_are_scoped():
    d=pd.read_parquet(ROOT/'outputs/rule_grid_reference/metrics_long.parquet')
    assert set(d.source)=={'aitod','uavdt','visdrone'}
    assert set(d.confidence)=={0.25} and set(d.iou)=={0.25}

def test_reproduction_entrypoint_exists():
    assert (ROOT/'sourceaudit/reproduce.py').exists()

def test_registered_contract_path_is_in_manuscript():
    tex=(ROOT/'manuscript/final_grsl.tex').read_text()
    assert 'We register four contracts as ordered pairs' in tex
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

def test_metric_rank_bootstrap_and_ap_crosscheck_are_registered():
    boot=pd.read_csv(ROOT/'outputs/metric_qualified_rank/paired_bootstrap.csv')
    assert set(boot.resampling_unit)=={'image','sequence'}
    assert set(boot.replicates)=={10000}
    seq=boot[boot.resampling_unit=='sequence']
    assert set(seq.sampling_units)=={76}
    cross=pd.read_csv(ROOT/'outputs/metric_qualified_rank/evaluator_crosscheck.csv')
    assert cross['pass'].all()
    assert cross.absolute_difference.max() < 1e-12
