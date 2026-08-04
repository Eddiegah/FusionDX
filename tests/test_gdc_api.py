# -*- coding: utf-8 -*-
"""
Tests for GDC API connectivity and field correctness.

These tests hit the live GDC API and require internet access.
Mark them with -m gdc to run selectively:
    pytest -m gdc tests/test_gdc_api.py
"""

import json
import pytest
import requests

GDC_API = "https://api.gdc.cancer.gov"

pytestmark = pytest.mark.gdc  # all tests in this file require internet + GDC


@pytest.fixture(scope="module")
def sample_cases():
    """Fetch 3 TCGA-BRCA cases with full clinical expansion."""
    r = requests.get(f"{GDC_API}/cases", params={
        "filters": json.dumps({"op": "=", "content": {
            "field": "project.project_id", "value": "TCGA-BRCA"
        }}),
        "expand": "diagnoses,demographic,follow_ups.molecular_tests",
        "format": "JSON",
        "size": "3",
    }, timeout=30)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def test_gdc_api_reachable():
    r = requests.get(f"{GDC_API}/status", timeout=15)
    assert r.status_code == 200
    assert r.json().get("status") == "OK"


def test_tcga_brca_cases_returned(sample_cases):
    assert len(sample_cases) == 3


def test_submitter_ids_look_like_tcga(sample_cases):
    for h in sample_cases:
        sid = h.get("submitter_id", "")
        assert sid.startswith("TCGA-"), f"Unexpected submitter_id: {sid}"


def test_age_at_diagnosis_is_numeric(sample_cases):
    for h in sample_cases:
        for d in h.get("diagnoses", []):
            age = d.get("age_at_diagnosis")
            if age is not None:
                assert isinstance(age, (int, float))
                assert 10 * 365 < age < 120 * 365, f"Age {age} days out of expected range"


def test_ajcc_stage_field_present(sample_cases):
    """Confirm the correct field name is ajcc_pathologic_stage."""
    found_stage = False
    for h in sample_cases:
        for d in h.get("diagnoses", []):
            if d.get("ajcc_pathologic_stage") is not None:
                found_stage = True
                break
    assert found_stage, "No ajcc_pathologic_stage found in any sample case"


def test_primary_diagnosis_non_empty(sample_cases):
    for h in sample_cases:
        for d in h.get("diagnoses", []):
            dx = d.get("primary_diagnosis")
            assert dx is not None and len(dx) > 0


def test_receptor_status_in_follow_ups(sample_cases):
    """ER/PR/HER2 should be in follow_ups.molecular_tests."""
    gene_symbols_found = set()
    for h in sample_cases:
        for fu in h.get("follow_ups", []):
            for mt in fu.get("molecular_tests", []):
                gene_symbols_found.add(mt.get("gene_symbol"))
    # At least one of ESR1/PGR/ERBB2 should appear across 3 patients
    receptor_genes = {"ESR1", "PGR", "ERBB2"}
    assert len(gene_symbols_found & receptor_genes) > 0, (
        f"No receptor genes found in follow_ups. Found: {gene_symbols_found}"
    )


def test_sex_at_birth_field_present(sample_cases):
    """Confirm demographic uses sex_at_birth not gender."""
    for h in sample_cases:
        demo = h.get("demographic", {})
        assert "sex_at_birth" in demo, (
            f"sex_at_birth missing in demographic for {h.get('submitter_id')}"
        )


def test_diagnostic_slides_open_access():
    """1133 diagnostic slides should be available and open-access."""
    r = requests.get(f"{GDC_API}/files", params={
        "filters": json.dumps({"op": "and", "content": [
            {"op": "=", "content": {"field": "cases.project.project_id",
                                    "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy",
                                    "value": "Diagnostic Slide"}},
        ]}),
        "fields": "file_id,access",
        "format": "JSON",
        "size": "5",
    }, timeout=30)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]
    total = r.json()["data"]["pagination"]["total"]

    assert total >= 1000, f"Expected 1000+ slides, got {total}"
    for h in hits:
        assert h.get("access") == "open", (
            f"Slide {h['file_id']} has access={h.get('access')}, expected 'open'"
        )
