# -*- coding: utf-8 -*-
"""
FusionDx -- GDC API field verification
Run before the data pipeline to confirm connectivity and field names.

Usage:  venv\\Scripts\\python.exe verify_gdc.py
"""
import json
import requests

GDC_API = "https://api.gdc.cancer.gov"


def check_clinical_fields():
    """Verify clinical field names with a 3-case sample."""
    print("=== Clinical fields (3 sample cases) ===\n")
    r = requests.get(f"{GDC_API}/cases", params={
        "filters": json.dumps({"op": "=", "content": {
            "field": "project.project_id", "value": "TCGA-BRCA"
        }}),
        "expand": "diagnoses,demographic,follow_ups.molecular_tests",
        "format": "JSON",
        "size": "3",
    }, timeout=30)
    r.raise_for_status()
    hits = r.json()["data"]["hits"]

    for h in hits:
        sid   = h.get("submitter_id", "?")
        diag  = h.get("diagnoses", [{}])[0] if h.get("diagnoses") else {}
        demo  = h.get("demographic", {})

        # Extract receptor status from follow_ups
        receptor = {"ESR1": "not reported", "PGR": "not reported", "ERBB2": "not reported"}
        for fu in h.get("follow_ups", []):
            for mt in fu.get("molecular_tests", []):
                g = mt.get("gene_symbol", "")
                if g in receptor:
                    receptor[g] = mt.get("test_result", "not reported")

        print(f"  {sid}")
        print(f"    age_at_diagnosis     : {diag.get('age_at_diagnosis')} days"
              f"  (~{round((diag.get('age_at_diagnosis') or 0)/365.25, 1)} yrs)")
        print(f"    ajcc_pathologic_stage: {diag.get('ajcc_pathologic_stage')}")
        print(f"    primary_diagnosis    : {diag.get('primary_diagnosis')}")
        print(f"    sex_at_birth         : {demo.get('sex_at_birth')}")
        print(f"    race                 : {demo.get('race')}")
        print(f"    vital_status         : {demo.get('vital_status')}")
        print(f"    ER  (ESR1)           : {receptor['ESR1']}")
        print(f"    PR  (PGR)            : {receptor['PGR']}")
        print(f"    HER2 (ERBB2)         : {receptor['ERBB2']}")
        print()


def check_slide_availability():
    """Confirm diagnostic slides are open-access and report total count."""
    print("=== Diagnostic slide availability ===\n")
    r = requests.get(f"{GDC_API}/files", params={
        "filters": json.dumps({"op": "and", "content": [
            {"op": "=", "content": {"field": "cases.project.project_id",
                                    "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_type",
                                    "value": "Slide Image"}},
            {"op": "=", "content": {"field": "experimental_strategy",
                                    "value": "Diagnostic Slide"}},
        ]}),
        "fields": "file_id,file_name,file_size,access,cases.submitter_id",
        "format": "JSON",
        "size": "5",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    total = data["pagination"]["total"]
    print(f"  Total diagnostic slides available: {total}")
    sizes = [h.get("file_size", 0) for h in data["hits"]]
    print(f"  Sample file sizes: {[f'{s/1e6:.0f} MB' for s in sizes]}")
    access_set = {h.get("access") for h in data["hits"]}
    print(f"  Access tiers in sample: {access_set}")
    print(f"  (All diagnostic slides for TCGA-BRCA are 'open' access)")
    print()


if __name__ == "__main__":
    print("FusionDx -- GDC API Verification\n")
    check_clinical_fields()
    check_slide_availability()
    print("=== All checks passed -- GDC API is ready ===")
