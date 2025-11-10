"""Example script to demonstrate enhanced logging features.

This script demonstrates how to use the enhanced logging to track
discarded cells and other processing issues.
"""

from pathlib import Path
import json
from typing import Dict, Any

def analyze_logs(log_file: Path) -> Dict[str, Any]:
    """Analyze logs and extract key metrics.
    
    Args:
        log_file: Path to the JSON log file
        
    Returns:
        Dictionary with key metrics
    """
    metrics = {
        'total_cells': 0,
        'cells_with_valid_age': 0,
        'cells_with_null_age': 0,
        'genes_total': 0,
        'genes_mapped_hgnc': 0,
        'genes_mapped_fallback': 0,
        'genes_unmapped': 0,
        'cells_discarded_in_split': 0,
        'missing_columns': [],
        'sample_null_ages': [],
    }
    
    with open(log_file) as f:
        for line in f:
            event = json.loads(line)
            msg_type = event.get('message_type')
            
            # H5AD loading metadata
            if msg_type == 'warning_missing_column':
                metrics['missing_columns'].append(event.get('column'))
            
            # Gene mapping stats
            elif msg_type == 'gene_mapping_summary':
                metrics['genes_total'] = event.get('total_genes', 0)
                metrics['genes_mapped_hgnc'] = event.get('mapped_via_hgnc', 0)
                metrics['genes_mapped_fallback'] = event.get('mapped_via_fallback', 0)
                metrics['genes_unmapped'] = event.get('unmapped_using_ensembl_id', 0)
            
            # Age extraction stats
            elif msg_type == 'age_extraction_summary':
                metrics['total_cells'] = event.get('total_cells', 0)
                metrics['cells_with_valid_age'] = event.get('cells_with_valid_age', 0)
                metrics['cells_with_null_age'] = event.get('cells_with_null_age', 0)
            
            elif msg_type == 'sample_null_age_development_stages':
                metrics['sample_null_ages'] = event.get('samples', [])
            
            # Train/test split filtering
            elif msg_type == 'filtering_summary':
                metrics['cells_discarded_in_split'] = event.get('cells_discarded', 0)
    
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Print metrics in a human-readable format.
    
    Args:
        metrics: Dictionary with processing metrics
    """
    print("\n" + "="*80)
    print("PROCESSING METRICS SUMMARY")
    print("="*80)
    
    # Missing columns
    if metrics['missing_columns']:
        print("\n⚠️  MISSING COLUMNS:")
        for col in metrics['missing_columns']:
            print(f"   - {col}")
    else:
        print("\n✓ All required columns present")
    
    # Gene mapping
    if metrics['genes_total'] > 0:
        print(f"\n📊 GENE MAPPING:")
        print(f"   Total genes:        {metrics['genes_total']:,}")
        print(f"   Mapped via HGNC:    {metrics['genes_mapped_hgnc']:,} "
              f"({metrics['genes_mapped_hgnc']/metrics['genes_total']*100:.1f}%)")
        print(f"   Mapped via fallback: {metrics['genes_mapped_fallback']:,} "
              f"({metrics['genes_mapped_fallback']/metrics['genes_total']*100:.1f}%)")
        print(f"   Unmapped:           {metrics['genes_unmapped']:,} "
              f"({metrics['genes_unmapped']/metrics['genes_total']*100:.1f}%)")
    
    # Age extraction
    if metrics['total_cells'] > 0:
        print(f"\n📊 AGE EXTRACTION:")
        print(f"   Total cells:        {metrics['total_cells']:,}")
        print(f"   Valid age:          {metrics['cells_with_valid_age']:,} "
              f"({metrics['cells_with_valid_age']/metrics['total_cells']*100:.1f}%)")
        print(f"   Null age:           {metrics['cells_with_null_age']:,} "
              f"({metrics['cells_with_null_age']/metrics['total_cells']*100:.1f}%)")
        
        if metrics['sample_null_ages']:
            print("\n   Sample null age cases (for debugging):")
            for i, sample in enumerate(metrics['sample_null_ages'][:5], 1):
                dev_stage = sample.get('development_stage', 'N/A')
                print(f"      {i}. development_stage: {dev_stage}")
    
    # Discarded cells
    if metrics['cells_discarded_in_split'] > 0:
        print(f"\n⚠️  CELLS DISCARDED IN TRAIN/TEST SPLIT:")
        print(f"   Discarded:          {metrics['cells_discarded_in_split']:,}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function to demonstrate log analysis."""
    # Example: Analyze logs from a test run
    log_dir = Path("logs")
    
    # Find the most recent test log
    test_logs = sorted(log_dir.glob("test_*/integration_test.json"))
    
    if not test_logs:
        print("No test logs found. Please run the integration test first:")
        print("  uv run pytest tests/test_integration.py::TestLogging -v")
        return
    
    latest_log = test_logs[-1]
    print(f"\nAnalyzing log file: {latest_log}")
    
    metrics = analyze_logs(latest_log)
    print_metrics(metrics)
    
    # Provide recommendations
    print("📋 RECOMMENDATIONS:")
    
    if metrics['cells_with_null_age'] / max(metrics['total_cells'], 1) > 0.1:
        print("   ⚠️  More than 10% of cells have null age. Consider:")
        print("      - Checking the development_stage format in your dataset")
        print("      - Reviewing sample_null_age_development_stages in logs")
        print("      - Modifying the age extraction regex if needed")
    
    if metrics['genes_unmapped'] / max(metrics['genes_total'], 1) > 0.05:
        print("   ⚠️  More than 5% of genes are unmapped. Consider:")
        print("      - Checking if gene IDs are in Ensembl format")
        print("      - Updating the HGNC mapper data")
        print("      - Reviewing sample_unmapped_genes in logs")
    
    if not metrics['missing_columns'] and \
       metrics['cells_with_null_age'] / max(metrics['total_cells'], 1) < 0.1 and \
       metrics['genes_unmapped'] / max(metrics['genes_total'], 1) < 0.05:
        print("   ✓ All metrics look good! Pipeline is working well.")
    
    print()


if __name__ == "__main__":
    main()

