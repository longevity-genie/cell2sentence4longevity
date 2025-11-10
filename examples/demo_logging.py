#!/usr/bin/env python3
"""Quick demonstration of enhanced logging features.

This script creates a sample dataset and runs through the age extraction
and filtering steps to demonstrate the enhanced logging.
"""

from pathlib import Path
import polars as pl
from eliot import start_action
from pycomfort.logging import to_nice_file

from cell2sentence4longevity.preprocessing import add_age_and_cleanup


def create_sample_data(output_dir: Path) -> Path:
    """Create sample parquet data for testing.
    
    Args:
        output_dir: Directory to save sample data
        
    Returns:
        Path to sample data directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data with various age formats
    data = {
        'cell_sentence': [f'GENE1 GENE2 GENE3' for _ in range(1000)],
        'development_stage': [
            # Valid ages
            '22-year-old stage' for _ in range(700)
        ] + [
            '25-year-old stage' for _ in range(200)
        ] + [
            # Invalid formats that will result in null age
            'adult', 'unknown', None, '', 'pediatric',
            'newborn', '5-month-old', 'stage 22', 'twenty-year-old',
            'young adult'
        ] * 10,
        'cell_type': ['T cell' for _ in range(1000)],
        'donor_id': [f'donor_{i%10}' for i in range(1000)],
    }
    
    df = pl.DataFrame(data)
    
    # Save as chunk
    chunk_path = output_dir / 'chunk_0000.parquet'
    df.write_parquet(chunk_path)
    
    print(f"\n✓ Created sample data with {len(df)} cells:")
    print(f"  - 700 cells with age 22")
    print(f"  - 200 cells with age 25")
    print(f"  - 100 cells with invalid age formats")
    print(f"  - Saved to: {chunk_path}\n")
    
    return output_dir


def main():
    """Main demonstration function."""
    print("\n" + "="*80)
    print("ENHANCED LOGGING DEMONSTRATION")
    print("="*80)
    
    # Setup
    demo_dir = Path("./demo_logging")
    data_dir = demo_dir / "data"
    log_dir = demo_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = log_dir / "demo.log"
    json_log = log_dir / "demo.json"
    to_nice_file(output_file=json_log, rendered_file=log_file)
    
    with start_action(action_type="demo_enhanced_logging"):
        # Create sample data
        print("\n📝 Step 1: Creating sample dataset")
        data_path = create_sample_data(data_dir)
        
        # Run age extraction with enhanced logging
        print("📝 Step 2: Running age extraction with enhanced logging")
        add_age_and_cleanup(data_path)
        
        print("\n✓ Processing complete!")
        print(f"\n📊 Check the logs to see detailed statistics:")
        print(f"   - Human-readable: {log_file}")
        print(f"   - Machine-readable: {json_log}")
        
        # Read and display key metrics from JSON logs
        print("\n" + "="*80)
        print("KEY METRICS FROM LOGS")
        print("="*80)
        
        import json
        with open(json_log) as f:
            for line in f:
                event = json.loads(line)
                msg_type = event.get('message_type')
                
                if msg_type == 'age_extraction_summary':
                    print("\n📊 Age Extraction Summary:")
                    print(f"   Total cells:        {event.get('total_cells', 0):,}")
                    print(f"   Valid age:          {event.get('cells_with_valid_age', 0):,} "
                          f"({event.get('valid_age_percentage', 0):.1f}%)")
                    print(f"   Null age:           {event.get('cells_with_null_age', 0):,} "
                          f"({event.get('null_age_percentage', 0):.1f}%)")
                
                elif msg_type == 'sample_null_age_development_stages':
                    print("\n⚠️  Sample cells with null age (for debugging):")
                    for i, sample in enumerate(event.get('samples', [])[:5], 1):
                        dev_stage = sample.get('development_stage', 'N/A')
                        print(f"      {i}. development_stage: '{dev_stage}'")
                
                elif msg_type == 'age_distribution':
                    print("\n📈 Age Distribution:")
                    for item in event.get('distribution', []):
                        age = item.get('age')
                        count = item.get('count')
                        print(f"      Age {age}: {count:,} cells")
        
        print("\n" + "="*80)
        print("\n💡 Key Features Demonstrated:")
        print("   ✓ Total cell count tracking")
        print("   ✓ Valid vs null age statistics")
        print("   ✓ Sample of problematic development_stage values")
        print("   ✓ Age distribution for valid ages")
        print("   ✓ Percentage calculations for easy assessment")
        
        print("\n🎯 Use Cases:")
        print("   - Quickly identify if a dataset has age extraction issues")
        print("   - See exact format of problematic age values")
        print("   - Understand what percentage of data will be lost")
        print("   - Debug age extraction regex patterns")
        
        print("\n📚 For more information, see: docs/LOGGING.md")
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

