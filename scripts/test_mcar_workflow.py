#!/usr/bin/env python3
"""
Test the full MCAR workflow: Generate synthetic MCAR data and validate with Little's test
"""

import pandas as pd
import numpy as np
from lacuna.data.simulators import MCARSimulator
from lacuna import MCARDetector
from lacuna.utils.forge_config import LACUNAForgeConfig

def test_mcar_workflow():
    """Test complete MCAR generation and detection workflow"""
    
    print("🧪 Testing MCAR Workflow")
    print("=" * 50)
    
    # Initialize components with forge config
    config = LACUNAForgeConfig()
    simulator = MCARSimulator(random_state=42)
    detector = MCARDetector(alpha=0.05)
    
    # Step 1: Generate or load base complete data
    print("\n📊 Step 1: Loading base complete data...")
    try:
        # Try to load real heart disease data (uses forge config internally)
        base_data = simulator.load_heart_disease_data()
        print(f"✅ Loaded heart disease data: {base_data.shape}")
        print(f"   Variables: {list(base_data.columns)}")
    except Exception as e:
        print(f"⚠️ Failed to load real data ({e}), using synthetic...")
        base_data = simulator._create_synthetic_health_data(n_patients=200)
        print(f"✅ Created synthetic data: {base_data.shape}")
    
    # Show base data stats
    print(f"   Complete cases: {len(base_data)}")
    print(f"   No missing values: {not base_data.isnull().any().any()}")
    
    # Step 2: Generate MCAR data with different patterns
    print("\n🎲 Step 2: Generating MCAR data...")
    
    scenarios = [
        {"name": "Low MCAR (10%)", "rate": 0.10, "pattern": "uniform"},
        {"name": "Medium MCAR (20%)", "rate": 0.20, "pattern": "uniform"},
        {"name": "High MCAR (35%)", "rate": 0.35, "pattern": "uniform"},
        {"name": "Variable-specific MCAR", "rate": 0.20, "pattern": "variable_specific"},
        {"name": "Block MCAR", "rate": 0.20, "pattern": "block"}
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n   🔹 {scenario['name']}:")
        
        # Generate MCAR data
        mcar_result = simulator.generate_mcar_data(
            base_data=base_data,
            missing_rate=scenario['rate'],
            pattern_type=scenario['pattern']
        )
        
        mcar_data = mcar_result['mcar_data']
        actual_rate = mcar_data.isnull().sum().sum() / mcar_data.size
        
        print(f"      Target missing rate: {scenario['rate']:.1%}")
        print(f"      Actual missing rate: {actual_rate:.1%}")
        print(f"      Missing cells: {mcar_data.isnull().sum().sum()}")
        
        # Step 3: Run Little's MCAR test
        print(f"      Running Little's test...")
        detection_result = detector.test(mcar_data)
        
        # Store results
        result_summary = {
            'scenario': scenario['name'],
            'target_rate': scenario['rate'],
            'actual_rate': actual_rate,
            'pattern': scenario['pattern'],
            'detected_as_mcar': detection_result['is_mcar_plausible'],
            'p_value': detection_result['p_value'],
            'test_statistic': detection_result['test_statistic'],
            'confidence': detection_result['confidence'],
            'recommendation': detection_result['recommendation']
        }
        results.append(result_summary)
        
        # Print results
        if detection_result['is_mcar_plausible']:
            status = "✅ CORRECTLY detected as MCAR"
        else:
            status = "❌ INCORRECTLY rejected MCAR"
        
        print(f"      {status}")
        print(f"      P-value: {detection_result['p_value']:.4f}")
        print(f"      Confidence: {detection_result['confidence']}")
    
    # Step 4: Summary analysis
    print("\n" + "=" * 50)
    print("📋 SUMMARY RESULTS")
    print("=" * 50)
    
    # Create results table
    results_df = pd.DataFrame(results)
    
    # Save results to forge location
    results_file = config.EVALUATIONS_DIR / "mcar_workflow_test_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"💾 Results saved to: {results_file}")
    
    print(f"\nScenario Performance:")
    for _, row in results_df.iterrows():
        status_icon = "✅" if row['detected_as_mcar'] else "❌"
        print(f"{status_icon} {row['scenario']:<25} | "
              f"Missing: {row['actual_rate']:.1%} | "
              f"P-val: {row['p_value']:.3f} | "
              f"Detected MCAR: {row['detected_as_mcar']}")
    
    # Overall statistics
    correct_detections = results_df['detected_as_mcar'].sum()
    total_scenarios = len(results_df)
    accuracy = correct_detections / total_scenarios
    
    print(f"\n📊 Overall Performance:")
    print(f"   Correct MCAR detections: {correct_detections}/{total_scenarios}")
    print(f"   Detection accuracy: {accuracy:.1%}")
    
    # Statistical insights
    avg_p_value = results_df['p_value'].mean()
    print(f"   Average p-value: {avg_p_value:.3f}")
    print(f"   All p-values > 0.05: {(results_df['p_value'] > 0.05).all()}")
    
    return results_df

if __name__ == "__main__":
    results = test_mcar_workflow()