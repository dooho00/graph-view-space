import os
import datetime
import pandas as pd

def print_final_summary(all_summaries, dataset_list, results_dir=None):
    """Save final summary tables for all datasets to both text and CSV files."""
    
    # Use provided results_dir or create a new timestamped one
    if results_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = f'results/{timestamp}'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate filenames
    txt_filename = f'{results_dir}/summary_all_datasets.txt'
    
    # Save separate CSV files for each experiment
    save_separate_experiment_csvs(all_summaries, dataset_list, results_dir)
    
    # Then save text format (keeping original functionality)
    save_results_to_text(all_summaries, dataset_list, txt_filename, results_dir)
    
    print(f"Summary saved to:")
    print(f"  Text: {txt_filename}")
    print(f"  CSV files saved in: {results_dir}")

def save_separate_experiment_csvs(all_summaries, dataset_list, results_dir):
    """Save frozen-RGVT adaptation results to CSV."""
    
    adaptation_data = []
    for dataset in dataset_list:
        if dataset not in all_summaries:
            continue
        s = all_summaries[dataset]
        if 'RGVT_best_val' in s:
            row = {
                'Dataset': dataset,
                'RGVT_Val_Acc': s['RGVT_best_val'] * 100,
                'RGVT_Test_Acc': s['RGVT_best_test'] * 100,
                'Best_Depth_L': s.get('best_depth', ''),
            }
            adaptation_data.append(row)
    
    if adaptation_data:
        df = pd.DataFrame(adaptation_data)
        df = df.round(2)
        df.to_csv(f'{results_dir}/rgvt_adaptation_results.csv', index=False)

def save_results_to_text(all_summaries, dataset_list, txt_filename, results_dir):
    """Save results in original text format."""
    
    # Collect all output in a list
    output_lines = []
    
    output_lines.append('\n=== RGVT Predictor Adaptation Summary Across All Datasets ===')
    header = f"{'Dataset':<15} | {'RGVT Val (%)':<12} | {'Best L':<6} | {'RGVT Test (%)':<13}"
    output_lines.append(header)
    output_lines.append('-' * len(header))
    
    for ds in dataset_list:
        if ds in all_summaries:
            s = all_summaries[ds]
            output_lines.append(f"{ds:<15} | "
                  f"{100 * s['RGVT_best_val']:<12.2f} | "
                  f"{str(s['best_depth']):<6} | "
                  f"{100 * s['RGVT_best_test']:<13.2f}")

    # Average across all datasets
    avg_val = sum(s['RGVT_best_val'] for s in all_summaries.values() if 'RGVT_best_val' in s) / len(all_summaries)
    avg_test = sum(s['RGVT_best_test'] for s in all_summaries.values() if 'RGVT_best_test' in s) / len(all_summaries)
    
    output_lines.append(f"{'Average':<15} | "
          f"{100 * avg_val:<12.2f} | "
          f"{'':<6} | "
          f"{100 * avg_test:<13.2f}")
    
    # Add a note about the results directory
    output_lines.append(f"\nResults saved to: {results_dir}")
    
    # Write all output to file
    with open(txt_filename, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Summary saved to: {txt_filename}")
