import logging
import json
import csv
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
import pandas as pd
from utils.semantic_combiner import combine as semantic_combine
from utils.semantic_combiner_enhanced import combine as semantic_enhanced_combine
from utils.semantic_combiner_adaptive import combine as semantic_adaptive_combine
from utils.simple_combiner import combine as simple_combine
from utils.weighted_combiner import combine as weighted_combine
from utils.adaptive_combiner import combine as adaptive_combine
from utils.adaptive_rule_combiner import combine as adaptive_rule_combine

logger = logging.getLogger(__name__)

def test_combiners(transcription, diarization, pipeline_model, output_directory):
    combiners = {
        # 'semantic': semantic_combine,
        'semantic_enhanced': semantic_enhanced_combine,
        # 'semantic_adaptive': semantic_adaptive_combine,
        # 'simple': simple_combine,
        # 'weighted': weighted_combine,
        # 'adaptive': adaptive_combine,
        # 'adaptive_rule': adaptive_rule_combine
    }

    results = {}
    all_segments = []

    for name, combiner in combiners.items():
        logger.info(f"Starting test for {name} combiner...")
        try:
            logger.info(f"Calling {name} combiner...")
            result = combiner(transcription, diarization)
            logger.info(f"{name} combiner returned result with {len(result)} segments.")
            results[name] = result
            
            # Save individual combiner results
            save_combiner_result(result, name, output_directory)
            create_pdf_result(result, name, output_directory)
            
            # Prepare data for CSV
            for segment in result:
                segment['combiner'] = name
                all_segments.append(segment)
            
            logger.info(f"{name} combiner test completed successfully.")
        except Exception as e:
            logger.error(f"Error testing {name} combiner: {str(e)}", exc_info=True)
            results[name] = None

    # Save comprehensive CSV
    save_comprehensive_csv(all_segments, output_directory)

    # Compare and save summary
    comparison = compare_results(results)
    save_comparison_results(comparison, output_directory)

    # Visualize results
    visualize_results(results, output_directory)

    return results

def save_combiner_result(result, combiner_name, output_directory):
    file_path = os.path.join(output_directory, f"{combiner_name}_result.json")
    with open(file_path, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved {combiner_name} result to {file_path}")

def create_pdf_result(result, combiner_name, output_directory):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=f"Results for {combiner_name} combiner", ln=1, align='C')
    pdf.ln(10)
    
    for segment in result:
        pdf.multi_cell(0, 10, txt=f"Speaker {segment['speaker']}: {segment['text']}", align='L')
        pdf.cell(200, 10, txt=f"Start: {segment['start']:.2f}s, End: {segment['end']:.2f}s", ln=1)
        pdf.ln(5)
    
    pdf_path = os.path.join(output_directory, f"{combiner_name}_result.pdf")
    pdf.output(pdf_path)
    logger.info(f"Saved {combiner_name} PDF result to {pdf_path}")

def save_comprehensive_csv(all_segments, output_directory):
    df = pd.DataFrame(all_segments)
    csv_path = os.path.join(output_directory, "all_combiners_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved comprehensive CSV to {csv_path}")

def compare_results(results):
    comparison = {}
    for name, result in results.items():
        if result:
            comparison[name] = {
                'num_segments': len(result),
                'avg_segment_duration': calculate_avg_segment_duration(result),
                'num_speaker_changes': count_speaker_changes(result),
                'total_duration': calculate_total_duration(result)
            }
    return comparison

def calculate_avg_segment_duration(result):
    durations = [(segment['end'] - segment['start']) for segment in result]
    return sum(durations) / len(durations) if durations else 0

def count_speaker_changes(result):
    return sum(1 for i in range(1, len(result)) if result[i]['speaker'] != result[i-1]['speaker'])

def calculate_total_duration(result):
    return result[-1]['end'] - result[0]['start'] if result else 0

def save_comparison_results(comparison, output_directory):
    # Save as JSON
    json_path = os.path.join(output_directory, "combiner_comparison.json")
    with open(json_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Save as CSV
    csv_path = os.path.join(output_directory, "combiner_comparison.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Combiner', 'Num Segments', 'Avg Segment Duration', 'Num Speaker Changes', 'Total Duration'])
        for name, metrics in comparison.items():
            writer.writerow([name, metrics['num_segments'], metrics['avg_segment_duration'], 
                             metrics['num_speaker_changes'], metrics['total_duration']])
    
    logger.info(f"Saved comparison results to {json_path} and {csv_path}")

def visualize_results(results, output_directory):
    plt.figure(figsize=(12, 6))
    for name, result in results.items():
        if result:
            speakers = [segment['speaker'] for segment in result]
            starts = [segment['start'] for segment in result]
            durations = [segment['end'] - segment['start'] for segment in result]
            plt.barh(y=name, width=durations, left=starts, height=0.5)

    plt.title("Speaker Segments Comparison")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Combiner")
    plt.tight_layout()
    
    viz_path = os.path.join(output_directory, "combiner_visualization.png")
    plt.savefig(viz_path)
    plt.close()
    logger.info(f"Saved visualization to {viz_path}")