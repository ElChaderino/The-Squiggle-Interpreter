import os
import sys
import threading
import time
import subprocess
import numpy as np
import matplotlib
import multiprocessing as mp
import io
import signal
from pathlib import Path
import mne
from modules import clinical_report, pyramid_model, data_to_csv, phenotype, pdf_report_builder
from modules.phenotype import classify_eeg_profile
from modules.vigilance import plot_vigilance_hypnogram
# Updated module imports - ensure processing and plotting are imported
from modules import io_utils, processing, plotting, report, clinical, vigilance, stats_utils, display_utils
from mne.io.constants import FIFF
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from scipy.stats import zscore, pearsonr
import pandas as pd
import logging # Import logging
import matplotlib.pyplot as plt # Import pyplot for closing plots

# Try to import psd_welch; if unavailable, fall back to psd_array_welch.
try:
    from mne.time_frequency import psd_welch
except ImportError:
    from mne.time_frequency import psd_array_welch as psd_welch


# --- Task Execution Helpers ---
def execute_task(func, args):
    # Wrapper to execute function and handle potential exceptions
    try:
        return func(*args)
    except Exception as e:
        # Log the error or return a specific indicator
        logging.error(f"Error in task {func.__name__}: {e}", exc_info=True)
        return None # Indicate failure


def execute_task_with_queue(func, args, queue):
    """
    Execute a function with given arguments and put the result into a queue.
    Includes basic error handling.
    """
    try:
        result = func(*args)
        queue.put(result)
    except Exception as e:
        logging.error(f"Error executing {func.__name__} in queue task: {e}", exc_info=True)
        # Optionally put an error indicator into the queue
        queue.put(f"ERROR: {e}")


# --- Signal Handling ---
def sigint_handler(signum, frame):
    print("\n🛑 SIGINT received, attempting graceful shutdown...")
    if 'stop_event' in globals():
        stop_event.set() # Signal threads/processes relying on this
    # Add any other necessary cleanup before exiting
    # Consider setting a global flag checked by long processes
    print("Exiting due to SIGINT.")
    sys.exit(0)


# --- Refactored Pipeline Functions ---

def parse_arguments():
    """Parses command line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        prog='The Squiggle Interpreter',
        description='Comprehensive EEG Analysis & Clinical Report Generation'
    )
    # Input/Output Arguments
    parser.add_argument('--project_dir', type=str, default=os.getcwd(),
                        help="Project root directory (default: current working directory). Output and logs will be saved here.")
    parser.add_argument('--input_dir', type=str, default=None,
                        help="Directory containing input EDF files (default: project directory itself).")

    # Processing Options
    parser.add_argument('--use_csd', default=None, action=argparse.BooleanOptionalAction,
                        help="Apply Current Source Density (CSD). If not provided, you will be prompted.")
    parser.add_argument('--zscore_method', type=str, default=None,
                        help="Z-score normalization method. If not provided, you will be prompted.")
    parser.add_argument('--num_workers', type=int, default=None,
                        help="Number of worker processes for parallel computation (default: all available CPU cores).")

    # Optional Filters
    parser.add_argument('--lpf', type=float, default=None,
                        help="Apply a low-pass filter at the specified frequency (e.g., 40.0).")
    parser.add_argument('--notch', type=float, default=None,
                        help="Apply a notch filter at the specified frequency (e.g., 50.0 or 60.0).")

    # Output Options
    parser.add_argument('--skip_report', action='store_true',
                        help="Skip generating the main HTML/PDF report.")
    parser.add_argument('--skip_phenotype', action='store_true',
                        help="Skip phenotype classification.")
    parser.add_argument('--skip_vigilance', action='store_true',
                        help="Skip vigilance analysis.")
    parser.add_argument('--skip_source', action='store_true',
                        help="Skip source localization.")

    # Special Modes
    parser.add_argument('--live_display', action='store_true',
                        help="Run the live EEG display simulation (requires manual stop with Ctrl+C).")
    parser.add_argument('--export_csv', action='store_true',
                        help="Run in CSV export mode. Requires --edf_path and --output_csv.")
    parser.add_argument('--edf_path', type=str, help="Path to the EDF file for CSV export mode.")
    parser.add_argument('--epoch_length', type=float, default=2.0,
                        help="Epoch length (in seconds) for CSV export (default: 2.0).")
    parser.add_argument('--output_csv', type=str, help="Output CSV file path for CSV export mode.")

    # Optional argument for clinical outcome comparison
    parser.add_argument('--clinical_csv', type=str, default=None,
                        help="Path to optional CSV file containing clinical outcome scores for Z-score comparison.")

    args = parser.parse_args()

    # Post-process/validate args if needed
    if args.input_dir is None:
        args.input_dir = args.project_dir
    if args.num_workers is None:
        args.num_workers = mp.cpu_count()

    if args.export_csv:
        if not args.edf_path or not args.output_csv:
            parser.error("--export_csv mode requires --edf_path and --output_csv.")

    return args


# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(project_dir, files, use_csd, lpf_freq=None, notch_freq=None):
    """
    Loads and preprocesses EO and EC data for a subject.
    Applies CSD if requested. Returns raw objects and potentially a config update.
    """
    eo_filename = files.get("EO")
    ec_filename = files.get("EC")

    raw_eo, raw_ec = None, None
    raw_graph_eo, raw_graph_ec = None, None # For CSD versions used in plotting

    # Load EO data
    if eo_filename:
        eo_filepath = Path(project_dir) / eo_filename # Use project_dir which is now absolute path from main
        if eo_filepath.exists():
            try:
                logging.info(f"Loading EO file: {eo_filepath}")
                raw_eo = io_utils.load_eeg_data(str(eo_filepath)) # Pass CSD=False initially
                if raw_eo and use_csd:
                     # Compute CSD version specifically for graphing if requested
                     try:
                          logging.info("Applying CSD to EO data for graphing...")
                          raw_graph_eo = mne.preprocessing.compute_current_source_density(raw_eo.copy())
                          logging.info("CSD applied successfully to EO data.")
                     except Exception as e_csd:
                          logging.warning(f"Could not apply CSD to EO data: {e_csd}. Using raw data for graphs.")
                          raw_graph_eo = raw_eo # Fallback
                else:
                     raw_graph_eo = raw_eo # Use raw data if CSD not requested or failed
            except Exception as e:
                logging.error(f"Failed to load or preprocess EO file {eo_filepath}: {e}", exc_info=True)
        else:
            logging.warning(f"EO file not found: {eo_filepath}")
    else:
        logging.info("No EO file specified for subject.")

    # Load EC data
    if ec_filename:
        ec_filepath = Path(project_dir) / ec_filename
        if ec_filepath.exists():
            try:
                logging.info(f"Loading EC file: {ec_filepath}")
                raw_ec = io_utils.load_eeg_data(str(ec_filepath)) # Pass CSD=False initially
                if raw_ec and use_csd:
                     # Compute CSD version
                     try:
                          logging.info("Applying CSD to EC data for graphing...")
                          raw_graph_ec = mne.preprocessing.compute_current_source_density(raw_ec.copy())
                          logging.info("CSD applied successfully to EC data.")
                     except Exception as e_csd:
                          logging.warning(f"Could not apply CSD to EC data: {e_csd}. Using raw data for graphs.")
                          raw_graph_ec = raw_ec # Fallback
                else:
                     raw_graph_ec = raw_ec # Use raw data if CSD not requested or failed
            except Exception as e:
                logging.error(f"Failed to load or preprocess EC file {ec_filepath}: {e}", exc_info=True)
        else:
            logging.warning(f"EC file not found: {ec_filepath}")
    else:
        logging.info("No EC file specified for subject.")

    # If CSD wasn't applied, ensure graph versions are same as raw
    if raw_eo and not raw_graph_eo: raw_graph_eo = raw_eo
    if raw_ec and not raw_graph_ec: raw_graph_ec = raw_ec

    # TODO: Return any config updates if needed? For now, just return data.
    config_update = {}
    return raw_eo, raw_ec, raw_graph_eo, raw_graph_ec, config_update


# --- Report Generation (Adapts to New Inputs) ---
def generate_reports(raw_eo, raw_ec, folders, subject_folder, subject, band_list, config,
                     computation_results, # Dict of computed data
                     plot_filenames): # Dict of relative plot filenames
    """Generates all reports (HTML, PDF, CSV) using computed data and plot filenames."""
    print(f"\n📄 Generating reports for subject {subject}...")
    report_start_time = time.time()
    subject_folder_path = Path(subject_folder) # Ensure Path object

    # --- CSV Data Export ---
    csv_path = Path(folders["data"]) / f"{subject}_eeg_features.csv"
    try:
        # Gather features needed for CSV export from computation_results
        # This needs careful implementation based on what save_features_to_csv expects
        # Example: just saving band powers for now
        features_to_save = {
            "band_powers_eo": computation_results.get("band_powers_eo"),
            "band_powers_ec": computation_results.get("band_powers_ec"),
            "zscores_eo": computation_results.get("zscores_eo"),
            # Add other relevant computed data...
        }
        # Assuming raw_eo info is representative if available
        info_for_csv = raw_eo.info if raw_eo else (raw_ec.info if raw_ec else None)

        if info_for_csv:
             # Call the function from the data_to_csv module
             data_to_csv.save_computed_features_to_csv(features_to_save, info_for_csv, str(csv_path))
             print(f"  ✅ Saved features to {csv_path}")
        else:
             print(f"  ⚠️ Skipping CSV export: No valid MNE info object available.")

    except Exception as e:
        logging.error(f"  ❌ Error saving features to CSV for {subject}: {e}", exc_info=True)

    # --- Main HTML/PDF Report Generation --- #
    if config.get('skip_report', False):
        print("  Skipping main HTML report generation as requested.")
    else:
        html_report_path = Path(folders["reports"]) / f"{subject}_comprehensive_report.html"
        pdf_report_path = Path(folders["reports"]) / f"{subject}_comprehensive_report.pdf"

        # --- Prepare data structure for the report template ---
        # The structure should match what report.build_html_report expects
        report_data = {}
        try:
             # Add plot filenames directly from the input dictionary
             # Ensure all top-level plot categories exist before accessing
             expected_plot_keys = ["topomaps", "waveforms", "coherence", "erp", 
                                   "zscores", "variance", "tfr", "ica", 
                                   "source_localization", "hypnograms"] # Add hypnograms here
             for key in expected_plot_keys:
                 if key not in plot_filenames:
                     logging.warning(f"Plot category '{key}' missing in plot_filenames. Initializing empty.")
                     # Initialize based on expected structure (assuming dicts, potentially nested)
                     if key in ["erp", "ica"]: 
                          plot_filenames[key] = {"EO": "", "EC": ""} # Expects single string path per condition
                     else: 
                          plot_filenames[key] = {"EO": {}, "EC": {}} # Expects dicts per condition
                 elif not isinstance(plot_filenames[key], dict): 
                      logging.warning(f"Plot category '{key}' is not a dict. Re-initializing empty.")
                      if key in ["erp", "ica"]: plot_filenames[key] = {"EO": "", "EC": ""} 
                      else: plot_filenames[key] = {"EO": {}, "EC": {}}
             
             report_data.update(plot_filenames) # Now update with sanitized plot_filenames

             # Add necessary metadata
             report_data['subject_id'] = subject
             report_data['band_list'] = band_list
             report_data['site_list'] = raw_eo.ch_names if raw_eo else (raw_ec.ch_names if raw_ec else [])

             # --- Ensure band keys exist within plot dictionaries --- #
             plot_types_with_bands = ["topomaps", "waveforms", "coherence", "zscores", "variance", "tfr", "source_localization"]
             for plot_type in plot_types_with_bands:
                 if plot_type in report_data:
                     for cond in ["EO", "EC"]:
                         if cond in report_data[plot_type] and isinstance(report_data[plot_type][cond], dict):
                             for band in band_list:
                                 # Ensure the band key exists, default to None if not
                                 report_data[plot_type][cond].setdefault(band, None) 
                         elif cond not in report_data[plot_type]: # Ensure condition dict exists
                             report_data[plot_type][cond] = {band: None for band in band_list}


             # Add phenotype results (assuming process_phenotype returned them)
             # You might need to fetch this from computation_results if stored there,
             # or load from the file saved by process_phenotype.
             # phenotype_data = computation_results.get('phenotype_result', {}) # Example
             phenotype_report_path = Path(subject_folder) / f"{subject}_phenotype_results.txt"
             phenotype_data = {}
             if phenotype_report_path.exists():
                  try:
                      with open(phenotype_report_path, "r", encoding="utf-8") as f:
                           lines = f.readlines()
                           for line in lines[2:]: # Skip header lines
                               if ":" in line:
                                    key, value = line.split(":", 1)
                                    phenotype_data[key.strip()] = value.strip()
                  except Exception as e_pheno_read:
                      print(f"  ⚠️ Could not read phenotype results file: {e_pheno_read}")
             report_data['phenotype'] = phenotype_data

             # Add hypnogram and strip chart filenames from vigilance output
             hypnograms = {"EO": {}, "EC": {}, "EO_CSD": {}, "EC_CSD": {}}
             strips = {"EO": {}, "EC": {}, "EO_CSD": {}, "EC_CSD": {}}
             vig_plot_dir = Path(folders["plots_vigilance"])
             if vig_plot_dir.exists():
                 for cond in hypnograms.keys():
                     for ch_name in report_data['site_list']:
                         base = f"{cond}_{ch_name}"
                         hypno_file = vig_plot_dir / f"{base}_hypnogram.png"
                         strip_file = vig_plot_dir / f"{base}_strip.png"
                         hypnograms[cond][ch_name] = str(hypno_file.relative_to(subject_folder_path)) if hypno_file.exists() else ""
                         strips[cond][ch_name] = str(strip_file.relative_to(subject_folder_path)) if strip_file.exists() else ""
             report_data['hypnograms'] = hypnograms
             report_data['strips'] = strips

             # Add paths relative to the subject folder for finding plots within HTML
             report_data['global_topomaps_path'] = str(Path(folders["plots_topo"]).relative_to(subject_folder_path))
             report_data['global_waveforms_path'] = str(Path(folders["plots_wave"]).relative_to(subject_folder_path))
             report_data['coherence_path'] = str(Path(folders["plots_coh"]).relative_to(subject_folder_path))
             report_data['global_erp_path'] = str(Path(folders["plots_erp"]).relative_to(subject_folder_path))
             report_data['zscore_path'] = str(Path(folders["plots_zscore"]).relative_to(subject_folder_path))
             report_data['variance_path'] = str(Path(folders["plots_var"]).relative_to(subject_folder_path))
             report_data['tfr_path'] = str(Path(folders["plots_tfr"]).relative_to(subject_folder_path))
             report_data['ica_path'] = str(Path(folders["plots_ica"]).relative_to(subject_folder_path))
             report_data['source_path'] = str(Path(folders["plots_src"]).relative_to(subject_folder_path))
             # 'sites_path' might refer to detailed site reports, requires separate handling if needed

             # --- Add Z-Score data from computation_results --- #
             # The template expects a key 'zscore', likely containing EO/EC results
             # Convert Z-score data (potentially DataFrames/arrays) to JSON-serializable lists
             zscore_data_eo = computation_results.get("zscores_eo")
             zscore_data_ec = computation_results.get("zscores_ec")

             def convert_to_serializable(data):
                 if data is None:
                     return None
                 # Example: Assuming data is {band: [z_scores_array]}
                 serializable_data = {}
                 try:
                     for band, z_array in data.items():
                         if isinstance(z_array, np.ndarray):
                             # Replace NaN/inf with None for JSON compatibility
                             clean_array = np.where(np.isfinite(z_array), z_array, None)
                             serializable_data[band] = clean_array.tolist()
                         elif isinstance(z_array, list):
                              # Clean list items too
                              serializable_data[band] = [x if np.isfinite(x) else None for x in z_array]
                         else:
                             serializable_data[band] = None # Or handle other types if needed
                 except Exception as e:
                     logging.warning(f"Could not convert z-score data to serializable format: {e}")
                     return None # Return None if conversion fails
                 return serializable_data

             report_data['zscore'] = {
                 "EO": convert_to_serializable(zscore_data_eo),
                 "EC": convert_to_serializable(zscore_data_ec)
             }

             # --- Build per-site plots dictionary ---
             site_plots_dir = Path(folders["plots_site"])
             site_dict = {}
             for site in report_data['site_list']:
                 ch_folder = site_plots_dir / site
                 psd_folder = ch_folder / "PSD_Overlay"
                 wave_folder = ch_folder / "Waveform_Overlay"
                 diff_folder = ch_folder / "Difference"
                 site_dict[site] = {}
                 for band in band_list:
                     band_dict = {}
                     psd_file = psd_folder / f"{site}_PSD_{band}.png"
                     wave_file = wave_folder / f"{site}_Waveform_{band}.png"
                     diff_bar_file = diff_folder / f"{site}_Difference_{band}.png"
                     band_dict['psd'] = str(psd_file.relative_to(subject_folder_path)) if psd_file.exists() else ""
                     band_dict['wave'] = str(wave_file.relative_to(subject_folder_path)) if wave_file.exists() else ""
                     band_dict['diff_bar'] = str(diff_bar_file.relative_to(subject_folder_path)) if diff_bar_file.exists() else ""
                     band_dict['diff_topo'] = ""  # no per-site topomap available
                     site_dict[site][band] = band_dict
             report_data['site_dict'] = site_dict

             # --- Call the HTML generation function ---
             print(f"  Attempting to generate HTML report at: {html_report_path}")
             report.build_html_report(report_data, str(html_report_path))
             print(f"  ✅ Generated HTML report: {html_report_path}")

             # --- PDF Report Generation ---
             try:
                 print(f"  Attempting to generate PDF report at: {pdf_report_path}")
                 # Extract required data for PDF report
                 band_powers = {
                     "EO": computation_results.get("band_powers_eo", {}),
                     "EC": computation_results.get("band_powers_ec", {})
                 }
                 
                 # Fix: Correctly extract instability indices from computation results
                 instability_results = computation_results.get("instability_results", {})
                 instability_indices = {
                     "EO": instability_results.get('indices', {}).get('EO', {}),
                     "EC": instability_results.get('indices', {}).get('EC', {})
                 }
                 
                 source_localization = computation_results.get("source_localization", {})
                 
                 vigilance_plots = {}
                 if "hypnograms" in report_data and "strips" in report_data:
                     for condition in ["EO", "EC"]:
                         if condition in report_data.get("hypnograms", {}) and condition in report_data.get("strips", {}):
                             hypno_path = report_data["hypnograms"][condition]
                             strip_path = report_data["strips"][condition]
                             if isinstance(hypno_path, str) and isinstance(strip_path, str):
                                 vigilance_plots[condition] = {
                                     "hypnogram": str(Path(subject_folder) / hypno_path),
                                     "strip": str(Path(subject_folder) / strip_path)
                                 }
                 
                 channels = report_data.get('site_list', [])
                 
                 coherence = {
                     "EO": computation_results.get("coherence_eo", {}),
                     "EC": computation_results.get("coherence_ec", {})
                 }
                 
                 zscores = {
                     "EO": computation_results.get("zscores_eo", {}),
                     "EC": computation_results.get("zscores_ec", {})
                 }
                 
                 connectivity = {
                     "EO": computation_results.get("connectivity_eo", {}),
                     "EC": computation_results.get("connectivity_ec", {})
                 }
                 
                 site_metrics = computation_results.get("site_metrics", {})

                 # Create output directory for PDF report if it doesn't exist
                 pdf_report_dir = pdf_report_path.parent
                 pdf_report_dir.mkdir(parents=True, exist_ok=True)
                 
                 # Generate the PDF report
                 pdf_report_builder.build_pdf_report(
                     report_output_dir=pdf_report_dir,
                     band_powers=band_powers,
                     instability_indices=instability_indices,
                     source_localization=source_localization,
                     vigilance_plots=vigilance_plots,
                     channels=channels,
                     coherence=coherence,
                     zscores=zscores,
                     connectivity=connectivity,
                     site_metrics=site_metrics
                 )
                 print(f"  ✅ Generated PDF report: {pdf_report_path}")
             except Exception as e_pdf:
                 logging.error(f"  ❌ Error generating PDF report for {subject}: {e_pdf}", exc_info=True)

             # Generate clinical report
             try:
                 print(f"  Attempting to generate clinical report...")
                 clinical_report_results = clinical_report._generate_clinical_report(
                     raw_ec=raw_ec,
                     raw_eo=raw_eo,
                     output_dir=str(folders["reports"]),
                     channels=report_data.get('site_list', [])
                 )
                 print(f"  ✅ Generated clinical report")
             except Exception as e_clinical:
                 logging.error(f"  ❌ Error generating clinical report for {subject}: {e_clinical}", exc_info=True)

        except Exception as e:
            logging.error(f"  ❌ Error during report generation for {subject}: {e}", exc_info=True)

    report_end_time = time.time()
    print(f"Report generation finished in {report_end_time - report_start_time:.2f}s")


# --- Extension Script Execution ---
def run_extension_scripts(project_dir, subject):
    """Runs optional extension scripts found in the project directory."""
    extension_script = Path(project_dir) / "extensions" / "EEG_Extension.py"
    if extension_script.exists():
        print(f"\n🔌 Running extension script for {subject}: {extension_script}")
        try:
            # Ensure it runs with the same Python interpreter
            # Pass subject ID and project dir as arguments for potential use
            result = subprocess.run(
                [sys.executable, str(extension_script), f"--subject={subject}", f"--project_dir={project_dir}"],
                capture_output=True, text=True, check=True, encoding='utf-8'
            )
            print(f"  Extension script output:\n{result.stdout}")
            if result.stderr:
                print(f"  Extension script error output:\n{result.stderr}")
            print(f"  ✅ Extension script completed successfully for {subject}.")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Extension script failed for {subject} with exit code {e.returncode}.")
            print(f"  Stderr:\n{e.stderr}")
            print(f"  Stdout:\n{e.stdout}")
        except Exception as e:
            print(f"  ❌ An unexpected error occurred while running extension script for {subject}: {e}")
    else:
        logging.info(f"No extension script found at {extension_script}. Skipping.")


# --- Main Subject Processing Function (Refactored) ---
def process_subject(subject, files, project_dir, config):
    """Main processing pipeline for a single subject."""
    print(f"\n============== Processing Subject: {subject} ==============")
    start_time_subject = time.time()
    num_workers = config.get('num_workers', mp.cpu_count()) # Get num_workers from config

    # --- Setup Output Directories ---
    try:
        folders = io_utils.setup_output_directories(project_dir, subject)
        subject_folder = Path(folders["subject"]) # Ensure subject_folder is a Path object
        band_list = list(processing.BANDS.keys()) # Get band list
    except Exception as e:
        logging.error(f"Failed to set up output directories for {subject}: {e}", exc_info=True)
        return # Cannot proceed without output directories

    # --- Load and Preprocess Data ---
    raw_eo, raw_ec, raw_graph_eo, raw_graph_ec = None, None, None, None # Initialize
    try:
        logging.info(f"Loading data for subject {subject}...")
        # Pass absolute project_dir
        raw_eo, raw_ec, raw_graph_eo, raw_graph_ec, _ = load_and_preprocess_data(
            project_dir,
            files,
            config.get('use_csd', False),
            lpf_freq=config.get('lpf'),      # Pass LPF frequency from config
            notch_freq=config.get('notch') # Pass Notch frequency from config
        )
        data_load_time = time.time()
        logging.info(f"Data loaded in {data_load_time - start_time_subject:.2f}s")

        if raw_eo is None and raw_ec is None:
            logging.warning(f"Skipping processing for subject {subject}: No valid EEG data loaded.")
            return

    except Exception as e:
        logging.error(f"Error loading/preprocessing data for subject {subject}: {e}", exc_info=True)
        return

    # --- Compute Z-Score Normalization Stats ---
    norm_method = config.get('zscore_method', 'standard')
    norm_stats_for_calc = None
    if norm_method.startswith('published_'):
        try:
             norm_stats_for_calc = stats_utils.load_zscore_stats(norm_method)
             if norm_stats_for_calc is None:
                  logging.warning(f"Failed to load published norms '{norm_method}'. Z-scores will be relative.")
        except Exception as e:
             logging.error(f"Error loading z-score stats for method {norm_method}: {e}")


    # --- Define Computation Tasks for Parallel Processing ---
    logging.info("⚙️ Preparing parallel computation tasks...")
    tasks = []
    result_keys = []

    # 1. Band Powers (Absolute)
    if raw_eo: tasks.append((processing.compute_all_band_powers, (raw_eo,), "band_powers_eo"))
    if raw_ec: tasks.append((processing.compute_all_band_powers, (raw_ec,), "band_powers_ec"))

    # 2. Coherence
    for band in band_list:
        band_range = processing.BANDS[band]
        if raw_eo:
            data_eo = raw_eo.get_data().astype(np.float64)
            tasks.append((processing.compute_coherence_matrix, (data_eo, raw_eo.info['sfreq'], band_range, int(raw_eo.info['sfreq'] * 2)), f"coherence_eo_{band}"))
        if raw_ec:
            data_ec = raw_ec.get_data().astype(np.float64)
            tasks.append((processing.compute_coherence_matrix, (data_ec, raw_ec.info['sfreq'], band_range, int(raw_ec.info['sfreq'] * 2)), f"coherence_ec_{band}"))

    # 3. Z-Scores (Uses norm_stats_for_calc)
    if raw_eo: tasks.append((processing.compute_all_zscore_maps, (raw_eo, norm_stats_for_calc), "zscores_eo"))
    if raw_ec: tasks.append((processing.compute_all_zscore_maps, (raw_ec, norm_stats_for_calc), "zscores_ec"))

    # 4. TFR
    n_cycles_tfr = config.get('tfr_n_cycles', 2.0)
    tmin_tfr, tmax_tfr = config.get('tfr_tmin', 0.0), config.get('tfr_tmax', 4.0)
    if raw_eo: tasks.append((processing.compute_all_tfr_maps, (raw_eo, n_cycles_tfr, tmin_tfr, tmax_tfr), "tfr_eo"))
    if raw_ec: tasks.append((processing.compute_all_tfr_maps, (raw_ec, n_cycles_tfr, tmin_tfr, tmax_tfr), "tfr_ec"))

    # Add instability index computation
    if raw_eo and raw_ec:
        tasks.append((clinical_report.compute_instability_index, (raw_ec, raw_eo), "instability_results"))

    # 5. ICA (Typically run on EO data)
    if raw_eo:
        n_components_ica = config.get('ica_n_components', 0.95)
        method_ica = config.get('ica_method', 'fastica')
        tasks.append((processing.compute_ica, (raw_eo, n_components_ica, method_ica), "ica_eo"))
        
        # --- Add Task for EC ICA --- #
        if raw_ec:
            tasks.append((processing.compute_ica, (raw_ec, n_components_ica, method_ica), "ica_ec"))
            logging.info("  Added ICA computation task for EC.")
        # --- End Add --- #

    # --- Run Computations in Parallel ---
    computation_results = {}
    if tasks:
        logging.info(f"🚀 Starting {len(tasks)} parallel computation tasks using {num_workers} workers...")
        computation_start_time = time.time()
        try:
             # Using Pool context manager for cleaner exit
             with mp.Pool(processes=num_workers) as pool:
                  async_results = []
                  temp_result_keys = [] # Use temp list to match async_results order

                  # Disable Numba JIT in worker processes if needed
                  # os.environ['NUMBA_DISABLE_JIT'] = '1' # Potentially set via initializer in Pool

                  for func, args, key in tasks:
                       temp_result_keys.append(key)
                       res = pool.apply_async(execute_task, args=(func, args))
                       async_results.append(res)

                  pool.close() # No more tasks

                  # Collect results with timeout
                  timeout_seconds = 300 # 5 minutes per task
                  for i, res in enumerate(async_results):
                       key = temp_result_keys[i]
                       try:
                            result = res.get(timeout=timeout_seconds)
                            computation_results[key] = result # Store even if None (indicates failure in execute_task)
                            if result is not None:
                                 logging.info(f"  ✅ Task '{key}' completed.")
                            else:
                                 logging.warning(f"  ⚠️ Task '{key}' failed (returned None). Check logs for errors.")
                       except mp.TimeoutError:
                            logging.error(f"  ❌ Task '{key}' timed out after {timeout_seconds}s.")
                            computation_results[key] = None # Mark as failed
                       except Exception as e:
                            logging.error(f"  ❌ Task '{key}' failed with error: {e}", exc_info=True)
                            computation_results[key] = None # Mark as failed

                  pool.join() # Wait for workers to finish
                  # os.environ['NUMBA_DISABLE_JIT'] = '0' # Reset if set earlier

        except Exception as pool_e:
             logging.error(f"Multiprocessing pool error: {pool_e}", exc_info=True)
             # Mark all potentially uncollected results as failed
             for key in temp_result_keys:
                  if key not in computation_results: computation_results[key] = None

        computation_end_time = time.time()
        logging.info(f"Computations finished in {computation_end_time - computation_start_time:.2f}s")
    else:
        logging.info("No computation tasks to run in parallel.")


    # --- Sequential Plotting and Saving ---
    logging.info("📊 Generating and saving plots...")
    plotting_start_time = time.time()
    plot_filenames = { # Initialize dict to store relative plot paths
        "topomaps": {"EO": {}, "EC": {}}, "waveforms": {"EO": {}, "EC": {}},
        "coherence": {"EO": {}, "EC": {}}, "erp": {"EO": "", "EC": ""},
        "zscores": {"EO": {}, "EC": {}}, "variance": {"EO": {}, "EC": {}},
        "tfr": {"EO": {}, "EC": {}}, 
        # Change ICA structure to hold components and properties paths
        "ica": {"EO": {"components": "", "properties": ""}, "EC": {"components": "", "properties": ""}}, 
        "source_localization": {"EO": {}, "EC": {}} # Keep structure
    }

    # Determine which raw objects and info to use for plotting
    plot_raw_eo = raw_graph_eo if config.get('use_csd', False) and raw_graph_eo else raw_eo
    plot_raw_ec = raw_graph_ec if config.get('use_csd', False) and raw_graph_ec else raw_ec
    info = None
    ch_names = []
    if plot_raw_eo:
        info = plot_raw_eo.info
        ch_names = plot_raw_eo.ch_names
    elif plot_raw_ec:
        info = plot_raw_ec.info
        ch_names = plot_raw_ec.ch_names

    if info is None:
        logging.error("❌ Cannot generate plots: No valid MNE info object available.")
    else:
        # 1. Topomaps (Absolute & Relative)
        logging.info("  Generating Topomaps...")
        for cond, raw_obj, bp_key in [("EO", plot_raw_eo, "band_powers_eo"), ("EC", plot_raw_ec, "band_powers_ec")]:
            if raw_obj and computation_results.get(bp_key):
                band_powers = computation_results[bp_key] # {ch: {band: power}}
                if not band_powers: continue # Skip if empty
                total_power_per_ch = {ch: sum(powers.values()) for ch, powers in band_powers.items()}

                for band in band_list:
                    try:
                        abs_power = [band_powers.get(ch, {}).get(band, np.nan) for ch in ch_names]
                        rel_power = [(band_powers.get(ch, {}).get(band, 0) / total_power_per_ch.get(ch, 1)) if total_power_per_ch.get(ch, 0) > 0 else 0 for ch in ch_names]

                        # Check if power arrays contain valid numbers before plotting
                        if not np.any(np.isfinite(abs_power)) and not np.any(np.isfinite(rel_power)):
                             logging.warning(f"    Skipping topomap for {cond}-{band}: All power values are NaN/inf.")
                             continue

                        fig_topo = plotting.plot_topomap_abs_rel(abs_power, rel_power, info, band, cond)
                        topo_path = Path(folders["plots_topo"]) / f"topomap_{cond}_{band}.png"
                        fig_topo.savefig(str(topo_path), facecolor='black', bbox_inches='tight')
                        plt.close(fig_topo)
                        plot_filenames["topomaps"][cond][band] = str(topo_path.relative_to(subject_folder))
                    except Exception as e:
                        logging.error(f"    ❌ Error plotting topomap for {cond}-{band}: {e}", exc_info=True)

        # 2. Waveforms (Plot one band per condition for example)
        logging.info("  Generating Waveforms...")
        for cond, raw_obj in [("EO", plot_raw_eo), ("EC", plot_raw_ec)]:
            if raw_obj:
                # --- FIX: Loop through all bands --- 
                for band_to_plot in band_list: # Use band_list from process_subject scope
                    # Plot for Alpha band as an example, adjust if needed
                    # band_to_plot = 'Alpha' # <-- REMOVED HARDCODING
                    if band_to_plot in processing.BANDS:
                         try:
                              band_range = processing.BANDS[band_to_plot]
                              # Ensure data is C-contiguous and float64
                              wf_data = np.ascontiguousarray(raw_obj.get_data() * 1e6, dtype=np.float64) # Convert to µV
                              wf_fig = plotting.plot_waveform_grid(wf_data, ch_names, info['sfreq'], band=band_range, epoch_length=10)
                              wf_path = Path(folders["plots_wave"]) / f"waveforms_{cond}_{band_to_plot}.png"
                              wf_fig.savefig(str(wf_path), facecolor='black', bbox_inches='tight')
                              plt.close(wf_fig)
                              # Store under the specific band plotted
                              # Ensure the nested dictionary exists
                              if cond not in plot_filenames["waveforms"]: plot_filenames["waveforms"][cond] = {}
                              plot_filenames["waveforms"][cond][band_to_plot] = str(wf_path.relative_to(subject_folder))
                         except Exception as e:
                              logging.error(f"    ❌ Error plotting waveform grid for {cond}-{band_to_plot}: {e}", exc_info=True)
                              if 'wf_fig' in locals() and isinstance(wf_fig, plt.Figure): plt.close(wf_fig) # Close figure on error
                    else:
                        logging.warning(f"    Skipping waveform plot for {cond}-{band_to_plot}: Band not defined in processing.BANDS")
                # --- END FIX --- 

        # 3. Coherence
        logging.info("  Generating Coherence Plots...")
        for cond in ["EO", "EC"]:
            for band in band_list:
                coh_key = f"coherence_{cond.lower()}_{band}"
                if computation_results.get(coh_key) is not None:
                    try:
                        coh_matrix = computation_results[coh_key]
                        if not np.any(np.isfinite(coh_matrix)):
                            logging.warning(f"    Skipping coherence plot for {cond}-{band}: Matrix contains NaNs/infs.")
                            continue
                        fig_coh = plotting.plot_coherence_matrix(coh_matrix, ch_names)
                        coh_path = Path(folders["plots_coh"]) / f"coherence_{cond}_{band}.png"
                        fig_coh.savefig(str(coh_path), facecolor='black', bbox_inches='tight')
                        plt.close(fig_coh)
                        plot_filenames["coherence"][cond][band] = str(coh_path.relative_to(subject_folder))
                    except Exception as e:
                        logging.error(f"    ❌ Error plotting coherence for {cond}-{band}: {e}", exc_info=True)

        # 4. ERP (Run sequentially as compute_pseudo_erp returns a figure)
        logging.info("  Generating ERP Plots...")
        for cond, raw_obj in [("EO", plot_raw_eo), ("EC", plot_raw_ec)]:
             if raw_obj:
                 try:
                     erp_fig = processing.compute_pseudo_erp(raw_obj) # Returns figure
                     if erp_fig: # Check if figure was created
                          erp_path = Path(folders["plots_erp"]) / f"erp_{cond}.png"
                          erp_fig.savefig(str(erp_path), facecolor='black', bbox_inches='tight')
                          plt.close(erp_fig)
                          plot_filenames["erp"][cond] = str(erp_path.relative_to(subject_folder))
                     else:
                          logging.warning(f"    Skipping ERP plot for {cond}: Figure generation failed.")
                 except Exception as e:
                      logging.error(f"    ❌ Error plotting ERP for {cond}: {e}", exc_info=True)

        # 5. Z-Scores
        logging.info("  Generating Z-Score Plots...")
        for cond, raw_obj, zscores_key in [("EO", plot_raw_eo, "zscores_eo"), ("EC", plot_raw_ec, "zscores_ec")]:
             if raw_obj and computation_results.get(zscores_key):
                 zscore_maps = computation_results[zscores_key] # {band: [z_scores]}
                 if not zscore_maps: continue
                 for band in band_list:
                     if band in zscore_maps and zscore_maps[band] is not None:
                         try:
                             z_scores = zscore_maps[band]
                             if not np.any(np.isfinite(z_scores)):
                                  logging.warning(f"    Skipping z-score plot for {cond}-{band}: All values are NaN/inf.")
                                  continue
                             fig_zscore = plotting.plot_zscore_topomap(z_scores, info, band, cond)
                             zscore_path = Path(folders["plots_zscore"]) / f"zscore_{cond}_{band}.png"
                             fig_zscore.savefig(str(zscore_path), facecolor='black', bbox_inches='tight')
                             plt.close(fig_zscore)
                             plot_filenames["zscores"][cond][band] = str(zscore_path.relative_to(subject_folder))
                         except Exception as e:
                             logging.error(f"    ❌ Error plotting zscore for {cond}-{band}: {e}", exc_info=True)

        # 6. Variance Topomaps
        logging.info("  Generating Variance Plots...")
        for cond, raw_obj, bp_key in [("EO", plot_raw_eo, "band_powers_eo"), ("EC", plot_raw_ec, "band_powers_ec")]:
             if raw_obj and computation_results.get(bp_key):
                 band_powers = computation_results[bp_key] # {ch: {band: power}}
                 if not band_powers: continue
                 for band in band_list:
                     try:
                         power_values = [band_powers.get(ch, {}).get(band, np.nan) for ch in ch_names]
                         valid_powers = [p for p in power_values if not np.isnan(p)]
                         if not valid_powers: continue
                         variance_val = np.var(valid_powers)

                         # Use absolute power for map, display variance in title
                         fig_variance = plotting.plot_topomap_abs_rel(power_values, [variance_val]*len(power_values), info, band, f"{cond} (Var={variance_val:.2f})")
                         variance_path = Path(folders["plots_var"]) / f"variance_{cond}_{band}.png"
                         fig_variance.savefig(str(variance_path), facecolor='black', bbox_inches='tight')
                         plt.close(fig_variance)
                         plot_filenames["variance"][cond][band] = str(variance_path.relative_to(subject_folder))
                     except Exception as e:
                         logging.error(f"    ❌ Error plotting variance for {cond}-{band}: {e}", exc_info=True)

        # 7. TFR
        logging.info("  Generating TFR Plots...")
        for cond, tfr_key in [("EO", "tfr_eo"), ("EC", "tfr_ec")]:
             if computation_results.get(tfr_key):
                 tfr_maps = computation_results[tfr_key] # {band: tfr_object}
                 if not tfr_maps: continue
                 for band in band_list:
                      if band in tfr_maps and tfr_maps[band] is not None:
                          try:
                              tfr_object = tfr_maps[band]
                              # Plot TFR for a representative channel (e.g., central)
                              central_chs = ['Cz', 'Pz', 'Fz']
                              pick_ch = None
                              for ch in central_chs:
                                   if ch in ch_names:
                                        pick_ch = ch
                                        break
                              pick_idx = ch_names.index(pick_ch) if pick_ch else 0 # Default to first channel

                              fig_tfr = plotting.plot_tfr(tfr_object, picks=pick_idx) # Pass channel index
                              tfr_path = Path(folders["plots_tfr"]) / f"tfr_{cond}_{band}.png"
                              fig_tfr.savefig(str(tfr_path), facecolor='black', bbox_inches='tight')
                              plt.close(fig_tfr)
                              plot_filenames["tfr"][cond][band] = str(tfr_path.relative_to(subject_folder))
                          except Exception as e:
                               logging.error(f"    ❌ Error plotting TFR for {cond}-{band}: {e}", exc_info=True)

        # 8. ICA
        logging.info("  Generating ICA Plots...")
        # --- Loop through conditions --- #
        for cond, plot_raw, ica_key in [("EO", plot_raw_eo, "ica_eo"), ("EC", plot_raw_ec, "ica_ec")]:
            logging.info(f"  Processing ICA plots for {cond}...")
            # Check if data and computed ICA exist for this condition
            if plot_raw and computation_results.get(ica_key):
                try:
                    ica_obj = computation_results[ica_key]
                    if ica_obj:
                         # --- Plot Component Topographies ---
                         ica_path_components = Path(folders["plots_ica"]) / f"ica_components_{cond}.png"
                         try:
                             # Pass inst=info to ensure channel locations are used
                             figs_components = ica_obj.plot_components(inst=plot_raw.info, show=False)

                             if isinstance(figs_components, list) and figs_components:
                                 if isinstance(figs_components[0], plt.Figure):
                                     figs_components[0].savefig(str(ica_path_components), facecolor='black', bbox_inches='tight')
                                     logging.info(f"    Saved ICA components plot ({cond}) to {ica_path_components}")
                                     plot_filenames["ica"][cond]["components"] = str(ica_path_components.relative_to(subject_folder))
                                     for fig in figs_components: plt.close(fig)
                                 else:
                                     logging.warning(f"    ica.plot_components ({cond}) did not return a list of Figures.")
                             elif isinstance(figs_components, plt.Figure):
                                 figs_components.savefig(str(ica_path_components), facecolor='black', bbox_inches='tight')
                                 plt.close(figs_components)
                                 logging.info(f"    Saved ICA components plot ({cond}) to {ica_path_components}")
                                 plot_filenames["ica"][cond]["components"] = str(ica_path_components.relative_to(subject_folder))
                             else:
                                 logging.warning(f"    ica.plot_components ({cond}) returned unexpected type: {type(figs_components)}")

                         except Exception as e_ica_comp:
                             logging.error(f"    ❌ Error plotting ICA components ({cond}): {e_ica_comp}", exc_info=True)
                             # Close figures if error occurred
                             if 'figs_components' in locals():
                                 if isinstance(figs_components, list):
                                     for fig in figs_components:
                                         if isinstance(fig, plt.Figure): plt.close(fig)
                                 elif isinstance(figs_components, plt.Figure):
                                    plt.close(figs_components)

                         # --- Plot properties --- 
                         ica_path_props = Path(folders["plots_ica"]) / f"ica_properties_{cond}.png"
                         try:
                              picks_to_plot = range(min(5, ica_obj.n_components_))
                              fig_ica_props_list = ica_obj.plot_properties(plot_raw, picks=picks_to_plot, show=False)
                              
                              if fig_ica_props_list: 
                                   fig_ica_props_list[0].savefig(str(ica_path_props), facecolor='black', bbox_inches='tight')
                                   plot_filenames["ica"][cond]["properties"] = str(ica_path_props.relative_to(subject_folder))
                                   logging.info(f"    Saved first ICA properties plot ({cond}) to {ica_path_props}")
                                   for fig in fig_ica_props_list: 
                                       plt.close(fig)
                              else:
                                   logging.warning(f"    ica.plot_properties ({cond}) returned an empty list.")
                                   
                         except MemoryError as e_mem:
                             logging.error(f"    ❌ MemoryError plotting ICA properties ({cond}). Consider reducing components or plotting fewer picks: {e_mem}", exc_info=False) 
                         except Exception as e_ica_prop:
                              logging.error(f"    ❌ Error plotting ICA properties ({cond}): {e_ica_prop}", exc_info=True)
                              if 'fig_ica_props_list' in locals() and isinstance(fig_ica_props_list, list):
                                  for fig in fig_ica_props_list: plt.close(fig)
                                   
                    else:
                         logging.warning(f"    Skipping ICA plotting ({cond}): ICA object is None.")
                except Exception as e:
                     logging.error(f"    ❌ Error during ICA plotting setup for {cond}: {e}", exc_info=True)
            else:
                logging.info(f"  Skipping ICA plots for {cond}: Missing raw data or computed ICA object.")
        # --- End condition loop --- #

    plotting_end_time = time.time()
    logging.info(f"Plotting finished in {plotting_end_time - plotting_start_time:.2f}s")

    # --- Generate Per-Site Reports/Plots --- #
    try:
        logging.info("📊 Generating Per-Site Reports...")
        # Call the function from clinical.py to generate detailed site plots
        if raw_eo and raw_ec:
            clinical.generate_full_site_reports(
                raw_eo=raw_eo, # Use original raw data
                raw_ec=raw_ec,
                output_dir=folders["plots_site"] # Save to the dedicated site reports folder
            )
            logging.info(f"  ✅ Per-Site reports generated in {folders['plots_site']}")
        else:
            logging.warning("  Skipping Per-Site reports: Both EO and EC raw data are required.")
    except Exception as e_site_rep:
        logging.error(f"  ❌ Error generating Per-Site Reports: {e_site_rep}", exc_info=True)


    # --- Compare Z-Scores with Clinical Outcomes (Optional) --- #
    clinical_csv_path = config.get('clinical_csv')
    if clinical_csv_path:
        logging.info(f"📉 Attempting to load clinical outcomes from: {clinical_csv_path}")
        if info: # Need info for n_channels
            try:
                n_channels = info['nchan']
                clinical_outcomes = stats_utils.load_clinical_outcomes(clinical_csv_path, n_channels)
                
                # Check if loading was successful (load_clinical_outcomes returns fallback data on error)
                # A more robust check might involve checking if the loaded data is not the random fallback
                # For now, we assume if the file path was valid, loading likely worked or logged errors.
                if clinical_outcomes is not None: 
                    logging.info("  Clinical outcomes loaded. Comparing with Z-scores...")
                    # Get the computed z-scores (e.g., for EO condition)
                    zscores_to_compare = computation_results.get("zscores_eo")
                    if zscores_to_compare:
                        stats_utils.compare_zscores(
                            z_scores_dict=zscores_to_compare,
                            clinical_outcomes=clinical_outcomes,
                            z_score_type_name=config.get('zscore_method', 'Unknown') + " (EO)" # Pass method name
                        )
                    else:
                        logging.warning("  Skipping Z-score comparison: Computed Z-scores (EO) not found.")
                        
                    # Optionally, compare EC z-scores too
                    # zscores_ec_to_compare = computation_results.get("zscores_ec")
                    # if zscores_ec_to_compare:
                    #     stats_utils.compare_zscores(
                    #         z_scores_dict=zscores_ec_to_compare,
                    #         clinical_outcomes=clinical_outcomes,
                    #         z_score_type_name=config.get('zscore_method', 'Unknown') + " (EC)"
                    #     )
            except Exception as e_comp_z:
                 logging.error(f"  ❌ Error during Z-score comparison with clinical outcomes: {e_comp_z}", exc_info=True)
        else:
             logging.warning("  Skipping Z-score comparison: MNE info object not available to determine channel count.")
    else:
        logging.info("ℹ️ Skipping Z-score comparison with clinical outcomes: No --clinical_csv provided.")


    # --- Source Localization ---
    if not config.get('skip_source', False):
        try:
             # Call the new function from the processing module
             source_localization_results = processing.run_source_localization_analysis(
                 raw_eo=raw_eo, 
                 raw_ec=raw_ec, 
                 folders=folders, # Pass the folders dict
                 band_list=band_list,
                 num_workers=config.get('num_workers') # Pass worker count
             )
             # Store the returned filenames dict in the main plot_filenames dict
             plot_filenames['source_localization'] = source_localization_results 
        except Exception as e_src:
             logging.error(f"Error running source localization analysis: {e_src}", exc_info=True)
             # Ensure the entry exists even if it failed, to avoid errors in report generation
             plot_filenames['source_localization'] = {"EO": {}, "EC": {}} 
    else:
         logging.info("Skipping Source Localization as requested.")
         plot_filenames['source_localization'] = {"EO": {}, "EC": {}} # Ensure structure exists


    # --- Phenotype Processing ---
    phenotype_result_data = None
    if not config.get('skip_phenotype', False):
        try:
            # Determine optional inputs
            # We need csd_raw_eo if CSD features are used
            csd_for_pheno = raw_graph_eo if config.get('use_csd') else None
            # We need sloreta_data if source features are used
            # This might come from computation_results or source_localization_results
            # Placeholder: Assume sloreta_data is not readily available here yet
            sloreta_for_pheno = None 
            # We need vigilance_states if used by feature extraction
            # Placeholder: Assume vigilance_states not readily available here yet
            vigilance_for_pheno = None
            
            # Call the new function
            phenotype_result_data = phenotype.run_phenotype_analysis(
                raw_eo=raw_eo, 
                subject_folder=subject_folder, # Pass Path object
                subject=subject,
                raw_ec=raw_ec, # Pass optional EC data
                csd_raw_eo=csd_for_pheno, # Pass optional CSD EO data
                sloreta_data=sloreta_for_pheno, # Pass optional source data
                vigilance_states=vigilance_for_pheno # Pass optional vigilance
            )
        except Exception as e_pheno:
             logging.error(f"Error running phenotype analysis: {e_pheno}", exc_info=True)
    else:
        logging.info("Skipping Phenotype processing as requested.")


    # --- Vigilance Processing ---
    if not config.get('skip_vigilance', False):
        # Determine channels dynamically based on availability and desired locations
        # Use info from whichever raw object is available
        vig_info = raw_eo.info if raw_eo else (raw_ec.info if raw_ec else None)
        available_channels_upper = [ch.upper() for ch in vig_info.ch_names] if vig_info else []
        channels_to_process_vigilance = [] # Reset list

        # Use all available channels from the Raw object
        if vig_info:
            channels_to_process_vigilance = vig_info.ch_names

        if not channels_to_process_vigilance:
            logging.warning("  Skipping vigilance: No channels found in the provided data.")
        else:
            logging.info(f"  Processing vigilance for all available channels: {len(channels_to_process_vigilance)} channels")
            # Use appropriate raw object (consistent with plotting)
            vig_raw_eo = plot_raw_eo
            vig_raw_ec = plot_raw_ec
            output_dir_vigilance = Path(folders["plots_vigilance"]) # Get Path object for output dir

            # Call the new function for EO
            if vig_raw_eo:
                try:
                    # --- DEBUG LOG --- 
                    logging.info(f"Passing object of type {type(vig_raw_eo)} to run_vigilance_analysis for EO")
                    # --- END DEBUG LOG ---
                    vigilance.run_vigilance_analysis(
                        raw=vig_raw_eo,
                        output_plot_dir=output_dir_vigilance,
                        condition="EO",
                        channels_to_process=channels_to_process_vigilance
                    )
                except Exception as e_vig_eo:
                    logging.error(f"Error during EO vigilance analysis run: {e_vig_eo}", exc_info=True)

            # Call the new function for EC
            if vig_raw_ec:
                try:
                    # --- DEBUG LOG --- 
                    logging.info(f"Passing object of type {type(vig_raw_ec)} to run_vigilance_analysis for EC")
                    # --- END DEBUG LOG ---
                    vigilance.run_vigilance_analysis(
                        raw=vig_raw_ec,
                        output_plot_dir=output_dir_vigilance,
                        condition="EC",
                        channels_to_process=channels_to_process_vigilance
                    )
                except Exception as e_vig_ec:
                    logging.error(f"Error during EC vigilance analysis run: {e_vig_ec}", exc_info=True)
    else:
        logging.info("Skipping Vigilance processing as requested.")


    # --- Generate Clinical Interpretation Report --- #
    try:
        logging.info("📄 Generating Clinical Interpretation Report...")
        # Call clinical report generation with correct parameters
        clinical_report._generate_clinical_report(
            raw_eo=raw_eo,
            raw_ec=raw_ec,
            output_dir=subject_folder,
            channels=ch_names
        )
        logging.info("  ✅ Clinical Interpretation Report generated.")
    except AttributeError:
        logging.warning("Skipping Clinical Interpretation Report: Function _generate_clinical_report not found in clinical_report module.")
    except Exception as e_clin_rep:
        logging.error(f"  ❌ Error generating Clinical Interpretation Report: {e_clin_rep}", exc_info=True)


    # --- Generate Reports ---
    if not config.get('skip_report', False):
        generate_reports(raw_eo, raw_ec, folders, subject_folder, subject, band_list, config,
                         computation_results, plot_filenames)
    else:
        logging.info("Skipping Report generation as requested.")

    # --- Run Extension Scripts ---
    run_extension_scripts(project_dir, subject) # Pass absolute project_dir

    subject_end_time = time.time()
    logging.info(f"============== Finished processing Subject: {subject} in {subject_end_time - start_time_subject:.2f}s ==============")


# --- Main Execution Block ---
def main():
    args = parse_arguments() # Get parsed args
    project_dir = args.project_dir # Use project_dir from args
    input_dir = args.input_dir # Use input_dir from args
    config = vars(args) # Use dict derived from args

    # --- Setup Logging ---
    log_dir = Path(project_dir) / 'output'
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'processing.log'
        # Setup root logger
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
                            handlers=[logging.FileHandler(log_file, encoding='utf-8'),
                                      logging.StreamHandler(sys.stdout)])
    except Exception as e:
        print(f"Error setting up logging: {e}. Continuing without file logging.")
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])


    logging.info("=======================================================")
    logging.info(" EEG Processing Pipeline Initialized ")
    logging.info(f"Project Directory: {project_dir}")
    logging.info(f"Input Directory: {input_dir}")
    # Log relevant config options
    logging.info(f"Config: {config}")
    logging.info("=======================================================")

    # --- INTERACTIVE CHOICE FOR Z-SCORE METHOD ---
    if config['zscore_method'] is None:
        print("\n--- Z-Score Method Selection ---")
        zscore_options = {
            '1': 'standard',
            '2': 'robust_mad',
            '3': 'robust_iqr',
            '4': 'published_kolk',
            '5': 'published_smith'
        }
        print("Please choose a Z-score normalization method:")
        for key, value in zscore_options.items():
            print(f"  {key}: {value}")
        
        while True:
            choice = input("Enter the number of your choice: ").strip()
            if choice in zscore_options:
                config['zscore_method'] = zscore_options[choice]
                print(f"Using Z-score method: {config['zscore_method']}")
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
    else:
        # Log the method provided via command line
        logging.info(f"Using Z-score method from command line: {config['zscore_method']}")

    # --- INTERACTIVE CHOICE FOR CSD --- 
    if config['use_csd'] is None: # Check if CSD option was provided
        print("\n--- CSD Selection ---")
        while True:
            choice = input("Apply Current Source Density (CSD)? (y/n): ").strip().lower()
            if choice == 'y':
                config['use_csd'] = True
                print("CSD will be applied.")
                break
            elif choice == 'n':
                config['use_csd'] = False
                print("CSD will NOT be applied.")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        logging.info(f"Using CSD setting from command line: {config['use_csd']}")

    # --- INTERACTIVE CHOICE FOR LOW-PASS FILTER ---
    if config.get('lpf') is None:
        print("\n--- Low-Pass Filter Selection ---")
        while True:
            choice = input("Apply low-pass filter? (y/n): ").strip().lower()
            if choice == 'y':
                freq_str = input("Enter low-pass cutoff frequency in Hz (e.g., 40.0): ").strip()
                try:
                    config['lpf'] = float(freq_str)
                    print(f"Low-pass filter set to {config['lpf']} Hz")
                    break
                except ValueError:
                    print("Invalid frequency. Please enter a numeric value.")
            elif choice == 'n':
                config['lpf'] = None
                print("No low-pass filtering will be applied.")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        logging.info(f"Using low-pass filter from command line: {config['lpf']} Hz")

    # --- INTERACTIVE CHOICE FOR NOTCH FILTER ---
    if config.get('notch') is None:
        print("\n--- Notch Filter Selection ---")
        while True:
            choice = input("Apply notch filter? (y/n): ").strip().lower()
            if choice == 'y':
                freq_str = input("Enter notch filter frequency in Hz (e.g., 50.0): ").strip()
                try:
                    config['notch'] = float(freq_str)
                    print(f"Notch filter set to {config['notch']} Hz")
                    break
                except ValueError:
                    print("Invalid frequency. Please enter a numeric value.")
            elif choice == 'n':
                config['notch'] = None
                print("No notch filtering will be applied.")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    else:
        logging.info(f"Using notch filter from command line: {config['notch']} Hz")

    # --- Validate chosen Z-score method against known options (optional but good practice) ---
    known_methods = ['standard', 'robust_mad', 'robust_iqr', 'published_kolk', 'published_smith']
    if config['zscore_method'] not in known_methods:
        logging.error(f"Invalid zscore_method specified: '{config['zscore_method']}'. Must be one of {known_methods}. Exiting.")
        sys.exit(1)

    # --- Log Final Config --- # Placed after potential interactive choice
    logging.info("=======================================================")
    logging.info(" EEG Processing Pipeline Initialized ") # Repeat key info after potential interaction
    logging.info(f"Project Directory: {project_dir}")
    logging.info(f"Input Directory: {input_dir}")
    logging.info(f"Using configuration: {config}") # Log the final config
    logging.info("=======================================================")

    # --- CSV Export Mode ---
    if config.get('export_csv'):
        logging.info("Running in CSV Export Mode...")
        try:
            data_to_csv.process_edf_to_csv(config['edf_path'], config['epoch_length'], config['output_csv'])
            logging.info(f"Successfully exported {config['edf_path']} to {config['output_csv']}")
        except Exception as e:
            logging.error(f"CSV Export failed: {e}", exc_info=True)
        sys.exit(0)

    # --- Live Display Thread ---
    live_display_thread = None
    if config.get('live_display'):
        if 'stop_event' not in globals(): # Ensure stop_event is defined
             global stop_event
             stop_event = threading.Event()
        logging.info("🚀 Starting live EEG display simulation...")
        live_display_thread = threading.Thread(target=display_utils.live_eeg_display, args=(stop_event,), daemon=True)
        live_display_thread.start()
        time.sleep(1) # Allow display to initialize

    # Set up signal handler (should be done in main thread)
    signal.signal(signal.SIGINT, sigint_handler)

    # --- Find Subject Files ---
    logging.info(f"🔍 Searching for EDF files in: {input_dir}")
    try:
        subjects = io_utils.find_subject_edf_files(input_dir) # Pass input_dir from args
        if not subjects:
            logging.error(f"No subject EDF files found in {input_dir}. Exiting.")
            if live_display_thread: stop_event.set()
            sys.exit(1)
        logging.info(f"Found {len(subjects)} subjects: {list(subjects.keys())}")
    except Exception as e:
        logging.error(f"Error finding subject files in {input_dir}: {e}", exc_info=True)
        if live_display_thread: stop_event.set()
        sys.exit(1)


    # --- Process Subjects ---
    start_all_subjects = time.time()
    processed_count = 0
    for subject, files in subjects.items():
        try:
            # Pass the config dict to process_subject
            process_subject(subject, files, project_dir, config)
            processed_count += 1
        except Exception as e:
             logging.error(f"Unhandled exception during processing of subject {subject}: {e}", exc_info=True)
             # Continue to next subject or exit? Continuing for now.
    end_all_subjects = time.time()
    logging.info(f"--- Processed {processed_count}/{len(subjects)} subjects in {end_all_subjects - start_all_subjects:.2f} seconds ---")


    # --- Wait for Live Display to Finish ---
    if live_display_thread and live_display_thread.is_alive():
        logging.info("⏳ Live display running. Press Ctrl+C to stop processing and exit.")
        try:
            # Keep main thread alive while display runs
            while live_display_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
             # SIGINT handler should have been called, setting stop_event
             logging.info("Ctrl+C detected in main thread. Waiting for display thread to exit...")
             if not stop_event.is_set(): stop_event.set() # Ensure it's set
             live_display_thread.join(timeout=5.0) # Wait with timeout
             if live_display_thread.is_alive():
                  logging.warning("Live display thread did not exit cleanly.")

    logging.info("🏁 EEG Processing Pipeline Finished Gracefully 🏁")
    print("\n🏁 Processing complete. Check the 'output' directory for results and logs.")


if __name__ == "__main__":
    # Ensure multiprocessing start method is set correctly, especially for Windows/macOS
    if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
         # 'spawn' is generally safer across platforms than 'fork'
         try:
              mp.set_start_method('spawn', force=True)
         except RuntimeError as e:
              print(f"Note: Could not force multiprocessing start method to 'spawn': {e}")

    # Define stop_event globally for signal handler access
    stop_event = threading.Event()
    # Import argparse here if not already imported at top level
    import argparse
    main()
    
