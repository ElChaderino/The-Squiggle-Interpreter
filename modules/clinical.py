import os
import logging
import mne
import matplotlib.pyplot as plt
import numpy as np
import psutil
import shutil
from modules import processing, plotting

logger = logging.getLogger(__name__)

def compute_site_metrics(bp_EO, bp_EC):
    """
    Compute site-specific metrics based on band power for EO and EC.
    
    For each channel, compute:
      - Percentage change in Alpha power (EO → EC)
      - Theta/Beta ratio (using EO data)
    
    Parameters:
      bp_EO (dict): Dictionary of band powers for EO, structured as {channel: {band: power, ...}}
      bp_EC (dict): Same structure as bp_EO, for EC.
      
    Returns:
      dict: {channel: {metric_name: value, ...}, ...}
    """
    metrics = {}
    if not bp_EO or not bp_EC:
        logger.error("Cannot compute site metrics: Missing EO or EC band powers.")
        return metrics

    # Find common channels
    eo_channels = set(bp_EO.keys())
    ec_channels = set(bp_EC.keys())
    common_channels = eo_channels.intersection(ec_channels)
    missing_channels = eo_channels.symmetric_difference(ec_channels)
    if missing_channels:
        logger.warning(f"Channels missing in one condition: {missing_channels}")
        with open("missing_channels_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Missing channels in compute_site_metrics: {missing_channels}\n")
    if not common_channels:
        logger.error("No common channels between EO and EC for site metrics computation.")
        return metrics

    for ch in common_channels:
        metrics[ch] = {}
        # Percentage change in Alpha power
        alpha_EO = bp_EO[ch].get("Alpha", 0)
        alpha_EC = bp_EC[ch].get("Alpha", 0)
        if alpha_EO != 0:
            metrics[ch]["Alpha_Change"] = ((alpha_EC - alpha_EO) / alpha_EO) * 100
        else:
            metrics[ch]["Alpha_Change"] = np.nan
        
        # Theta/Beta ratio using EO data
        theta = bp_EO[ch].get("Theta", np.nan)
        beta = bp_EO[ch].get("Beta", np.nan)
        if beta and beta != 0:
            metrics[ch]["Theta_Beta_Ratio"] = theta / beta
        else:
            metrics[ch]["Theta_Beta_Ratio"] = np.nan
    return metrics

def save_site_metrics(metrics, output_path):
    """
    Save computed site-specific metrics to a CSV file.
    
    Parameters:
      metrics (dict): Output from compute_site_metrics.
      output_path (str): File path to save the CSV.
    """
    rows = []
    for ch, met in metrics.items():
        row = {"Channel": ch}
        row.update(met)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Clinical metrics saved to: {output_path}")

def generate_site_reports(bp_EO, bp_EC, output_dir):
    """
    Compute clinical metrics for each channel and save a CSV summary.
    
    Parameters:
      bp_EO (dict): Band power dictionary for EO.
      bp_EC (dict): Band power dictionary for EC.
      output_dir (str): Directory where the CSV will be saved.
    """
    try:
        metrics = compute_site_metrics(bp_EO, bp_EC)
        csv_path = os.path.join(output_dir, "clinical_metrics.csv")
        save_site_metrics(metrics, csv_path)
    except Exception as e:
        logger.error(f"Failed to generate site reports: {e}")

def generate_full_site_reports(raw_eo, raw_ec, output_dir):
    """
    Generate detailed per-site, per-band plots for EO vs. EC.
    
    For each channel (site) and each frequency band (from processing.BANDS), this function:
      - Creates a subfolder for the channel.
      - Generates a PSD overlay plot comparing EO and EC.
      - Generates a waveform overlay plot comparing EO and EC.
      - Generates a difference bar plot comparing EO vs. EC band power.
      
    All plots are generated in dark mode and saved in an organized folder structure.
    
    Parameters:
      raw_eo (mne.io.Raw): Raw data for Eyes Open.
      raw_ec (mne.io.Raw): Raw data for Eyes Closed.
      output_dir (str): Base directory where per-site plots will be saved.
    
    Returns:
      dict: siteDict mapping sites to their plot paths for use in clinical_report.py.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Generating site reports in {output_dir}")
    if raw_eo is None or raw_ec is None:
        logger.error("Cannot generate site reports: Missing EO or EC data.")
        return {}

    # Verify channel alignment
    eo_channels = set(raw_eo.ch_names)
    ec_channels = set(raw_ec.ch_names)
    common_channels = eo_channels.intersection(ec_channels)
    if not common_channels:
        logger.error("No common channels between EO and EC data.")
        return {}
    logger.info(f"Common channels for site plots: {common_channels}")

    sfreq_eo = raw_eo.info['sfreq']
    sfreq_ec = raw_ec.info['sfreq']
    if sfreq_eo != sfreq_ec:
        logger.warning(f"Sampling frequencies differ: EO={sfreq_eo}, EC={sfreq_ec}. Using EO sampling frequency.")
    sfreq = sfreq_eo

    site_dict = {ch: {} for ch in common_channels}  # Initialize siteDict
    
    # Check disk space
    total, used, free = shutil.disk_usage(output_dir)
    logger.debug(f"Disk space - Total: {total / (2**30):.2f} GB, Used: {used / (2**30):.2f} GB, Free: {free / (2**30):.2f} GB")
    if free < 2**30:  # Less than 1 GB free
        logger.warning("Low disk space detected. May cause issues saving plots.")

    for ch in common_channels:
        # Create a folder for each channel/site
        ch_folder = os.path.join(output_dir, ch)
        psd_folder = os.path.join(ch_folder, "PSD_Overlay")
        wave_folder = os.path.join(ch_folder, "Waveform_Overlay")
        diff_folder = os.path.join(ch_folder, "Difference")
        try:
            os.makedirs(psd_folder, exist_ok=True)
            os.makedirs(wave_folder, exist_ok=True)
            os.makedirs(diff_folder, exist_ok=True)
            logger.debug(f"Created directories for {ch}: {psd_folder}, {wave_folder}, {diff_folder}")
        except Exception as e:
            logger.error(f"Failed to create directories for {ch}: {e}")
            continue
        
        # Extract signals for this channel (convert to microvolts)
        try:
            eo_sig = raw_eo.get_data(picks=[ch])[0] * 1e6
            ec_sig = raw_ec.get_data(picks=[ch])[0] * 1e6
            logger.info(f"Channel {ch} - EO samples: {len(eo_sig)}, EC samples: {len(ec_sig)}")
        except Exception as e:
            logger.error(f"Failed to extract data for channel {ch}: {e}")
            continue
        
        if len(eo_sig) < 32 or len(ec_sig) < 32:
            logger.warning(f"Insufficient samples for {ch}: EO={len(eo_sig)}, EC={len(ec_sig)}.")
            continue
        
        # Loop through each frequency band defined in processing.BANDS
        for band_name, band_range in processing.BANDS.items():
            site_dict[ch][band_name] = {}
            try:
                # Log memory usage before plotting
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                logger.debug(f"Memory usage before plotting {ch} {band_name}: {mem_info.rss / (2**20):.2f} MB")
                
                # Generate PSD overlay plot
                try:
                    logger.info(f"Generating PSD overlay for {ch} {band_name}.")
                    fig_psd = plotting.plot_band_psd_overlay(eo_sig, ec_sig, sfreq, band_range, ch, band_name, colors=("cyan", "magenta"))
                    psd_path = os.path.join(psd_folder, f"{ch}_PSD_{band_name}.png")
                    if fig_psd:
                        fig_psd.savefig(psd_path, facecolor='black')
                        plt.close(fig_psd)
                        logger.info(f"Saved PSD overlay for {ch} {band_name} to {psd_path}")
                        site_dict[ch][band_name]["psd"] = os.path.relpath(psd_path, start=output_dir)
                    else:
                        logger.warning(f"Failed to generate PSD plot for {ch} {band_name}.")
                        site_dict[ch][band_name]["psd"] = ""
                except Exception as e:
                    logger.error(f"Failed to generate/save PSD plot for {ch} {band_name}: {e}")
                    site_dict[ch][band_name]["psd"] = ""
                
                # Generate waveform overlay plot
                try:
                    logger.info(f"Generating waveform overlay for {ch} {band_name}.")
                    fig_wave = plotting.plot_band_waveform_overlay(eo_sig, ec_sig, sfreq, band_range, ch, band_name, colors=("cyan", "magenta"), epoch_length=10)
                    wave_path = os.path.join(wave_folder, f"{ch}_Waveform_{band_name}.png")
                    if fig_wave:
                        fig_wave.savefig(wave_path, facecolor='black')
                        plt.close(fig_wave)
                        logger.info(f"Saved waveform overlay for {ch} {band_name} to {wave_path}")
                        site_dict[ch][band_name]["wave"] = os.path.relpath(wave_path, start=output_dir)
                    else:
                        logger.warning(f"Failed to generate waveform plot for {ch} {band_name}.")
                        site_dict[ch][band_name]["wave"] = ""
                except Exception as e:
                    logger.error(f"Failed to generate/save waveform plot for {ch} {band_name}: {e}")
                    site_dict[ch][band_name]["wave"] = ""
                
                # Generate difference bar plot
                try:
                    logger.info(f"Generating difference bar plot for {ch} {band_name}.")
                    power_eo = processing.compute_band_power(eo_sig, sfreq, band_range)
                    power_ec = processing.compute_band_power(ec_sig, sfreq, band_range)
                    if np.isnan(power_eo) or np.isnan(power_ec):
                        logger.warning(f"Invalid band power for {ch} {band_name}: EO={power_eo}, EC={power_ec}")
                        site_dict[ch][band_name]["diff_bar"] = ""
                        continue
                    fig_diff, ax = plt.subplots(figsize=(4, 4), facecolor='black')
                    ax.bar(["EO", "EC"], [power_eo, power_ec], color=["cyan", "magenta"])
                    ax.set_title(f"{ch} {band_name} Difference", color='white', fontsize=10)
                    ax.set_ylabel("Power", color='white')
                    ax.tick_params(colors='white')
                    fig_diff.tight_layout()
                    diff_path = os.path.join(diff_folder, f"{ch}_Difference_{band_name}.png")
                    fig_diff.savefig(diff_path, facecolor='black')
                    plt.close(fig_diff)
                    logger.info(f"Saved difference bar plot for {ch} {band_name} to {diff_path}")
                    site_dict[ch][band_name]["diff_bar"] = os.path.relpath(diff_path, start=output_dir)
                except Exception as e:
                    logger.error(f"Failed to generate/save difference bar plot for {ch} {band_name}: {e}")
                    site_dict[ch][band_name]["diff_bar"] = ""
                
                # Log memory usage after plotting
                mem_info = process.memory_info()
                logger.debug(f"Memory usage after plotting {ch} {band_name}: {mem_info.rss / (2**20):.2f} MB")
                
            except Exception as e:
                logger.error(f"Unexpected error in plotting loop for {ch} {band_name}: {e}")
                with open(os.path.join(output_dir, "clinical_errors.log"), "a", encoding="utf-8") as f:
                    f.write(f"Unexpected error in plotting loop for {ch} {band_name}: {e}\n")
                # Ensure site_dict has entries even if plots fail
                site_dict[ch][band_name]["psd"] = ""
                site_dict[ch][band_name]["wave"] = ""
                site_dict[ch][band_name]["diff_bar"] = ""
                continue
    
    return site_dict
