#!/usr/bin/env python
import os
import sys
import threading
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import mne

from modules.vigilance import plot_vigilance_hypnogram
from modules import io_utils, processing, plotting, report, clinical, vigilance
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

# --- Utility: Group EDF files by subject ---
def find_subject_edf_files(directory):
    """
    Find and group EDF files by subject.
    Assumes filenames are in the format: <subjectID>eo.edf and <subjectID>ec.edf
    e.g., "c1eo.edf", "c1ec.edf", "e1eo.edf", "e1ec.edf"
    Returns a dict: { subjectID: {"EO": filename, "EC": filename}, ... }
    """
    edf_files = [f for f in os.listdir(directory) if f.lower().endswith('.edf')]
    subjects = {}
    for f in edf_files:
        f_lower = f.lower()
        subject_id = f_lower[:2]
        if subject_id not in subjects:
            subjects[subject_id] = {"EO": None, "EC": None}
        if "eo" in f_lower:
            subjects[subject_id]["EO"] = f
        elif "ec" in f_lower:
            subjects[subject_id]["EC"] = f
    return subjects

def live_eeg_display(stop_event, update_interval=0.5):
    def generate_eeg_wave(num_points=80):
        x = np.linspace(0, 4 * np.pi, num_points)
        wave = np.sin(x) + np.random.normal(0, 0.3, size=num_points)
        gradient = " .:-=+*#%@"
        norm_wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-10)
        indices = (norm_wave * (len(gradient) - 1)).astype(int)
        return "".join(gradient[i] for i in indices)
    
    def get_random_quote():
        quotes = [
            "The Dude abides.",
            "That rug really tied the room together.",
            "Yeah, well, you know, that's just, like, your opinion, man.",
            "Sometimes you eat the bear, and sometimes, well, the bear eats you.",
            "Watch the squiggles, man. They're pure EEG poetry."
        ]
        return np.random.choice(quotes)
    
    console = Console()
    with Live(refresh_per_second=10, console=console) as live:
        while not stop_event.is_set():
            line = generate_eeg_wave()
            quote = get_random_quote() if np.random.rand() < 0.1 else ""
            text = f"[bold green]{line}[/bold green]"
            if quote:
                text += f"\n[bold red]{quote}[/bold red]"
            panel = Panel(text, title="Live EEG Display", subtitle="Simulated Waveform", style="white on black")
            live.update(panel)
            time.sleep(update_interval)

def main():
    project_dir = os.getcwd()
    overall_output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(overall_output_dir, exist_ok=True)
    
    # Ask user if they want to use CSD for graphing only.
    csd_choice = input("Use current source density (CSD) for graphs only? (y/n, default n in 5 sec): ")
    use_csd_for_graphs = True if csd_choice.lower() == "y" else False
    print(f"Using CSD for graphs: {use_csd_for_graphs}")
    
    # --- Batch processing: Group EDF files by subject ---
    subject_edf_groups = find_subject_edf_files(project_dir)
    print("Found subject EDF files:", subject_edf_groups)
    
    # Process each subject separately.
    for subject, files in subject_edf_groups.items():
        subject_folder = os.path.join(overall_output_dir, subject)
        os.makedirs(subject_folder, exist_ok=True)
        
        # Create subject-specific output folders.
        folders = {
            "topomaps_eo": os.path.join(subject_folder, "topomaps", "EO"),
            "topomaps_ec": os.path.join(subject_folder, "topomaps", "EC"),
            "waveforms_eo": os.path.join(subject_folder, "waveforms", "EO"),
            "erp": os.path.join(subject_folder, "erp"),
            "coherence_eo": os.path.join(subject_folder, "coherence", "EO"),
            "coherence_ec": os.path.join(subject_folder, "coherence", "EC"),
            "zscore_eo": os.path.join(subject_folder, "zscore", "EO"),
            "zscore_ec": os.path.join(subject_folder, "zscore", "EC"),
            "tfr_eo": os.path.join(subject_folder, "tfr", "EO"),
            "tfr_ec": os.path.join(subject_folder, "tfr", "EC"),
            "ica_eo": os.path.join(subject_folder, "ica", "EO"),
            "source": os.path.join(subject_folder, "source_localization"),
            "detailed": os.path.join(subject_folder, "detailed_site_plots"),
            "vigilance": os.path.join(subject_folder, "vigilance")
        }
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Start live display (optional)
        stop_event = threading.Event()
        live_thread = threading.Thread(target=live_eeg_display, args=(stop_event,))
        live_thread.start()
        
        # Use the subject's EDF files.
        eo_file = files["EO"] if files["EO"] else files["EC"]
        ec_file = files["EC"] if files["EC"] else files["EO"]
        print(f"Subject {subject}: EO file: {eo_file}, EC file: {ec_file}")
        
        # Load data without CSD for forward/inverse modeling.
        raw_eo = io_utils.load_eeg_data(os.path.join(project_dir, eo_file), use_csd=False)
        raw_ec = io_utils.load_eeg_data(os.path.join(project_dir, ec_file), use_csd=False)
        print("Loaded data for subject", subject)
        
        # Create CSD copies for graphing if requested.
        if use_csd_for_graphs:
            raw_eo_csd = raw_eo.copy().load_data()
            raw_ec_csd = raw_ec.copy().load_data()
            try:
                raw_eo_csd = mne.preprocessing.compute_current_source_density(raw_eo_csd)
                print("CSD applied for graphs (EO).")
            except Exception as e:
                print("CSD for graphs (EO) failed:", e)
                raw_eo_csd = raw_eo  # fallback
            try:
                raw_ec_csd = mne.preprocessing.compute_current_source_density(raw_ec_csd)
                print("CSD applied for graphs (EC).")
            except Exception as e:
                print("CSD for graphs (EC) failed:", e)
                raw_ec_csd = raw_ec  # fallback
        else:
            raw_eo_csd = raw_eo
            raw_ec_csd = raw_ec
        
        # --- Integrate Vigilance Module ---
        # Use a 2-second epoch (to allow a longer filter transition band)
        vigilance_states = vigilance.compute_vigilance_states(raw_eo, epoch_length=2.0)

        # Plot the hypnogram and save it without displaying
        fig = vigilance.plot_vigilance_hypnogram(vigilance_states, epoch_length=2.0)
        hypno_path = os.path.join(folders["vigilance"], "vigilance_hypnogram.png")
        fig.savefig(hypno_path, facecolor='black')
        plt.close(fig)
        print(f"Saved vigilance hypnogram to {hypno_path}")


        print("Vigilance states (time in s, stage):")
        for t, stage in vigilance_states:
            print(f"{t:5.1f}s: {stage}")

        # Plot and save the original color vigilance strip (if needed)
        fig_strip = vigilance.plot_vigilance_strip(vigilance_states, epoch_length=2.0)
        vigilance_strip_path = os.path.join(folders["vigilance"], "vigilance_strip.png")
        fig_strip.savefig(vigilance_strip_path, facecolor='black')
        plt.close(fig_strip)
        print(f"Saved vigilance strip to {vigilance_strip_path}")

        
        # --- New: Plot a step-style hypnogram ---
        # (Make sure that modules.vigilance has the function plot_vigilance_hypnogram)
        try:
            fig_hypno = vigilance.plot_vigilance_hypnogram(vigilance_states, epoch_length=2.0)
            vigilance_hypno_path = os.path.join(folders["vigilance"], "vigilance_hypnogram.png")
            fig_hypno.savefig(vigilance_hypno_path, facecolor='black')
            plt.close(fig_hypno)
            print(f"Saved vigilance hypnogram to {vigilance_hypno_path}")
        except AttributeError:
            print("Error: 'plot_vigilance_hypnogram' not found in modules.vigilance. Please update the vigilance module accordingly.")
        
        # Continue with existing processing...
        bp_eo = processing.compute_all_band_powers(raw_eo)
        bp_ec = processing.compute_all_band_powers(raw_ec)
        print(f"Subject {subject} - Computed band powers for EO channels:", list(bp_eo.keys()))
        print(f"Subject {subject} - Computed band powers for EC channels:", list(bp_ec.keys()))
        
        clinical.generate_site_reports(bp_eo, bp_ec, subject_folder)
        
        band_list = list(processing.BANDS.keys())
        
        # --- Global Topomaps for each band ---
        for band in band_list:
            # EO Topomap.
            abs_eo = [bp_eo[ch][band] for ch in raw_eo.ch_names]
            rel_eo = []
            for ch in raw_eo.ch_names:
                total_power = sum(bp_eo[ch][b] for b in band_list)
                rel_eo.append(bp_eo[ch][band] / total_power if total_power else 0)
            fig_topo_eo = plotting.plot_topomap_abs_rel(abs_eo, rel_eo, raw_eo.info, band, "EO")
            eo_topo_path = os.path.join(folders["topomaps_eo"], f"topomap_{band}.png")
            fig_topo_eo.savefig(eo_topo_path, facecolor='black')
            plt.close(fig_topo_eo)
            print(f"Subject {subject}: Saved EO topomap for {band} to {eo_topo_path}")
            
            # EC Topomap.
            abs_ec = [bp_ec[ch][band] for ch in raw_ec.ch_names]
            rel_ec = []
            for ch in raw_ec.ch_names:
                total_power = sum(bp_ec[ch][b] for b in band_list)
                rel_ec.append(bp_ec[ch][band] / total_power if total_power else 0)
            fig_topo_ec = plotting.plot_topomap_abs_rel(abs_ec, rel_ec, raw_ec.info, band, "EC")
            ec_topo_path = os.path.join(folders["topomaps_ec"], f"topomap_{band}.png")
            fig_topo_ec.savefig(ec_topo_path, facecolor='black')
            plt.close(fig_topo_ec)
            print(f"Subject {subject}: Saved EC topomap for {band} to {ec_topo_path}")
        
        # --- Global Waveform Grids (EO) ---
        global_waveforms = {}
        data_eo = raw_eo.get_data() * 1e6
        sfreq = raw_eo.info['sfreq']
        for band, band_range in processing.BANDS.items():
            wf_fig = plotting.plot_waveform_grid(data_eo, raw_eo.ch_names, sfreq, band=band_range, epoch_length=10)
            wf_path = os.path.join(folders["waveforms_eo"], f"waveforms_{band}.png")
            wf_fig.savefig(wf_path, facecolor='black')
            plt.close(wf_fig)
            global_waveforms[band] = os.path.basename(wf_path)
            print(f"Subject {subject}: Saved EO waveform grid for {band} to {wf_path}")
        
        # --- Global ERP Plots ---
        erp_eo_fig = processing.compute_pseudo_erp(raw_eo)
        erp_eo_path = os.path.join(folders["erp"], "erp_EO.png")
        erp_eo_fig.savefig(erp_eo_path, facecolor='black')
        plt.close(erp_eo_fig)
        print(f"Subject {subject}: Saved ERP EO to {erp_eo_path}")
        
        erp_ec_fig = processing.compute_pseudo_erp(raw_ec)
        erp_ec_path = os.path.join(folders["erp"], "erp_EC.png")
        erp_ec_fig.savefig(erp_ec_path, facecolor='black')
        plt.close(erp_ec_fig)
        print(f"Subject {subject}: Saved ERP EC to {erp_ec_path}")
        
        # --- Global Coherence Maps for each band ---
        coherence_maps = {"EO": {}, "EC": {}}
        for band in band_list:
            band_range = processing.BANDS[band]
            # EO coherence.
            coh_matrix_eo = processing.compute_coherence_matrix(raw_eo.get_data() * 1e6, sfreq, band_range, nperseg=int(sfreq*2))
            fig_coh_eo = plotting.plot_coherence_matrix(coh_matrix_eo, raw_eo.ch_names)
            coh_path_eo = os.path.join(folders["coherence_eo"], f"coherence_{band}.png")
            fig_coh_eo.savefig(coh_path_eo, facecolor='black')
            plt.close(fig_coh_eo)
            coherence_maps["EO"][band] = os.path.basename(coh_path_eo)
            print(f"Subject {subject}: Saved EO coherence ({band}) to {coh_path_eo}")
            
            # EC coherence.
            coh_matrix_ec = processing.compute_coherence_matrix(raw_ec.get_data() * 1e6, sfreq, band_range, nperseg=int(sfreq*2))
            fig_coh_ec = plotting.plot_coherence_matrix(coh_matrix_ec, raw_ec.ch_names)
            coh_path_ec = os.path.join(folders["coherence_ec"], f"coherence_{band}.png")
            fig_coh_ec.savefig(coh_path_ec, facecolor='black')
            plt.close(fig_coh_ec)
            coherence_maps["EC"][band] = os.path.basename(coh_path_ec)
            print(f"Subject {subject}: Saved EC coherence ({band}) to {coh_path_ec}")
        
        # --- Global Robust Z-Score Topomaps ---
        norm_stats = {
            "Alpha": {"median": 18.0, "mad": 6.0},
            "Theta": {"median": 15.0, "mad": 5.0},
            "Delta": {"median": 20.0, "mad": 7.0},
            "SMR": {"median": 6.0, "mad": 2.0},
            "Beta": {"median": 5.0, "mad": 2.0},
            "HighBeta": {"median": 3.5, "mad": 1.5}
        }
        zscore_maps_eo = processing.compute_all_zscore_maps(raw_eo, norm_stats, epoch_len_sec=2.0)
        zscore_maps_ec = processing.compute_all_zscore_maps(raw_ec, norm_stats, epoch_len_sec=2.0)
        zscore_images_eo = {}
        zscore_images_ec = {}
        for band in band_list:
            if band in zscore_maps_eo and zscore_maps_eo[band] is not None:
                fig_zscore_eo = plotting.plot_zscore_topomap(zscore_maps_eo[band], raw_eo.info, band, "EO")
                zscore_path_eo = os.path.join(folders["zscore_eo"], f"zscore_{band}.png")
                fig_zscore_eo.savefig(zscore_path_eo, facecolor='black')
                plt.close(fig_zscore_eo)
                zscore_images_eo[band] = os.path.basename(zscore_path_eo)
                print(f"Subject {subject}: Saved EO z-score ({band}) to {zscore_path_eo}")
            if band in zscore_maps_ec and zscore_maps_ec[band] is not None:
                fig_zscore_ec = plotting.plot_zscore_topomap(zscore_maps_ec[band], raw_ec.info, band, "EC")
                zscore_path_ec = os.path.join(folders["zscore_ec"], f"zscore_{band}.png")
                fig_zscore_ec.savefig(zscore_path_ec, facecolor='black')
                plt.close(fig_zscore_ec)
                zscore_images_ec[band] = os.path.basename(zscore_path_ec)
                print(f"Subject {subject}: Saved EC z-score ({band}) to {zscore_path_ec}")
        
        # --- Global TFR Maps ---
        n_cycles = 2.0  # Example value; adjust as needed.
        tfr_maps_eo = processing.compute_all_tfr_maps(raw_eo, n_cycles, tmin=0.0, tmax=4.0)
        tfr_maps_ec = processing.compute_all_tfr_maps(raw_ec, n_cycles, tmin=0.0, tmax=4.0)
        tfr_images_eo = {}
        tfr_images_ec = {}
        for band in band_list:
            if band in tfr_maps_eo and tfr_maps_eo[band] is not None:
                fig_tfr_eo = plotting.plot_tfr(tfr_maps_eo[band], picks=0)
                tfr_path_eo = os.path.join(folders["tfr_eo"], f"tfr_{band}.png")
                fig_tfr_eo.savefig(tfr_path_eo, facecolor='black')
                plt.close(fig_tfr_eo)
                tfr_images_eo[band] = os.path.basename(tfr_path_eo)
                print(f"Subject {subject}: Saved TFR EO ({band}) to {tfr_path_eo}")
            else:
                print(f"Subject {subject}: TFR EO for {band} was not computed.")
            if band in tfr_maps_ec and tfr_maps_ec[band] is not None:
                fig_tfr_ec = plotting.plot_tfr(tfr_maps_ec[band], picks=0)
                tfr_path_ec = os.path.join(folders["tfr_ec"], f"tfr_{band}.png")
                fig_tfr_ec.savefig(tfr_path_ec, facecolor='black')
                plt.close(fig_tfr_ec)
                tfr_images_ec[band] = os.path.basename(tfr_path_ec)
                print(f"Subject {subject}: Saved TFR EC ({band}) to {tfr_path_ec}")
            else:
                print(f"Subject {subject}: TFR EC for {band} was not computed.")
        
        # --- Global ICA Components Plot (EO) ---
        ica = processing.compute_ica(raw_eo)
        fig_ica = plotting.plot_ica_components(ica, raw_eo)
        ica_path = os.path.join(folders["ica_eo"], "ica_EO.png")
        fig_ica.savefig(ica_path, facecolor='black')
        plt.close(fig_ica)
        print(f"Subject {subject}: Saved ICA EO to {ica_path}")
        
        # --- Global Source Localization ---
        raw_source_eo = raw_eo.copy()
        raw_source_ec = raw_ec.copy()
        raw_source_eo.set_eeg_reference("average", projection=False)
        raw_source_ec.set_eeg_reference("average", projection=False)
        print("EEG channels for source localization (EO):", mne.pick_types(raw_source_eo.info, meg=False, eeg=True))
        
        fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
        subjects_dir = os.path.dirname(fs_dir)
        subject_fs = "fsaverage"
        src = mne.setup_source_space(subject_fs, spacing="oct6", subjects_dir=subjects_dir, add_dist=False)
        conductivity = (0.3, 0.006, 0.3)
        bem_model = mne.make_bem_model(subject=subject_fs, ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
        bem_solution = mne.make_bem_solution(bem_model)
        
        fwd_eo = mne.make_forward_solution(raw_source_eo.info, trans="fsaverage", src=src,
                                           bem=bem_solution, eeg=True, meg=False, verbose=False)
        fwd_ec = mne.make_forward_solution(raw_source_ec.info, trans="fsaverage", src=src,
                                           bem=bem_solution, eeg=True, meg=False, verbose=False)
        
        events_eo = mne.make_fixed_length_events(raw_eo, duration=2.0)
        epochs_eo = mne.Epochs(raw_eo, events_eo, tmin=-0.1, tmax=0.4, baseline=(None, 0),
                                preload=True, verbose=False)
        cov_eo = mne.compute_covariance(epochs_eo, tmax=0., method="empirical", verbose=False)
        
        inv_op_eo = processing.compute_inverse_operator(raw_source_eo, fwd_eo, cov_eo)
        inv_op_ec = processing.compute_inverse_operator(raw_source_ec, fwd_ec, cov_eo)
        
        source_methods = {"LORETA": "MNE", "sLORETA": "sLORETA", "eLORETA": "eloreta"}
        source_localization = {"EO": {}, "EC": {}}
        for cond, raw_data, inv_op in [("EO", raw_eo, inv_op_eo), ("EC", raw_ec, inv_op_ec)]:
            for band in band_list:
                band_range = processing.BANDS[band]
                raw_band = raw_data.copy().filter(band_range[0], band_range[1], verbose=False)
                events = mne.make_fixed_length_events(raw_band, duration=2.0)
                epochs = mne.Epochs(raw_band, events, tmin=-0.1, tmax=0.4, baseline=(None, 0),
                                    preload=True, verbose=False)
                evoked = epochs.average()
                cond_folder = os.path.join(folders["source"], cond)
                os.makedirs(cond_folder, exist_ok=True)
                for method, method_label in source_methods.items():
                    try:
                        stc = processing.compute_source_localization(evoked, inv_op, lambda2=1.0/9.0, method=method_label)
                        fig_src = plotting.plot_source_estimate(stc, view="lateral", time_point=0.1, subjects_dir=subjects_dir)
                        src_filename = f"source_{cond}_{method}_{band}.png"
                        src_path = os.path.join(cond_folder, src_filename)
                        fig_src.savefig(src_path, dpi=150, bbox_inches="tight", facecolor="black")
                        plt.close(fig_src)
                        source_localization[cond].setdefault(band, {})[method] = cond + "/" + src_filename
                        print(f"Subject {subject}: Saved {method} source localization for {cond} {band} to {src_path}")
                    except Exception as e:
                        print(f"Error computing source localization for {cond} {band} with {method}: {e}")
        
        print("Source Localization dictionary:")
        print(source_localization)
        
        # --- Detailed Per-Site Reports ---
        from modules.clinical import generate_full_site_reports
        generate_full_site_reports(raw_eo, raw_ec, folders["detailed"])
        
        # --- Build per-site plot dictionary for the report ---
        site_list = raw_eo.ch_names
        site_dict = {}
        for site in site_list:
            site_dict[site] = {}
            for b in band_list:
                psd_path = os.path.join("detailed_site_plots", site, "PSD_Overlay", f"{site}_PSD_{b}.png")
                wave_path = os.path.join("detailed_site_plots", site, "Waveform_Overlay", f"{site}_Waveform_{b}.png")
                site_dict[site][b] = {"psd": psd_path, "wave": wave_path}
        
        report_data = {
            "global_topomaps": {
                "EO": {b: os.path.basename(os.path.join(folders["topomaps_eo"], f"topomap_{b}.png")) for b in band_list},
                "EC": {b: os.path.basename(os.path.join(folders["topomaps_ec"], f"topomap_{b}.png")) for b in band_list}
            },
            "global_waveforms": global_waveforms,
            "coherence": {
                "EO": coherence_maps["EO"],
                "EC": coherence_maps["EC"]
            },
            "global_erp": {
                "EO": os.path.basename(erp_eo_path),
                "EC": os.path.basename(erp_ec_path)
            },
            "zscore": {
                "EO": {b: os.path.basename(os.path.join(folders["zscore_eo"], f"zscore_{b}.png")) for b in band_list},
                "EC": {b: os.path.basename(os.path.join(folders["zscore_ec"], f"zscore_{b}.png")) for b in band_list}
            },
            "tfr": {
                "EO": {b: os.path.basename(os.path.join(folders["tfr_eo"], f"tfr_{b}.png")) for b in band_list},
                "EC": {b: os.path.basename(os.path.join(folders["tfr_ec"], f"tfr_{b}.png")) for b in band_list}
            },
            "ica": {
                "EO": os.path.basename(ica_path),
                "EC": ""
            },
            "source_localization": source_localization,
            "site_list": site_list,
            "band_list": band_list,
            "site_dict": site_dict,
            "global_topomaps_path": "topomaps",
            "global_waveforms_path": "waveforms",
            "coherence_path": "coherence",
            "global_erp_path": "erp",
            "tfr_path": "tfr",
            "ica_path": "ica",
            "sites_path": "",
            "source_path": "./source_localization"
        }
        
        subject_report_path = os.path.join(subject_folder, "eeg_report.html")
        report.build_html_report(report_data, subject_report_path)
        print(f"Subject {subject}: Generated interactive HTML report at {subject_report_path}")
        
        extension_script = os.path.join(project_dir, "extensions", "EEG_Extension.py")
        if os.path.exists(extension_script):
            print(f"Subject {subject}: Running extension script: {extension_script}")
            subprocess.run([sys.executable, extension_script])
        else:
            print(f"Subject {subject}: No extension script found in 'extensions' folder.")
        
        stop_event.set()
        live_thread.join()
        print("Live EEG display stopped for subject", subject)

if __name__ == "__main__":
    main()
