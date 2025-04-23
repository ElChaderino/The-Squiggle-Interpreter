import os
from jinja2 import Environment, FileSystemLoader

def build_html_report(report_data, output_path):
    """
    Render an interactive HTML report using a Jinja2 template.
    
    This report includes:
      - Global topomaps for EO and EC (one image per frequency band)
      - Global waveform grid images (keyed by band for EO)
      - Pseudo-ERP plots for EO and EC
      - A global coherence matrix image (or images keyed by band/condition)
      - A robust z-score topomap (or topomaps keyed by band/condition)
      - TFR and ICA images (keyed by condition and band, if applicable)
      - (Optional) Dynamic dropdowns for per-site and per-band analysis using site_list, band_list, and site_dict.
    
    The report_data dictionary should include keys such as:
      - global_topomaps: dict, e.g. {"EO": {"Alpha": "topomap_Alpha.png", ...}, "EC": {...}}
      - global_waveforms: dict, e.g. {"Alpha": "waveforms_Alpha.png", "Theta": "waveforms_Theta.png", ...}
      - global_erp: dict, e.g. {"EO": "erp_EO.png", "EC": "erp_EC.png"}
      - coherence: dict, e.g. {"EO": {"Alpha": "coherence_Alpha.png"}, "EC": {…}}
      - zscore: dict, e.g. {"EO": {"Alpha": "zscore_Alpha.png"}, "EC": {…}}
      - tfr: dict, e.g. {"EO": {"Alpha": "tfr_Alpha.png"}, "EC": {…}}
      - ica: dict, e.g. {"EO": "ica_EO.png", "EC": ""}
      - site_list: list of site names.
      - band_list: list of frequency band names.
      - site_dict: dict mapping each site to its per-band plot filenames (e.g., {"PSD": ..., "wave": ...}).
      - Optionally, base path keys such as "global_topomaps_path", "global_waveforms_path", etc., that are used by the HTML template.
    
    Adjust the template and report_data as needed to capture all aspects of your analysis report.
    """
    template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("report_template.html")
    rendered_html = template.render(**report_data)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    print(f"Interactive HTML report generated at: {output_path}")
