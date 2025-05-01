from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor, black
import os
from datetime import datetime

MAX_PAGES = 50  # Define at the top

# ===== PSD Computation Imports =====
try:
    # MNE 1.0+: Use Raw object's built-in method
    def compute_raw_psd(raw, fmin=1, fmax=40):
        return raw.compute_psd(method='multitaper', fmin=fmin, fmax=fmax)
except ImportError:
    raise ImportError("PSD computation requires MNE-Python ≥1.0. Upgrade with: pip install --upgrade mne")

def flag_abnormal(value: float, metric: str) -> str:
    """
    Flags abnormal values based on metric-specific thresholds.
    Args:
        value: Metric value (e.g., power, coherence).
        metric: Metric name (e.g., "Alpha", "Coherence_Alpha").
    Returns:
        str: "High", "Low", or "Within normative range".
    """
    thresholds = {
        "Delta": (0, 20),  # Example thresholds (µV²)
        "Theta": (0, 10),
        "Alpha": (0, 15),
        "SMR": (0, 5),
        "Beta": (0, 8),
        "HighBeta": (0, 5),
        "Coherence_Alpha": (0, 0.8),  # Example coherence range
        "Frontal_Asymmetry": (-0.2, 0.2),
    }
    lower, upper = thresholds.get(metric, (None, None))
    if lower is None:
        return "N/A"
    if value < lower:
        return "Low"
    elif value > upper:
        return "High"
    return "Within normative range"

def generic_interpretation(band: str, value: float) -> str:
    """
    Provides a generic interpretation for abnormal values.
    """
    flag = flag_abnormal(value, band)
    if flag == "High":
        return f"Elevated {band} may indicate hyperactivation."
    elif flag == "Low":
        return f"Reduced {band} may suggest hypoactivation."
    return "Normal"

def get_table_style(header_color: str = "#D3D3D3") -> TableStyle:
    return TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, black),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor("#F5F5F5")),
    ])

def build_pdf_report(report_output_dir: Path, band_powers: dict = None, instability_indices: dict = None, source_localization: dict = None, vigilance_plots: dict = None, channels: list = None, global_metrics: dict = None, coherence: dict = None, zscores: dict = None, connectivity: dict = None, site_metrics: dict = None):
    """
    Builds a PDF report from EEG analysis results with enhanced metrics and visualizations.

    Args:
        report_output_dir (Path): Directory containing EEG analysis results.
        band_powers (dict, optional): Band powers for EO and EC conditions.
        instability_indices (dict, optional): Instability indices for EO and EC conditions.
        source_localization (dict, optional): Source localization results.
        vigilance_plots (dict, optional): Paths to vigilance plots.
        channels (list, optional): List of channel names.
        global_metrics (dict, optional): Global metrics (e.g., Frontal_Asymmetry).
        coherence (dict, optional): Coherence results.
        zscores (dict, optional): Z-score results.
        connectivity (dict, optional): Connectivity matrices per band and condition.
        site_metrics (dict, optional): Site-specific metrics for abnormality flagging.
    """
    pdf_output_path = report_output_dir / "clinical_report.pdf"
    clinical_interpretations_path = report_output_dir / "clinical_interpretations.txt"
    pyramid_mappings_path = report_output_dir / "pyramid_mappings.txt"
    topomap_dir = report_output_dir / "topomaps"
    site_plot_dir = report_output_dir / "per_site_plots"

    # Ensure instability_indices has the expected structure
    if not isinstance(instability_indices, dict):
        instability_indices = {"EO": {}, "EC": {}}
    
    for condition in ["EO", "EC"]:
        if condition not in instability_indices:
            instability_indices[condition] = {}
        
        # Initialize band dictionaries if missing
        for band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]:
            if band not in instability_indices[condition]:
                instability_indices[condition][band] = {}

    doc = SimpleDocTemplate(
        str(pdf_output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="EEG Clinical Report",
        author="The Squiggle Interpreter"
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CenteredTitle',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=HexColor("#2E4053"),
        fontName="Helvetica-Bold"
    ))
    styles.add(ParagraphStyle(
        name='SectionHeading',
        fontSize=14,
        leading=18,
        alignment=TA_LEFT,
        textColor=HexColor("#34495E"),
        fontName="Helvetica-Bold",
        spaceAfter=12
    ))
    styles.add(ParagraphStyle(
        name='NormalText',
        fontSize=10,
        leading=12,
        alignment=TA_LEFT,
        textColor=HexColor("#4A4A4A"),
        fontName="Helvetica",
        spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name='Caption',
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
        textColor=HexColor("#4A4A4A"),
        fontName="Helvetica-Oblique",
        spaceAfter=6
    ))

    elements = []
    toc_entries = []
    current_page = [1]

    def add_section(title, content_elements):
        toc_entries.append((title, current_page[0]))
        elements.append(Paragraph(title, styles['SectionHeading']))
        elements.extend(content_elements)
        elements.append(PageBreak())
        current_page[0] += 1
        if current_page[0] > MAX_PAGES:
            print(f"⚠️ Report truncated to {MAX_PAGES} pages. Consider splitting data.")

    def add_page_numbers(canvas, doc):
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(HexColor("#4A4A4A"))
        canvas.drawRightString(7.5 * inch, 0.5 * inch, text)

    # Cover Page
    elements.append(Paragraph("The Squiggle Interpreter", styles['CenteredTitle']))
    elements.append(Spacer(1, 0.5 * inch))
    logo_path = report_output_dir / "logo.png"
    if logo_path.exists():
        elements.append(Image(str(logo_path), width=2 * inch, height=2 * inch))
        elements.append(Spacer(1, 0.2 * inch))
    elements.append(Paragraph("Comprehensive EEG Clinical Report", styles['Title']))
    elements.append(Spacer(1, 0.2 * inch))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(f"Generated on: {timestamp}", styles['NormalText']))
    elements.append(PageBreak())
    current_page[0] += 1

    # Key Findings Section
    content = []
    content.append(Paragraph("Key Findings", styles['SectionHeading']))
    findings = []
    if band_powers:
        for condition in ["EO", "EC"]:
            for channel, powers in band_powers.get(condition, {}).items():
                for band, power in powers.items():
                    flag = flag_abnormal(power, band)
                    if flag != "Within normative range":
                        findings.append(f"{flag} {band} power ({power:.2f} µV²) in {channel} ({condition}): {generic_interpretation(band, power)}")
    if instability_indices:
        for condition in ["EO", "EC"]:
            for band, channels in instability_indices.get(condition, {}).items():
                for channel, variance in channels.items():
                    if variance > 1.5:
                        findings.append(f"Unstable {band} in {channel} (variance {variance:.2f}, {condition})")
    if global_metrics:
        for metric, value in global_metrics.items():
            flag = flag_abnormal(value, metric)
            if flag != "Within normative range":
                findings.append(f"{flag} {metric.replace('_', ' ')} ({value:.2f})")
    if site_metrics:
        for ch, metrics in site_metrics.items():
            for metric, value in metrics.items():
                if metric.startswith("Coherence_Alpha") and flag_abnormal(value, "Coherence_Alpha") != "Within normative range":
                    findings.append(f"{flag} {metric} ({value:.2f}) for {ch}")
    if not findings:
        findings.append("No significant abnormalities detected.")
    for finding in findings[:5]:
        content.append(Paragraph(finding, styles['NormalText']))
    add_section("Key Findings", content)

    # TOC Placeholder
    toc_placeholder = len(elements)
    elements.append(Paragraph("Table of Contents", styles['SectionHeading']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(PageBreak())
    current_page[0] += 1

    # Summary Section
    if pyramid_mappings_path.exists():
        content = []
        with open(pyramid_mappings_path, "r", encoding="utf-8") as f:
            mappings = f.readlines()
        summary_line = next((line for line in mappings if "Suggested Level" in line), "No summary available.")
        content.append(Paragraph(summary_line.strip(), styles["NormalText"]))
        content.append(Paragraph("This report includes detailed EEG analysis, including band powers, connectivity, and vigilance states.", styles["NormalText"]))
        add_section("Summary", content)

    # Global Metrics Section
    if global_metrics:
        content = []
        data = [["Metric", "Value", "Status"]]
        for metric, value in global_metrics.items():
            data.append([metric.replace("_", " ").title(), f"{value:.2f}", flag_abnormal(value, metric)])
        table = Table(data, colWidths=[2.5 * inch, 1.5 * inch, 2 * inch])
        table.setStyle(get_table_style())
        content.append(table)
        add_section("Global Metrics", content)

    # Clinical Interpretations
    if clinical_interpretations_path.exists():
        content = []
        with open(clinical_interpretations_path, "r", encoding="utf-8") as f:
            interpretations = f.readlines()
        for line in interpretations:
            content.append(Paragraph(line.strip(), styles["NormalText"]))
            content.append(Spacer(1, 0.1 * inch))
        add_section("Clinical Interpretations", content)

    # Pyramid Mappings
    if pyramid_mappings_path.exists():
        content = []
        with open(pyramid_mappings_path, "r", encoding="utf-8") as f:
            mappings = f.readlines()
        for line in mappings:
            content.append(Paragraph(line.strip(), styles["NormalText"]))
            content.append(Spacer(1, 0.1 * inch))
        add_section("Pyramid Mappings", content)

    # Eyes Open Analysis
    if band_powers.get("EO") or instability_indices.get("EO"):
        content = []
        if band_powers and "EO" in band_powers:
            content.append(Paragraph("Band Powers (Eyes Open)", styles['NormalText']))
            data = [["Channel", "Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]]
            for channel, powers in band_powers["EO"].items():
                row = [channel]
                for band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]:
                    row.append(f"{powers.get(band, 0):.2f} µV²")
                data.append(row)
            table = Table(data)
            table.setStyle(get_table_style())
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
        if instability_indices and "EO" in instability_indices:
            for band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]:
                content.append(Paragraph(f"Instability Index ({band}, Eyes Open)", styles['NormalText']))
                data = [["Channel", "Variance (µV²)"]]
                for channel, variance in instability_indices["EO"][band].items():
                    data.append([channel, f"{variance:.2f}"])
                table = Table(data)
                table.setStyle(get_table_style())
                content.append(table)
                content.append(Spacer(1, 0.2 * inch))
        add_section("Eyes Open Analysis", content)

    # Eyes Closed Analysis
    if band_powers.get("EC") or instability_indices.get("EC"):
        content = []
        if band_powers and "EC" in band_powers:
            content.append(Paragraph("Band Powers (Eyes Closed)", styles['NormalText']))
            data = [["Channel", "Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]]
            for channel, powers in band_powers["EC"].items():
                row = [channel]
                for band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]:
                    row.append(f"{powers.get(band, 0):.2f} µV²")
                data.append(row)
            table = Table(data)
            table.setStyle(get_table_style())
            content.append(table)
            content.append(Spacer(1, 0.2 * inch))
        if instability_indices and "EC" in instability_indices:
            for band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]:
                content.append(Paragraph(f"Instability Index ({band}, Eyes Closed)", styles['NormalText']))
                data = [["Channel", "Variance (µV²)"]]
                for channel, variance in instability_indices["EC"][band].items():
                    data.append([channel, f"{variance:.2f}"])
                table = Table(data)
                table.setStyle(get_table_style())
                content.append(table)
                content.append(Spacer(1, 0.2 * inch))
        add_section("Eyes Closed Analysis", content)

    # Connectivity Matrices
    if connectivity and channels:
        content = []
        for condition in ["EO", "EC"]:
            if condition in connectivity:
                content.append(Paragraph(f"Connectivity ({condition})", styles['NormalText']))
                for band in connectivity[condition]:
                    content.append(Paragraph(f"{band} Band", styles['NormalText']))
                    data = [[""] + channels]
                    matrix = connectivity[condition][band]
                    for i, ch1 in enumerate(channels):
                        row = [ch1]
                        for j, ch2 in enumerate(channels):
                            value = matrix[i, j]
                            flag = flag_abnormal(value, "Coherence_" + band) if band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"] else "N/A"
                            row.append(f"{value:.2f} ({flag})" if flag != "N/A" else f"{value:.2f}")
                        data.append(row)
                    table = Table(data)
                    table.setStyle(get_table_style())
                    content.append(table)
                    content.append(Spacer(1, 0.2 * inch))
        add_section("Connectivity Analysis", content)

    # Coherence and Z-Scores
    if coherence or zscores:
        content = []
        if coherence:
            for condition in ["EO", "EC"]:
                if condition in coherence:
                    content.append(Paragraph(f"Coherence ({condition})", styles['NormalText']))
                    data = [["Band", "Channel Pair", "Coherence", "Status"]]
                    for band, pairs in coherence[condition].items():
                        for (ch1, ch2), value in pairs.items():
                            flag = flag_abnormal(value, f"Coherence_{band}")
                            data.append([band, f"{ch1}-{ch2}", f"{value:.2f}", flag])
                    table = Table(data)
                    table.setStyle(get_table_style())
                    content.append(table)
                    content.append(Spacer(1, 0.2 * inch))
        if zscores:
            for condition in ["EO", "EC"]:
                if condition in zscores:
                    content.append(Paragraph(f"Z-Scores ({condition})", styles['NormalText']))
                    data = [["Channel", "Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]]
                    for channel, scores in zscores[condition].items():
                        row = [channel]
                        for band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]:
                            # Handle both dictionary and list formats
                            if isinstance(scores, dict):
                                value = scores.get(band, 0)
                            else:
                                # If scores is a list, assume it's in the same order as bands
                                try:
                                    band_idx = ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"].index(band)
                                    value = scores[band_idx] if band_idx < len(scores) else 0
                                except (ValueError, IndexError):
                                    value = 0
                            flag = flag_abnormal(value, band)
                            row.append(f"{value:.2f} ({flag})")
                        data.append(row)
                    table = Table(data)
                    table.setStyle(get_table_style())
                    content.append(table)
                    content.append(Spacer(1, 0.2 * inch))
        add_section("Coherence and Z-Scores", content)

    # Vigilance Plots
    if vigilance_plots:
        for condition in ["EO", "EC"]:
            if condition in vigilance_plots:
                content = []
                hypnogram_path = vigilance_plots[condition]["hypnogram"]
                if hypnogram_path.exists():
                    content.append(Paragraph(f"Vigilance Hypnogram ({condition})", styles['NormalText']))
                    content.append(Image(str(hypnogram_path), width=6.5 * inch, height=3.5 * inch))
                    content.append(Paragraph("Hypnogram showing vigilance state transitions.", styles['Caption']))
                    content.append(Spacer(1, 0.2 * inch))
                strip_path = vigilance_plots[condition]["strip"]
                if strip_path.exists():
                    content.append(Paragraph(f"Vigilance Strip ({condition})", styles['NormalText']))
                    content.append(Image(str(strip_path), width=6.5 * inch, height=1.5 * inch))
                    content.append(Paragraph("Strip plot of vigilance states over time.", styles['Caption']))
                    content.append(Spacer(1, 0.2 * inch))
                add_section(f"Vigilance Analysis ({condition})", content)

    # Topographic Maps
    if topomap_dir.exists():
        content = []
        for img_file in sorted(topomap_dir.glob("*.png")):
            try:
                band = img_file.stem.split("_")[1]
                condition = img_file.stem.split("_")[-1]
                caption = f"Topographic map of {band} power ({condition})"
                if instability_indices and condition in instability_indices and band in instability_indices[condition]:
                    unstable_channels = [ch for ch, v in instability_indices[condition][band].items() if v > 1.5]
                    if unstable_channels:
                        caption += f". Unstable in: {', '.join(unstable_channels)}"
                if site_metrics:
                    abnormal_channels = [ch for ch in site_metrics if flag_abnormal(site_metrics[ch].get(f"{band}_Power", np.nan), band) != "Within normative range"]
                    if abnormal_channels:
                        caption += f". Abnormal in: {', '.join(abnormal_channels)}"
                content.append(Image(str(img_file), width=6.5 * inch, height=3.5 * inch))
                content.append(Paragraph(caption, styles['Caption']))
                content.append(Spacer(1, 0.2 * inch))
            except Exception as e:
                print(f"⚠️ Failed to load {img_file}: {str(e)}")
        add_section("Topographic Maps", content)
    else:
        print(f"⚠️ Topomap directory not found: {topomap_dir}")

    # Source Localization
    if source_localization:
        for condition in ["EO", "EC"]:
            if condition in source_localization:
                content = []
                for band, methods in source_localization[condition].items():
                    content.append(Paragraph(f"Band: {band}", styles['NormalText']))
                    for method, img_path in methods.items():
                        img_full_path = report_output_dir / img_path
                        if img_full_path.exists():
                            content.append(Paragraph(f"Method: {method}", styles['NormalText']))
                            content.append(Image(str(img_full_path), width=6.5 * inch, height=3.5 * inch))
                            content.append(Paragraph(f"Source localization for {band} ({condition}, {method})", styles['Caption']))
                            content.append(Spacer(1, 0.2 * inch))
                add_section(f"Source Localization ({condition})", content)

    # Site Plots
    selected_sites = {"O1", "O2", "PZ", "CZ", "FZ", "F3", "F4", "T3", "T4"}
    for site in sorted(selected_sites):
        site_path = site_plot_dir / site
        if not site_path.exists():
            continue
        content = []
        plot_types = ["PSD_Overlay", "Waveform_Overlay", "Difference", "ERP_Overlay"]
        for plot_type in plot_types:
            plot_dir = site_path / plot_type
            if plot_dir.exists():
                for img in sorted(plot_dir.glob("*.png")):
                    caption = f"{plot_type.replace('_', ' ')} for {site}"
                    if site_metrics and site in site_metrics:
                        for band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]:
                            power = site_metrics[site].get(f"{band}_Power", np.nan)
                            flag = flag_abnormal(power, band)
                            if flag != "Within normative range":
                                caption += f". {flag} {band} power ({power:.2f} µV²): {generic_interpretation(band, power)}"
                    content.append(Image(str(img), width=6.5 * inch, height=3.5 * inch))
                    content.append(Paragraph(caption, styles['Caption']))
                    content.append(Spacer(1, 0.1 * inch))
        add_section(f"Site: {site}", content)

    # Insert TOC
    toc_content = [[Paragraph("Section", styles['NormalText']), Paragraph("Page", styles['NormalText'])]]
    for title, page in toc_entries:
        toc_content.append([Paragraph(title, styles['NormalText']), Paragraph(str(page), styles['NormalText'])])
    toc_table = Table(toc_content, colWidths=[5.5 * inch, 1 * inch])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#D3D3D3")),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, black),
        ('BACKGROUND', (0, 1), (-1, -1), HexColor("#F5F5F5")),
    ]))
    elements[toc_placeholder:toc_placeholder + 2] = [Paragraph("Table of Contents", styles['SectionHeading']), toc_table]

    doc.build(elements, onFirstPage=add_page_numbers, onLaterPages=add_page_numbers)
    print(f"PDF report generated at: {pdf_output_path}")

def compute_psd(raw, fmin=1, fmax=40):
    """Version-agnostic PSD computation"""
    try:
        # MNE 1.0+ style
        return raw.compute_psd(method='multitaper', fmin=fmin, fmax=fmax)
    except AttributeError:
        # Older MNE style
        return raw.compute_psd(
            method='multitaper', 
            fmin=fmin, 
            fmax=fmax, 
            tmin=None, 
            tmax=None
        )
