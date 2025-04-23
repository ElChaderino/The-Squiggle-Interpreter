from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.colors import HexColor, black  # Import black directly

import os
from datetime import datetime

class PageCounter:
    """A helper class to track page numbers during document creation."""
    def __init__(self):
        self.page_count = 0
        self.sections = []  # List of (title, page_number) tuples
        self.current_section = None

    def add_section(self, title):
        """Add a section title to track its starting page."""
        self.current_section = title
        self.sections.append((title, self.page_count + 1))

    def on_page(self, canvas, doc):
        """Called on each page to increment the page counter."""
        self.page_count += 1

def build_pdf_report(report_output_dir: Path, band_powers: dict = None, instability_indices: dict = None, source_localization: dict = None, vigilance_plots: dict = None, channels: list = None):
    """
    Builds a PDF report from EEG analysis results with enhanced styling and additional sections.

    Args:
        report_output_dir (Path): Directory containing the EEG analysis results, including:
            - clinical_interpretations.txt
            - pyramid_mappings.txt
            - topomaps directory
            - per_site_plots directory
        band_powers (dict, optional): Dictionary containing band powers for EO and EC conditions
        instability_indices (dict, optional): Dictionary containing instability indices for EO and EC conditions
        source_localization (dict, optional): Dictionary containing source localization results
        vigilance_plots (dict, optional): Dictionary containing paths to vigilance hypnogram and strip plots
            e.g., {'EO': {'hypnogram': Path, 'strip': Path}, 'EC': {'hypnogram': Path, 'strip': Path}}
        channels (list, optional): List of channel names to pair with instability indices
    """
    pdf_output_path = report_output_dir / "clinical_report.pdf"
    clinical_interpretations_path = report_output_dir / "clinical_interpretations.txt"
    pyramid_mappings_path = report_output_dir / "pyramid_mappings.txt"
    topomap_dir = report_output_dir / "topomaps"
    site_plot_dir = report_output_dir / "per_site_plots"

    # First Pass: Build the content to determine page numbers
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

    elements = []
    page_counter = PageCounter()

    def add_section(title, content_elements):
        page_counter.add_section(title)
        elements.append(Paragraph(title, styles['SectionHeading']))
        elements.extend(content_elements)
        elements.append(PageBreak())

    # Cover Page with Logo and Timestamp
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

    # Summary Section
    if pyramid_mappings_path.exists():
        content = []
        with open(pyramid_mappings_path, "r", encoding="utf-8") as f:
            mappings = f.readlines()
        summary_line = next((line for line in mappings if "Suggested Level" in line), "No summary available.")
        content.append(Paragraph(summary_line.strip(), styles["NormalText"]))
        content.append(Paragraph("This report includes detailed EEG analysis, including band powers, instability indices, and vigilance states.", styles["NormalText"]))
        add_section("Summary", content)

    # Placeholder for TOC (to be filled in the second pass)
    toc_placeholder = len(elements)
    elements.append(Paragraph("Table of Contents", styles['SectionHeading']))
    elements.append(Spacer(1, 0.2 * inch))
    elements.append(PageBreak())

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

    # Band Powers
    if band_powers:
        for condition in ["EO", "EC"]:
            if condition in band_powers:
                content = []
                content.append(Paragraph(f"Band Powers ({condition})", styles['SectionHeading']))
                content.append(Spacer(1, 0.2 * inch))
                data = [["Channel", "Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]]
                for channel, powers in band_powers[condition].items():
                    row = [channel]
                    for band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]:
                        row.append(f"{powers.get(band, 0):.2f} µV²")
                    data.append(row)
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor("#D3D3D3")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#000000")),
                    ('BACKGROUND', (0, 1), (-1, -1), HexColor("#F5F5F5")),
                ]))
                content.append(table)
                add_section(f"Band Powers ({condition})", content)

    # Instability Indices
    if instability_indices:
        for condition in ["EO", "EC"]:
            if condition in instability_indices:
                for band in ["Delta", "Theta", "Alpha", "SMR", "Beta", "HighBeta"]:
                    content = []
                    content.append(Paragraph(f"Instability Index ({band}, {condition})", styles['SectionHeading']))
                    content.append(Spacer(1, 0.2 * inch))
                    data = [["Channel", "Variance (µV²)"]]
                    for channel, variance in instability_indices[condition][band].items():
                        data.append([channel, f"{variance:.2f}"])
                    table = Table(data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#D3D3D3")),
                        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor("#FFFFFF")),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('GRID', (0, 0), (-1, -1), 0.5, HexColor("#000000")),
                        ('BACKGROUND', (0, 1), (-1, -1), HexColor("#F5F5F5")),
                    ]))
                    content.append(table)
                    add_section(f"Instability Index ({band}, {condition})", content)

    # Vigilance Plots (Hypnogram and Strip)
    if vigilance_plots:
        for condition in ["EO", "EC"]:
            if condition in vigilance_plots:
                content = []
                content.append(Paragraph(f"Vigilance Analysis ({condition})", styles['SectionHeading']))
                content.append(Spacer(1, 0.2 * inch))
                hypnogram_path = vigilance_plots[condition]["hypnogram"]
                if hypnogram_path.exists():
                    content.append(Paragraph(f"Vigilance Hypnogram ({condition})", styles['Heading3']))
                    content.append(Image(str(hypnogram_path), width=6.5 * inch, height=3.5 * inch))
                    content.append(Spacer(1, 0.2 * inch))
                strip_path = vigilance_plots[condition]["strip"]
                if strip_path.exists():
                    content.append(Paragraph(f"Vigilance Strip ({condition})", styles['Heading3']))
                    content.append(Image(str(strip_path), width=6.5 * inch, height=1.5 * inch))
                    content.append(Spacer(1, 0.2 * inch))
                add_section(f"Vigilance Analysis ({condition})", content)

    # Topomaps
    if topomap_dir.exists():
        content = []
        content.append(Paragraph("Topographic Maps (Absolute & Relative Power)", styles['SectionHeading']))
        content.append(Spacer(1, 0.2 * inch))
        for img_file in sorted(topomap_dir.glob("*.png")):
            content.append(Image(str(img_file), width=6.5 * inch, height=3.5 * inch))
            content.append(Spacer(1, 0.2 * inch))
        add_section("Topographic Maps", content)

    # Source Localization Results
    if source_localization:
        for condition in ["EO", "EC"]:
            if condition in source_localization:
                content = []
                content.append(Paragraph(f"Source Localization ({condition})", styles['SectionHeading']))
                content.append(Spacer(1, 0.2 * inch))
                for band, methods in source_localization[condition].items():
                    content.append(Paragraph(f"Band: {band}", styles['Heading3']))
                    for method, img_path in methods.items():
                        img_full_path = report_output_dir / img_path
                        if img_full_path.exists():
                            content.append(Paragraph(f"Method: {method}", styles['NormalText']))
                            content.append(Image(str(img_full_path), width=6.5 * inch, height=3.5 * inch))
                            content.append(Spacer(1, 0.2 * inch))
                add_section(f"Source Localization ({condition})", content)

    # Site Plots (Selective)
    selected_sites = {"O1", "O2", "PZ", "CZ", "FZ", "F3", "F4", "T3", "T4"}
    for site in selected_sites:
        site_path = site_plot_dir / site
        if not site_path.exists():
            continue
        content = []
        content.append(Paragraph(f"Site: {site}", styles['SectionHeading']))
        for plot_type in ["PSD_Overlay", "Waveform_Overlay", "Difference"]:
            plot_dir = site_path / plot_type
            if plot_dir.exists():
                for img in sorted(plot_dir.glob("*.png")):
                    content.append(Image(str(img), width=6.5 * inch, height=3.5 * inch))
                    content.append(Spacer(1, 0.1 * inch))
        add_section(f"Site: {site}", content)

    # First Pass: Determine page numbers
    temp_doc = SimpleDocTemplate(
        str(pdf_output_path) + ".temp",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    temp_doc.build(elements, onFirstPage=page_counter.on_page, onLaterPages=page_counter.on_page)

    # Second Pass: Rebuild the document with the TOC
    elements = elements[:toc_placeholder]  # Keep elements up to the TOC placeholder
    toc_content = []
    for title, page in page_counter.sections:
        toc_content.append([Paragraph(f"{title}", styles['NormalText']), Paragraph(f"Page {page}", styles['NormalText'])])
    toc_table = Table(toc_content)
    toc_table.setStyle(TableStyle([
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, black),  # Use black directly
    ]))
    elements.append(toc_table)
    elements.append(PageBreak())
    elements.extend(elements[toc_placeholder + 3:])  # Skip the old TOC elements

    # Final build with page numbers
    def add_page_numbers(canvas, doc):
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(HexColor("#4A4A4A"))
        canvas.drawRightString(7.5 * inch, 0.5 * inch, text)

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
    doc.build(elements, onFirstPage=add_page_numbers, onLaterPages=add_page_numbers)
    print(f"PDF report generated at: {pdf_output_path}")
