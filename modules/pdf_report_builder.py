from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER
import os

def build_pdf_report(report_output_dir: Path):
    ...


def build_pdf_report(report_output_dir: Path):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.enums import TA_CENTER
    import os

    pdf_output_path = report_output_dir / "clinical_report.pdf"
    text_report_path = report_output_dir / "text" / "clinical_report.txt"
    topomap_dir = report_output_dir / "topomaps"
    site_plot_dir = report_output_dir / "site_plots"

    # PDF setup
    doc = SimpleDocTemplate(str(pdf_output_path), pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='CenteredTitle', fontSize=18, leading=22, alignment=TA_CENTER))
    elements = []

    # Cover Page
    elements.append(Paragraph("The Squiggle Interpreter", styles['CenteredTitle']))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph("Comprehensive EEG Clinical Report", styles['Title']))
    elements.append(PageBreak())

    # Text
    if text_report_path.exists():
        with open(text_report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        for line in report_text.split("\n"):
            elements.append(Paragraph(line.strip(), styles["Normal"]))
            elements.append(Spacer(1, 0.1 * inch))
        elements.append(PageBreak())

    # Topomaps
    if topomap_dir.exists():
        elements.append(Paragraph("Topographic Maps (Absolute & Relative Power)", styles['Heading2']))
        for img_file in sorted(topomap_dir.glob("*.png")):
            elements.append(Image(str(img_file), width=6.5 * inch, height=3.5 * inch))
            elements.append(Spacer(1, 0.2 * inch))
        elements.append(PageBreak())

    # Site plots (selective)
    selected_sites = {"O1", "O2", "PZ", "CZ", "FZ", "F3", "F4", "T3", "T4"}
    for site in selected_sites:
        site_path = site_plot_dir / site
        if not site_path.exists():
            continue
        elements.append(Paragraph(f"Site: {site}", styles['Heading2']))
        for plot_type in ["PSD_Overlay", "Waveform_Overlay", "Difference"]:
            plot_dir = site_path / plot_type
            if plot_dir.exists():
                for img in sorted(plot_dir.glob("*.png")):
                    elements.append(Image(str(img), width=6.5 * inch, height=3.5 * inch))
                    elements.append(Spacer(1, 0.1 * inch))
        elements.append(PageBreak())

    doc.build(elements)
