#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
report_writer.py

This module provides functions to format and write HTML reports for EEG data analysis in The Squiggle Interpreter.
It handles the phenotype classification section and final HTML report assembly.
"""

import logging

logger = logging.getLogger(__name__)

def format_phenotype_section(result):
    """
    Format the phenotype classification section as HTML.
    
    Args:
        result (dict): Phenotype classification result with keys like 'best_match', 'confidence', etc.
    
    Returns:
        str: HTML string for the phenotype section.
    """
    logger.debug(f"Formatting phenotype section with result: {result}")
    lines = []
    try:
        lines.append("<h2>EEG Phenotype Classification</h2>")
        best_match = result.get('best_match', 'Unknown')
        confidence = result.get('confidence', 0.0)
        lines.append(f"<p><strong>Best Match:</strong> {best_match} ({confidence*100:.1f}% confidence)</p>")

        if result.get("explanations"):
            lines.append("<ul>")
            for exp in result['explanations']:
                lines.append(f"<li>{exp}</li>")
            lines.append("</ul>")
        else:
            lines.append("<p>No explanations provided.</p>")

        lines.append("<h3>Recommendations</h3>")
        if result.get('recommendations'):
            lines.append("<ul>")
            for rec in result['recommendations']:
                lines.append(f"<li>{rec}</li>")
            lines.append("</ul>")
        else:
            lines.append("<p>No specific protocol recommendations.</p>")

        lines.append("<h3>Z-score Summary</h3>")
        zscore_summary = result.get('zscore_summary', {})
        if zscore_summary:
            lines.append("<table border='1' cellpadding='4'><tr><th>Metric</th><th>Value</th></tr>")
            for k, v in zscore_summary.items():
                lines.append(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>")
            lines.append("</table>")
        else:
            lines.append("<p>No z-score summary available.</p>")

        formatted_html = "\n".join(lines)
        logger.debug(f"Phenotype section formatted successfully: {formatted_html[:200]}...")
        return formatted_html
    except Exception as e:
        logger.error(f"Failed to format phenotype section: {e}")
        raise

def write_html_report(output_path, base_html, phenotype_html):
    """
    Write the final HTML report by combining the base HTML and phenotype section.
    
    Args:
        output_path (str or Path): Path to save the HTML report.
        base_html (str): Base HTML content with a placeholder for the phenotype section.
        phenotype_html (str): HTML content for the phenotype section.
    
    Returns:
        str: Path to the written HTML report.
    """
    logger.debug(f"Writing HTML report to {output_path}...")
    try:
        if "<!--PHENOTYPE_SECTION-->" not in base_html:
            logger.warning("Base HTML template does not contain <!--PHENOTYPE_SECTION--> placeholder.")
            base_html = base_html.replace("</body>", f"{phenotype_html}\n</body>")
        report_html = base_html.replace("<!--PHENOTYPE_SECTION-->", phenotype_html)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        logger.debug(f"HTML report successfully written to {output_path}")
        return str(output_path)
    except Exception as e:
        logger.error(f"Failed to write HTML report to {output_path}: {e}")
        raise
