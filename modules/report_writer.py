# report_writer.py

def format_phenotype_section(result):
    lines = []
    lines.append("<h2>EEG Phenotype Classification</h2>")
    lines.append(f"<p><strong>Best Match:</strong> {result['best_match']} ({result['confidence']*100:.1f}% confidence)</p>")

    if result.get("explanations"):
        lines.append("<ul>")
        for exp in result['explanations']:
            lines.append(f"<li>{exp}</li>")
        lines.append("</ul>")

    lines.append("<h3>Recommendations</h3>")
    if result['recommendations']:
        lines.append("<ul>")
        for rec in result['recommendations']:
            lines.append(f"<li>{rec}</li>")
        lines.append("</ul>")
    else:
        lines.append("<p>No specific protocol recommendations.</p>")

    lines.append("<h3>Z-score Summary</h3>")
    lines.append("<table border='1' cellpadding='4'><tr><th>Metric</th><th>Value</th></tr>")
    for k, v in result['zscore_summary'].items():
        lines.append(f"<tr><td>{k}</td><td>{v:.2f}</td></tr>")
    lines.append("</table>")

    return "\n".join(lines)


def write_html_report(output_path, base_html, phenotype_html):
    report_html = base_html.replace("<!--PHENOTYPE_SECTION-->", phenotype_html)
    with open(output_path, "w") as f:
        f.write(report_html)
    return output_path
