import os
from jinja2 import Environment, FileSystemLoader
import logging

logger = logging.getLogger(__name__)

def build_html_report(report_data, output_path):
    """
    Render an interactive HTML report using a Jinja2 template.
    
    Args:
        report_data (dict): Data to populate the HTML template.
        output_path (str): Path to save the HTML report.
    """
    try:
        # Set up the Jinja2 environment
        template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "templates"))
        logger.info(f"Attempting to load template from directory: {template_dir}")
        
        if not os.path.exists(template_dir):
            logger.error(f"Template directory not found: {template_dir}")
            raise FileNotFoundError(f"Template directory not found: {template_dir}")

        template_file = "report_template.html"
        template_path = os.path.join(template_dir, template_file)
        if not os.path.exists(template_path):
            logger.error(f"Template file not found: {template_path}")
            raise FileNotFoundError(f"Template file not found: {template_path}")

        logger.debug(f"Loading template from {template_path}")
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(template_file)

        # Render the template with the report data
        logger.debug("Rendering HTML template with report data...")
        rendered_html = template.render(**report_data)

        # Write the rendered HTML to the output file
        logger.debug(f"Writing HTML report to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        logger.info(f"Interactive HTML report generated at: {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        raise
