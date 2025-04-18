#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import signal
import sys
import logging
from pathlib import Path
from modules.config import parse_arguments
from modules.io_utils import find_subject_edf_files
from modules.data_to_csv import process_edf_to_csv
from modules.pipeline import process_subject
from modules.live_display import sigint_handler

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    Return the current working directory as the project root.
    
    Returns:
        Path: Project root directory.
    """
    project_root = Path.cwd()
    logger.info(f"Using current directory as project root: {project_root}")
    return project_root

def setup_logging(project_dir: str):
    """
    Configure logging to console and file.
    
    Args:
        project_dir (str): Project directory path.
    """
    log_file = os.path.join(project_dir, "squiggle_log.txt")
    logging.getLogger('').handlers = []
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )

def main():
    project_dir = get_project_root()
    setup_logging(project_dir)
    config = parse_arguments()
    config['phenotype'] = False  # Disable phenotype reports
    config['csv'] = False  # Disable CSV output unless explicitly needed
    if config.get('csv') and config.get('edf_path') and config.get('output_csv'):
        try:
            logger.info(f"Processing EDF to CSV: {config['edf_path']} -> {config['output_csv']}")
            process_edf_to_csv(config['edf_path'], config['epoch_length'], config['output_csv'])
            logger.info(f"CSV export completed: {config['output_csv']}")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to process CSV export: {e}")
            sys.exit(1)
    
    subject_edf_groups = find_subject_edf_files(project_dir)
    if not subject_edf_groups:
        logger.warning("No EDF files found in the project directory. Exiting.")
        sys.exit(0)
    
    logger.info(f"Found subject EDF files: {subject_edf_groups}")
    for subject, files in subject_edf_groups.items():
        logger.info(f"Processing subject {subject}...")
        try:
            process_subject(subject, files, project_dir, config)
            logger.info(f"Completed processing for subject {subject}")
        except Exception as e:
            logger.error(f"Failed to process subject {subject}: {e}")
            continue

if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)
    main()
