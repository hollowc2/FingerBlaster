#!/usr/bin/env python3
"""Test script to debug Pulse startup."""

import sys
import traceback
import logging

# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data/pulse_debug.log')
    ]
)

logger = logging.getLogger("PulseDebug")

try:
    logger.info("Starting Pulse test...")
    logger.info("Importing pulse.gui.main...")
    from pulse.gui.main import run_pulse_app
    logger.info("Import successful!")

    logger.info("Calling run_pulse_app()...")
    run_pulse_app()
    logger.info("run_pulse_app() returned")

except Exception as e:
    logger.error(f"Exception occurred: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)
