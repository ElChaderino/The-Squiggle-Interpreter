#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pdf_report_builder.py

This module provides functions to build HTML and PDF reports for EEG data analysis in The Squiggle Interpreter.
It uses Jinja2 for HTML templating and a PDF library (e.g., weasyprint) for PDF generation.
"""

import logging
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template
import os
from .report_writer import format_phenotype_section, write_html_report

logger = logging.getLogger(__name__)

def build_html_report(report_data, output_path):
    """
    Build an interactive HTML report using a Jinja2 template.
    
    Args:
        report_data (dict): Data to populate the HTML template.
        output_path (Path or str): Path to save the HTML report.
    
    Raises:
        Exception: If HTML rendering fails.
    """
    try:
        # Ensure Jinja2 is installed
        logger.debug("Initializing Jinja2 environment for HTML report...")
        # Set up the Jinja2 environment
        template_dir = Path(__file__).parent / 'templates'
        if not template_dir.exists():
            template_dir = Path(__file__).parent
        env = Environment(loader=FileSystemLoader(template_dir))
        
        # Load the template
        template_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Interactive EEG Report</title>
  <style>
    /* Dark mode styling */
    body {
      background-color: #222;
      color: #eee;
      font-family: Arial, sans-serif;
      text-align: center;
      margin: 0;
      padding: 20px;
    }
    select, option {
      padding: 5px;
      margin: 10px;
      background-color: #333;
      color: #eee;
      border: 1px solid #555;
    }
    img {
      margin: 10px;
      border: 2px solid #555;
      max-width: 90%;
    }
    .hypnogram-img {
      width: 100%;
      max-width: 800px;
      height: auto;
      border: 2px solid #666;
      background: repeating-linear-gradient(
        45deg,
        #333,
        #333 10px,
        #444 10px,
        #444 20px
      ); /* Striped background for hypnograms */
    }
    a {
      color: #66ccee;
    }
    .container {
      margin-bottom: 30px;
    }
    h2 {
      border-bottom: 2px solid #555;
      padding-bottom: 10px;
    }
    .section-title {
      margin-top: 40px;
      border-bottom: 2px solid #555;
      padding-bottom: 10px;
    }
    table {
      margin: 20px auto;
      border-collapse: collapse;
      background-color: #333;
    }
    th, td {
      padding: 8px;
      border: 1px solid #555;
    }
    th {
      background-color: #444;
    }
  </style>

  <script>
    // ---------------------------
    // 1) DATA FROM PYTHON
    // ---------------------------
    var globalTopomaps     = {{ global_topomaps | tojson }};
    var globalWaveforms    = {{ global_waveforms | tojson }};
    var coherence          = {{ coherence | tojson }};
    var globalERP          = {{ global_erp | tojson }};
    var zscore             = {{ zscore | tojson }};
    var variance           = {{ variance | tojson }};
    var tfr                = {{ tfr | tojson }};
    var ica                = {{ ica | tojson }};
    var sourceLocalization = {{ source_localization | tojson }};
    var hypnograms         = {{ hypnograms | tojson }};

    var siteList = {{ site_list | tojson }};
    var bandList = {{ band_list | tojson }};
    var siteDict = {{ site_dict | tojson }};
    var phenotype = {{ phenotype | default({}) | tojson }};

    // ---------------------------
    // 2) PATHS PASSED FROM PYTHON
    // ---------------------------
    var topomapsPath    = "{{ global_topomaps_path }}";  // Should be "topomaps"
    var waveformsPath   = "{{ global_waveforms_path }}";
    var coherencePath   = "{{ coherence_path }}";
    var erpPath         = "{{ global_erp_path }}";
    var tfrPath         = "{{ tfr_path }}";
    var icaPath         = "{{ ica_path }}";
    var sourcePath      = "{{ source_path }}";
    var sitesPath       = "{{ sites_path }}";
    var variancePath    = "topomaps";  // Points to topomaps/ folder

    // ---------------------------
    // 3) UTILITY FUNCTIONS
    // ---------------------------
    function populateDropdown(id, items) {
      var sel = document.getElementById(id);
      sel.innerHTML = "";
      items.forEach(function(item) {
        var opt = document.createElement("option");
        opt.value = item;
        opt.text = item;
        sel.appendChild(opt);
      });
    }

    // ---------------------------
    // GLOBAL TOPOMAPS
    // ---------------------------
    function populateGlobalTopomapDropdowns() {
      populateDropdown("globalTopoCond", ["EO", "EC"]);
      populateDropdown("globalTopoSelect", bandList);
      updateGlobalTopo();
    }
    function updateGlobalTopo() {
      var cond = document.getElementById("globalTopoCond").value;
      var band = document.getElementById("globalTopoSelect").value;
      var imgName = globalTopomaps[cond][band];
      document.getElementById("globalTopoImg").src = topomapsPath + "/" + cond + "/" + imgName;
    }

    // ---------------------------
    // GLOBAL WAVEFORMS
    // ---------------------------
    function populateGlobalWaveDropdowns() {
      var waveBands = Object.keys(globalWaveforms);
      populateDropdown("globalWaveSelect", waveBands);
      updateGlobalWave();
    }
    function updateGlobalWave() {
      var band = document.getElementById("globalWaveSelect").value;
      var imgName = globalWaveforms[band];
      document.getElementById("globalWaveImg").src = waveformsPath + "/EO/" + imgName;
    }

    // ---------------------------
    // GLOBAL COHERENCE
    // ---------------------------
    function populateGlobalCohDropdowns() {
      populateDropdown("globalCohCond", ["EO", "EC"]);
      var cohBands = coherence["EO"] ? Object.keys(coherence["EO"]) : bandList;
      populateDropdown("globalCohSelect", cohBands);
      updateGlobalCoh();
    }
    function updateGlobalCoh() {
      var cond = document.getElementById("globalCohCond").value;
      var band = document.getElementById("globalCohSelect").value;
      if (coherence[cond] && coherence[cond][band]) {
        var imgName = coherence[cond][band];
        document.getElementById("globalCohImg").src = coherencePath + "/" + cond + "/" + imgName;
      } else {
        document.getElementById("globalCohImg").src = "";
      }
    }

    // ---------------------------
    // GLOBAL ERP
    // ---------------------------
    function populateGlobalERPDropdowns() {
      populateDropdown("globalERPCond", ["EO", "EC"]);
      updateGlobalERP();
    }
    function updateGlobalERP() {
      var cond = document.getElementById("globalERPCond").value;
      var imgName = globalERP[cond];
      document.getElementById("erpImg").src = erpPath + "/" + imgName;
    }

    // ---------------------------
    // Z-SCORE MAPS
    // ---------------------------
    function populateZscoreDropdowns() {
      populateDropdown("zscoreCond", ["EO", "EC"]);
      populateDropdown("zscoreBand", bandList);
      updateZscoreImage();
    }
    function updateZscoreImage() {
      var cond = document.getElementById("zscoreCond").value;
      var band = document.getElementById("zscoreBand").value;
      if (zscore[cond] && zscore[cond][band]) {
        var imgName = zscore[cond][band];
        document.getElementById("zscoreImg").src = "zscore/" + cond + "/" + imgName;
      } else {
        document.getElementById("zscoreImg").src = "";
      }
    }

    // ---------------------------
    // VARIANCE TOPOMAPS
    // ---------------------------
    function populateVarianceDropdowns() {
      populateDropdown("varianceCond", ["EO", "EC"]);
      populateDropdown("varianceBand", bandList);
      updateVarianceImage();
    }
    function updateVarianceImage() {
      var cond = document.getElementById("varianceCond").value;
      var band = document.getElementById("varianceBand").value;
      var capitalizedBand = band.charAt(0).toUpperCase() + band.slice(1);
      var imgName = "topomap_" + capitalizedBand + "_" + cond + ".png";
      document.getElementById("varianceImg").src = variancePath + "/" + imgName;
    }

    // ---------------------------
    // TFR
    // ---------------------------
    function populateTfrDropdowns() {
      populateDropdown("tfrCond", ["EO", "EC"]);
      populateDropdown("tfrBand", bandList);
      updateTfrImage();
    }
    function updateTfrImage() {
      var cond = document.getElementById("tfrCond").value;
      var band = document.getElementById("tfrBand").value;
      if (tfr[cond] && tfr[cond][band]) {
        var imgName = tfr[cond][band];
        document.getElementById("tfrImg").src = tfrPath + "/" + cond + "/" + imgName;
      } else {
        document.getElementById("tfrImg").src = "";
      }
    }

    // ---------------------------
    // ICA
    // ---------------------------
    function populateIcaDropdowns() {
      populateDropdown("icaCond", ["EO", "EC"]);
      updateIcaImage();
    }
    function updateIcaImage() {
      var cond = document.getElementById("icaCond").value;
      if (ica[cond]) {
        var imgName = ica[cond];
        document.getElementById("icaImg").src = icaPath + "/" + cond + "/" + imgName;
      } else {
        document.getElementById("icaImg").src = "";
      }
    }

    // ---------------------------
    // SOURCE LOCALIZATION
    // ---------------------------
    function populateSourceDropdowns() {
      populateDropdown("sourceCond", ["EO", "EC"]);
      populateDropdown("sourceBand", bandList);
      var knownMethods = ["LORETA", "sLORETA", "eLORETA"];
      populateDropdown("sourceMethod", knownMethods);
      updateSourceImage();
    }
    function updateSourceImage() {
      var cond   = document.getElementById("sourceCond").value;
      var band   = document.getElementById("sourceBand").value;
      var method = document.getElementById("sourceMethod").value;
      if (sourceLocalization[cond] && sourceLocalization[cond][band] && sourceLocalization[cond][band][method]) {
        var imgName = sourceLocalization[cond][band][method];
        var fullPath = sourcePath + "/" + imgName;
        console.log("Source Localization Image URL:", fullPath);
        document.getElementById("sourceImg").src = fullPath;
      } else {
        document.getElementById("sourceImg").src = "";
        console.log("No source localization image found for", cond, band, method);
      }
    }

    // ---------------------------
    // PER-SITE ANALYSIS
    // ---------------------------
    function populateSiteDropdowns() {
      populateDropdown("siteSelect", siteList);
      populateDropdown("bandSelect", bandList);
      populateDropdown("siteDiffSelect", bandList);
      updateSiteImages();
    }
    function updateSiteImages() {
      var site = document.getElementById("siteSelect").value;
      var band = document.getElementById("bandSelect").value;
      var diffBand = document.getElementById("siteDiffSelect").value;

      if (siteDict[site] && siteDict[site][band]) {
        document.getElementById("sitePsdImg").src = siteDict[site][band]["psd"];
        document.getElementById("siteWaveImg").src = siteDict[site][band]["wave"];
      } else {
        document.getElementById("sitePsdImg").src = "";
        document.getElementById("siteWaveImg").src = "";
      }

      if (siteDict[site] && siteDict[site][diffBand]) {
        document.getElementById("siteDiffTopoImg").src = siteDict[site][diffBand]["diff_topo"];
        document.getElementById("siteDiffBarImg").src = siteDict[site][diffBand]["diff_bar"];
      } else {
        document.getElementById("siteDiffTopoImg").src = "";
        document.getElementById("siteDiffBarImg").src = "";
      }
    }

    // ---------------------------
    // PER-SITE HYPNOGRAMS
    // ---------------------------
    function populateHypnogramDropdowns() {
      console.log("Populating hypnogram dropdowns with hardcoded sites: ['O1', 'O2', 'PZ']");
      populateDropdown("hypnogramSite", ["O1", "O2", "PZ"]); // Hardcode the sites
      populateDropdown("hypnogramCond", ["EO", "EC", "EO_CSD", "EC_CSD"]);
      updateHypnogramImage();
    }
    function updateHypnogramImage() {
      var site = document.getElementById("hypnogramSite").value;
      var cond = document.getElementById("hypnogramCond").value;
      console.log("Updating hypnogram image. Condition:", cond, "Site:", site);
      var imgName = "vigilance_hypnogram_" + cond + "_" + site + ".png";
      console.log("Loading hypnogram image: " + imgName);
      var imgElement = document.getElementById("hypnogramImg");
      imgElement.src = imgName;
      imgElement.onerror = function() {
        console.error("Failed to load hypnogram image: " + imgName);
        imgElement.src = "";
      };
    }

    // ---------------------------
    // PER-SITE STRIP CHARTS
    // ---------------------------
    function populateStripChartDropdowns() {
      console.log("Populating strip chart dropdowns with hardcoded sites: ['O1', 'O2', 'PZ']");
      populateDropdown("stripChartSite", ["O1", "O2", "PZ"]); // Hardcode the sites
      populateDropdown("stripChartCond", ["EO", "EC", "EO_CSD", "EC_CSD"]);
      updateStripChartImage();
    }
    function updateStripChartImage() {
      var site = document.getElementById("stripChartSite").value;
      var cond = document.getElementById("stripChartCond").value;
      console.log("Updating strip chart image. Condition:", cond, "Site:", site);
      var imgName = "vigilance_strip_" + cond + "_" + site + ".png";
      console.log("Loading strip chart image: " + imgName);
      var imgElement = document.getElementById("stripChartImg");
      imgElement.src = imgName;
      imgElement.onerror = function() {
        console.error("Failed to load strip chart image: " + imgName);
        imgElement.src = "";
      };
    }

    // ---------------------------
    // ON WINDOW LOAD
    // ---------------------------
    window.onload = function() {
      console.log("Site list from Python:", siteList);
      populateGlobalTopomapDropdowns();
      populateGlobalWaveDropdowns();
      populateGlobalCohDropdowns();
      populateGlobalERPDropdowns();
      populateZscoreDropdowns();
      populateVarianceDropdowns();
      populateTfrDropdowns();
      populateIcaDropdowns();
      populateSourceDropdowns();
      populateSiteDropdowns();
      populateHypnogramDropdowns();
      populateStripChartDropdowns();
      displayPhenotype();
    };

    function displayPhenotype() {
      const pheno = JSON.stringify(phenotype, null, 2);
      document.getElementById("phenotypeDisplay").innerText = pheno || "No phenotype data found.";
    }
  </script>
</head>
<body>
  <h1>Interactive EEG Report</h1>

  <!-- Global Topomaps Section -->
  <div class="container">
    <h2>Global Topomaps</h2>
    <label for="globalTopoCond">Condition:</label>
    <select id="globalTopoCond" onchange="updateGlobalTopo()"></select>
    <label for="globalTopoSelect">Band:</label>
    <select id="globalTopoSelect" onchange="updateGlobalTopo()"></select>
    <br>
    <img id="globalTopoImg" src="" alt="Global Topomap">
  </div>

  <!-- Global Waveforms Section -->
  <div class="container">
    <h2>Global Waveforms</h2>
    <label for="globalWaveSelect">Band:</label>
    <select id="globalWaveSelect" onchange="updateGlobalWave()"></select>
    <br>
    <img id="globalWaveImg" src="" alt="Global Waveform Grid">
  </div>

  <!-- Global Coherence Section -->
  <div class="container">
    <h2>Global Coherence</h2>
    <label for="globalCohCond">Condition:</label>
    <select id="globalCohCond" onchange="updateGlobalCoh()"></select>
    <label for="globalCohSelect">Band:</label>
    <select id="globalCohSelect" onchange="updateGlobalCoh()"></select>
    <br>
    <img id="globalCohImg" src="" alt="Global Coherence Matrix">
  </div>

  <!-- Global ERP Section -->
  <div class="container">
    <h2>Global ERP</h2>
    <label for="globalERPCond">Condition:</label>
    <select id="globalERPCond" onchange="updateGlobalERP()"></select>
    <br>
    <img id="erpImg" src="" alt="Global ERP Plot">
  </div>

  <!-- Z-Score Maps Section -->
  <div class="container">
    <h2 class="section-title">Z-Score Maps</h2>
    <label for="zscoreCond">Condition:</label>
    <select id="zscoreCond" onchange="updateZscoreImage()"></select>
    <label for="zscoreBand">Band:</label>
    <select id="zscoreBand" onchange="updateZscoreImage()"></select>
    <br>
    <img id="zscoreImg" src="" alt="Z-Score Topomap">
  </div>

  <!-- Variance Topomaps Section -->
  <div class="container">
    <h2 class="section-title">Variance Topomaps</h2>
    <label for="varianceCond">Condition:</label>
    <select id="varianceCond" onchange="updateVarianceImage()"></select>
    <label for="varianceBand">Band:</label>
    <select id="varianceBand" onchange="updateVarianceImage()"></select>
    <br>
    <img id="varianceImg" src="" alt="Variance Topomap">
  </div>

  <!-- TFR Section -->
  <div class="container">
    <h2 class="section-title">Time-Frequency Representation (TFR)</h2>
    <label for="tfrCond">Condition:</label>
    <select id="tfrCond" onchange="updateTfrImage()"></select>
    <label for="tfrBand">Band:</label>
    <select id="tfrBand" onchange="updateTfrImage()"></select>
    <br>
    <img id="tfrImg" src="" alt="TFR Plot">
  </div>

  <!-- ICA Section -->
  <div class="container">
    <h2 class="section-title">ICA Components</h2>
    <label for="icaCond">Condition:</label>
    <select id="icaCond" onchange="updateIcaImage()"></select>
    <br>
    <img id="icaImg" src="" alt="ICA Components">
  </div>

  <!-- Source Localization Section -->
  <div class="container">
    <h2 class="section-title">Source Localization</h2>
    <label for="sourceCond">Condition:</label>
    <select id="sourceCond" onchange="updateSourceImage()"></select>
    <label for="sourceBand">Band:</label>
    <select id="sourceBand" onchange="updateSourceImage()"></select>
    <label for="sourceMethod">Method:</label>
    <select id="sourceMethod" onchange="updateSourceImage()"></select>
    <br>
    <img id="sourceImg" src="" alt="Source Localization">
  </div>

  <!-- Per-Site Analysis Section -->
  <div class="container">
    <h2>Per-Site Analysis</h2>
    <label for="siteSelect">Site:</label>
    <select id="siteSelect" onchange="updateSiteImages()"></select>
    <label for="bandSelect">Band (PSD/Waveform):</label>
    <select id="bandSelect" onchange="updateSiteImages()"></select>
    <br>
    <h3>Site PSD Plot</h3>
    <img id="sitePsdImg" src="" alt="Site PSD Plot">
    <br>
    <h3>Site Waveform Plot</h3>
    <img id="siteWaveImg" src="" alt="Site Waveform Plot">
    <br>
    <label for="siteDiffSelect">Difference Band:</label>
    <select id="siteDiffSelect" onchange="updateSiteImages()"></select>
    <br>
    <h3>Site Difference Topomap</h3>
    <img id="siteDiffTopoImg" src="" alt="Site Difference Topomap">
    <br>
    <h3>Site Difference Bar Graph</h3>
    <img id="siteDiffBarImg" src="" alt="Site Difference Bar Graph">
  </div>

  <!-- Per-Site Hypnograms Section -->
  <div class="container">
    <h2 class="section-title">Per-Site Vigilance Hypnograms</h2>
    <label for="hypnogramSite">Site:</label>
    <select id="hypnogramSite" onchange="updateHypnogramImage()"></select>
    <label for="hypnogramCond">Condition:</label>
    <select id="hypnogramCond" onchange="updateHypnogramImage()"></select>
    <br>
    <img id="hypnogramImg" class="hypnogram-img" src="" alt="Vigilance Hypnogram">
  </div>

  <!-- Per-Site Strip Charts Section -->
  <div class="container">
    <h2 class="section-title">Per-Site Vigilance Strip Charts</h2>
    <label for="stripChartSite">Site:</label>
    <select id="stripChartSite" onchange="updateStripChartImage()"></select>
    <label for="stripChartCond">Condition:</label>
    <select id="stripChartCond" onchange="updateStripChartImage()"></select>
    <br>
    <img id="stripChartImg" class="hypnogram-img" src="" alt="Vigilance Strip Chart">
  </div>

  <!-- Phenotype Classification Section -->
  <div class="container">
    <!--PHENOTYPE_SECTION-->
  </div>

</body>
</html>
"""
        template = env.from_string(template_content)

        # Render the base HTML template with the report data (excluding phenotype for now)
        logger.debug("Rendering base HTML template with report data...")
        base_html = template.render(**report_data)

        # Generate the phenotype section
        phenotype_data = report_data.get("phenotype", {})
        logger.debug(f"Phenotype data: {phenotype_data}")
        if not phenotype_data:
            logger.warning("Phenotype data is empty; generating a placeholder phenotype section.")
            phenotype_data = {
                "best_match": "Unknown",
                "confidence": 0.0,
                "explanations": ["No phenotype data available."],
                "recommendations": [],
                "zscore_summary": {}
            }
        phenotype_html = format_phenotype_section(phenotype_data)
        logger.debug(f"Phenotype HTML section generated: {phenotype_html[:200]}...")

        # Combine the base HTML with the phenotype section and write to file
        logger.debug(f"Writing final HTML report to {output_path}...")
        write_html_report(output_path, base_html, phenotype_html)
        logger.debug(f"HTML report written to {output_path}")

    except Exception as e:
        logger.error(f"Failed to build HTML report: {e}")
        raise

def build_pdf_report(report_output_dir, band_powers, instability_indices, source_localization, vigilance_plots, channels):
    """
    Build a PDF report summarizing key EEG findings.
    
    Args:
        report_output_dir (Path or str): Directory to save the PDF report.
        band_powers (dict): Band power data.
        instability_indices (dict): Instability indices.
        source_localization (dict): Source localization results.
        vigilance_plots (dict): Vigilance plot paths.
        channels (list): List of channels.
    
    Note: This is a placeholder implementation. Use a library like weasyprint to convert HTML to PDF.
    """
    try:
        logger.debug("Building PDF report (placeholder implementation)...")
        output_path = Path(report_output_dir) / "clinical_report.pdf"
        with open(output_path, "w") as f:
            f.write("PDF report placeholder\n")
        logger.debug(f"PDF report written to {output_path}")
    except Exception as e:
        logger.error(f"Failed to build PDF report: {e}")
        raise
