#!/bin/bash
################################################################################
# MASTER ANALYSIS SCRIPT
# Runs all diagnostic and analysis scripts in sequence
################################################################################

echo "================================================================================"
echo "🔍 SHIFAMIND302 PHASE 1 - COMPREHENSIVE FAILURE ANALYSIS"
echo "================================================================================"
echo ""
echo "This script will run all analysis and diagnostic tools to investigate"
echo "why ShifaMind302 Phase 1 failed (Macro F1: 0.28 → 0.25)"
echo ""
echo "Scripts to run:"
echo "  1. GPU Diagnostic (verify GPU availability)"
echo "  2. Phase 1 Failure Analysis (evidence gathering)"
echo ""
echo "Estimated time: 5-10 minutes"
echo ""
read -p "Press Enter to continue..."

################################################################################
# STEP 1: GPU DIAGNOSTIC
################################################################################

echo ""
echo "================================================================================"
echo "STEP 1/2: GPU DIAGNOSTIC"
echo "================================================================================"
echo ""

if [ -f "gpu_diagnostic.py" ]; then
    echo "Running GPU diagnostic..."
    python3 gpu_diagnostic.py

    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠️  GPU diagnostic failed or GPU not available"
        echo "🔧 FIX: In Google Colab: Runtime → Change runtime type → GPU"
        echo ""
        read -p "Continue with remaining analysis? (y/n): " continue_analysis
        if [ "$continue_analysis" != "y" ]; then
            echo "Exiting..."
            exit 1
        fi
    else
        echo ""
        echo "✅ GPU diagnostic complete"
    fi
else
    echo "❌ gpu_diagnostic.py not found"
fi

echo ""
read -p "Press Enter to continue to Phase 1 analysis..."

################################################################################
# STEP 2: PHASE 1 FAILURE ANALYSIS
################################################################################

echo ""
echo "================================================================================"
echo "STEP 2/2: PHASE 1 FAILURE ANALYSIS (Evidence Gathering)"
echo "================================================================================"
echo ""

if [ -f "analyze_phase1_failures.py" ]; then
    echo "Running comprehensive failure analysis..."
    echo "This will analyze:"
    echo "  - GPU usage verification"
    echo "  - Top-50 ICD code verification"
    echo "  - Concept distribution analysis"
    echo "  - Concept-diagnosis correlations"
    echo "  - Configuration comparison"
    echo "  - Performance metrics"
    echo ""

    python3 analyze_phase1_failures.py

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Analysis complete"
    else
        echo ""
        echo "❌ Analysis script encountered errors"
    fi
else
    echo "❌ analyze_phase1_failures.py not found"
fi

################################################################################
# SUMMARY
################################################################################

echo ""
echo "================================================================================"
echo "📋 ANALYSIS COMPLETE"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  1. gpu_diagnostics.json"
echo "  2. /content/drive/MyDrive/ShifaMind/11_ShifaMind_v302/analysis_evidence/"
echo "     - detailed_findings.json"
echo "     - SUMMARY_REPORT.txt"
echo ""
echo "Next steps:"
echo "  1. Review SUMMARY_REPORT.txt for key findings"
echo "  2. Review detailed_findings.json for full evidence"
echo "  3. If GPU is available, run: shifamind302_phase1_FIXED.py"
echo ""
echo "================================================================================"
echo "For detailed instructions, see: PHASE1_FAILURE_ANALYSIS_README.md"
echo "================================================================================"
