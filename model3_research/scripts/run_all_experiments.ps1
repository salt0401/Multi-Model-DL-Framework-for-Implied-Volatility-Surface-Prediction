# Run All V3 Experiments in Background
# This script launches 3 separate PowerShell instances in the background.
# Each process will execute the train_models.py script with distinct arguments
# and automatically write out logs into the `logs/` directory.

$scriptsDir = "C:\Users\owen3\OneDrive\Desktop\Project\smart-data-analysis-main\model3_research\scripts"

Write-Host "Dispatching TFT CPR training..."
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoExit -Command `"cd '$scriptsDir'; python train_models.py --model tft --optimizer cpr --dtype float32`""

Write-Host "Dispatching TFT AdamW training..."
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoExit -Command `"cd '$scriptsDir'; python train_models.py --model tft --optimizer adamw --dtype float32`""

Write-Host "Dispatching Baseline CWD training..."
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoExit -Command `"cd '$scriptsDir'; python train_models.py --model baseline --optimizer cwd --dtype float64`""

Write-Host "All three jobs launched! Check the logs/ folder for execution progress."
