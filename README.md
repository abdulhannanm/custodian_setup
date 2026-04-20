# Custodian Setup Guide

Custodian is an HPC job analysis agent that runs inside your terminal using OpenCode. It ingests GPU metrics data files, performs K-means cluster analysis to identify execution phases, generates roofline and time-series plots, and produces structured performance diagnoses. This guide walks through everything needed to get it running on your machine.

---

## Prerequisites

Before starting, confirm that the following are installed and available on your PATH.

### OpenCode

OpenCode is the terminal AI agent runtime that Custodian runs on top of. Install it by following the official instructions at https://opencode.ai/docs.

Verify the installation:

```
opencode --version
```

### Bun

The tool files use `Bun.$` for shell execution, so Bun must be available at runtime regardless of how OpenCode was installed. If you installed OpenCode via `bun install -g opencode-ai`, Bun is already present. If you installed OpenCode via npm, Homebrew, a binary, or any other method, you need to install Bun separately from https://bun.sh.

To check whether Bun is already available:

```
bun --version
```

### Python 3.10 or later

The Python backend performs all data analysis. Verify that Python 3 is available:

```
python3 --version
```

### A CBorg API Key

The agent uses the CBorg API to access Claude Sonnet and the vision model. You must have a valid `CBORG_API_KEY`. If you do not have one, you can create one if you are a NERSC employee at https://cborg.lbl.gov/

---

## Repository Structure

After placing this directory on your machine, its layout should look like this:

```
custodian_setup/
  opencode.json
  requirements.txt
  README.md
  app/
    app.py
    generate_roofline_heatmap.py
    __init__.py
    models/
      dataObject.py
  prompts/
    custodian_agent.txt
  .opencode/
    package.json
    tools/
      get_cluster_analysis.ts
      generate_line_plot.ts
      generate_radar_plot.ts
      generate_roofline_heatmap.ts
      vision.ts
```

---

## Step 1: Set Up the Python Environment

The tool files call into the Python backend using a virtual environment. You can place this virtual environment anywhere on your machine. The path you choose will be exported as an environment variable in a later step.

Create a virtual environment inside the `~/custodian_setup` or whatever location you want. The example below uses `custodian-venv` inside `custodian_setup/`, but you may use any absolute path.

```
cd custodian_setup 
python3 -m venv custodian-venv
```

Activate the environment:

```
source custodian-venv/bin/activate
```

Install the Python dependencies:

```
pip install -r requirements.txt
```

Deactivate when done:

``
deactivate
```

---

## Step 2: Set Environment Variables

Three environment variables must be set before running the agent. Add these to your shell profile (e.g., `~/.zshrc` or `~/.bashrc`) so they persist across sessions.

### CBORG_API_KEY

Your CBorg API key. The agent uses this for both the language model and the vision model.

```
export CBORG_API_KEY=your_key_here
```

### CUSTODIAN_VENV_PATH

The absolute path to the `python3` binary inside the virtual environment you created in Step 1.

```
export CUSTODIAN_VENV_PATH=path_to_venv
```

### CUSTODIAN_APP_PATH

The absolute path to the `app/` directory inside this repository. This tells the tools where to find `app.py`.

```
export CUSTODIAN_APP_PATH=/absolute/path/to/custodian_setup/app
```

After editing your shell profile, reload it:

```
source ~/.zshrc
```

Or for bash:

```
source ~/.bashrc
```

---

## Step 3: Install the OpenCode Plugin Dependencies

The `.opencode/` directory contains the tool definitions that the agent uses. These tools depend on the `@opencode-ai/plugin` package, which must be installed via Bun.

Navigate into the `.opencode/` directory and install:

```
cd custodian_setup/.opencode
bun install
```

This creates a `node_modules/` folder inside `.opencode/`. Do not move or delete it.

---

## Step 4: Verify the Configuration

Open `opencode.json` and confirm that the `baseURL` and model settings match your CBorg endpoint. The file should look like this:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "cborg/anthropic/claude-sonnet-high",
  "provider": {
    "cborg": {
      "name": "CBorg API",
      "npm": "@ai-sdk/openai-compatible",
      "models": {
        "anthropic/claude-sonnet-high": {
          "id": "anthropic/claude-sonnet-high",
          "name": "Claude Sonnet High",
          "reasoning": true,
          "tool_call": true,
          "limit": {
            "context": 131072,
            "output": 131072
          }
        }
      },
      "options": {
        "baseURL": "https://api.cborg.lbl.gov",
        "apiKey": "{env:CBORG_API_KEY}"
      }
    }
  },
  "default_agent": "custodian_agent",
  "agent": {
    "custodian_agent": {
      "description": "Custodian agent that analyzes a hpc job using the tools given to it",
      "mode": "primary",
      "model": "cborg/anthropic/claude-sonnet-high",
      "prompt": "{file:./prompts/custodian_agent.txt}",
      "temperature": 0.8,
      "permission": {}
    }
  }
}
```
---

## Step 5: Launch the Agent

From the `custodian_setup/` directory, start OpenCode:

```
cd custodian_setup
opencode
```

OpenCode will automatically detect `opencode.json` in the current directory and load the `custodian_agent` as the default agent with all tools registered.

---

## Step 6: Run Your First Analysis

Once OpenCode is running, provide the agent with the path to a metrics data file. The agent expects a CSV-formatted file containing HPC GPU metrics collected via DCGM (Data Center GPU Manager).

Two sample datasets are included in the `data_samples/` directory to get you started.

### Sample 1: dcgm_metrics_new.dat

This dataset captures a bandwidth-bound GPU workload on a GH200. The job runs predominantly at near-zero compute utilization across metrics like GRACT, SMACT, FP64A, and INTAC, with activity concentrated in memory and PCIe transfer metrics. It is a good first dataset because the cluster analysis produces a clear, simple separation between idle and active phases. A supplementary workload output file, `wgbe_output.txt`, is included alongside it and will be automatically used by the agent to correlate workload phases with the metrics.

To analyze it, use this prompt:

```
Analyze this file: data_samples/dcgm_metrics_new.dat
```

### Sample 2: all_phases_metrics.dat

This dataset captures a multi-phase GPU workload that transitions through distinct execution states including idle, active compute, and high-throughput memory transfer. GRACT reaches 0.999, TENSO sustained above 0.86, and PCIe transmit values climb into the tens of billions. The column schema also differs from the first sample, using SMCLK, PCITX, PCIRX, and FBFRE in place of FP32A, FP16A, INTAC, and TDFMA. This makes it a more complex and representative analysis target.

To analyze it, use this prompt:

```
Analyze this file: data_samples/all_phases_metrics.dat
```

The agent will immediately call `get_cluster_analysis` on the file, interpret the resulting cluster map and roofline plots, and then propose and execute a set of follow-up investigations using the line plot, radar plot, and vision tools.

---

## How the Tools Work

### get_cluster_analysis

This is the primary entry point for every analysis. It reads the metrics file, applies K-means clustering to identify distinct execution phases, and returns a classification map, per-cluster activity time ranges, roofline plot image paths, and per-cluster compute metric means. The agent calls this exactly once per input file.

### generate_line_plot

Generates a smoothed time-series plot for a single metric over a specified time window. The agent uses this to examine temporal trends, spikes, and plateaus within a cluster's active period.

### generate_radar_plot

Generates a polar chart showing the relative values of all metrics at a single point in time. The agent uses this to get an instantaneous cross-metric snapshot at transitions or points of interest.

### generate_roofline_heatmap

Generates a roofline heatmap showing arithmetic intensity versus throughput. The agent uses this to determine whether the workload is memory-bound or compute-bound during a given phase.

### vision

Sends a plot image to the CBorg vision model (Granite 3.2) and returns a text description. The agent uses this exclusively for roofline plot images returned by `get_cluster_analysis` and `generate_roofline_heatmap`.

---

## Troubleshooting

### The agent cannot find app.py

Verify that `CUSTODIAN_APP_PATH` points to the `app/` directory and that `app.py` exists at that location. Run `echo $CUSTODIAN_APP_PATH` and confirm the path is correct.

### Python errors about missing packages

Make sure you installed the requirements into the correct virtual environment and that `CUSTODIAN_VENV_PATH` points to the `python3` binary inside that environment, not a system Python.

### CBORG_API_KEY errors

Confirm the variable is exported and not empty. Run `echo $CBORG_API_KEY` in your terminal. If it is blank, re-source your shell profile.

### Bun not found or tool execution fails

Confirm Bun is installed and on your PATH. Run `bun --version`. If the `.opencode/node_modules/` directory is missing, run `bun install` again from inside `.opencode/`.

### OpenCode does not load the custodian agent

Make sure you are launching OpenCode from inside the `custodian_setup/` directory, not a parent directory. The `opencode.json` file must be in the current working directory when OpenCode starts.
