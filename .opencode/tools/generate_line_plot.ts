import { tool } from "@opencode-ai/plugin";

export default tool({
  description:
    "Generate a smoothed line plot for a specific metric over a time range. Applies Savitzky-Golay filtering and median filtering to smooth the signal and highlight trends.",
  args: {
    label: tool.schema
      .string()
      .describe(
        "The metric/column name to plot, YOU MUST PICK FROM THE FOLLOWING LIST: ('TENSO', 'DRAMA', 'FP64A', 'PCIE_LOAD', 'PCIE_DIRECTION')",
      ),
    xmin: tool.schema.number().describe("The starting time point for the plot range"),
    xmax: tool.schema.number().describe("The ending time point for the plot range"),
    filePath: tool.schema.string().describe("Path to the CSV file containing HPC metrics data"),
  },
  async execute(args) {
    const pythonEnv = process.env.CUSTODIAN_VENV_PATH ?? "/opt/venv/bin/python3";
    const script = (process.env.CUSTODIAN_APP_PATH ?? "/opt/custodian-agent/app") + "/app.py";
    const command = Bun.$`${pythonEnv} ${script} --functionName plot_signal_smoothing --label ${args.label} --xmin ${args.xmin} --xmax ${args.xmax} --inputFile ${args.filePath}`;
    const result = await command.text();
    return result;
  },
});
