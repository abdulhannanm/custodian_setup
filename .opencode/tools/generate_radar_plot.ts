import { tool } from "@opencode-ai/plugin";

export default tool({
  description:
    "Generate a radar plot for a specific time point showing the feature values at that moment. Visualizes the relative values of different metrics (TENSO, DRAMA, FP64A, PCIE_LOAD, PCIE_DIRECTION) as a polar chart.",
  args: {
    time: tool.schema.number().describe("The time point to analyze (time = index + 1)"),
    filePath: tool.schema.string().describe("Path to the CSV file containing HPC metrics data"),
  },
  async execute(args) {
    const pythonEnv = process.env.CUSTODIAN_VENV_PATH ?? "/opt/venv/bin/python3";
    const script = (process.env.CUSTODIAN_APP_PATH ?? "/opt/custodian-agent/app") + "/app.py";
    const command = Bun.$`${pythonEnv} ${script} --functionName generate_radar_plot --time ${args.time} --inputFile ${args.filePath}`;
    const result = await command.text();
    return result;
  },
});
