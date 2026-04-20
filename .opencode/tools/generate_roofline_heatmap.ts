import { tool } from "@opencode-ai/plugin";

export default tool({
  description: "Generate a roofline heatmap plot showing arithmetic intensity vs throughput for HPC workload analysis.",
  args: {
    filePath: tool.schema.string().describe("Path to the CSV file containing HPC metrics data"),
    x_min: tool.schema.number().optional().describe("Minimum arithmetic intensity on the x-axis (default: 0.01)"),
    x_max: tool.schema.number().optional().describe("Maximum arithmetic intensity on the x-axis (default: 1000.0)"),
    y_min_log10: tool.schema.number().optional().describe("Minimum log10 throughput on the y-axis (default: -2.0)"),
    y_max_log10: tool.schema.number().optional().describe("Maximum log10 throughput on the y-axis (default: 3.0)"),
    x_bins: tool.schema.number().optional().describe("Number of x-axis bins for the heatmap (default: 40)"),
    y_bins: tool.schema.number().optional().describe("Number of y-axis bins for the heatmap (default: 40)"),
  },
  async execute(args) {
    const pythonEnv = process.env.CUSTODIAN_VENV_PATH ?? "/opt/venv/bin/python3";
    const script = (process.env.CUSTODIAN_APP_PATH ?? "/opt/custodian-agent/app") + "/app.py";
    const cmdArgs: string[] = [
      pythonEnv,
      script,
      "--functionName",
      "generate_roofline_heatmap",
      "--inputFile",
      args.filePath,
    ];
    if (args.x_min !== undefined) cmdArgs.push("--x-min", String(args.x_min));
    if (args.x_max !== undefined) cmdArgs.push("--x-max", String(args.x_max));
    if (args.y_min_log10 !== undefined) cmdArgs.push("--y-min-log10", String(args.y_min_log10));
    if (args.y_max_log10 !== undefined) cmdArgs.push("--y-max-log10", String(args.y_max_log10));
    if (args.x_bins !== undefined) cmdArgs.push("--x-bins", String(args.x_bins));
    if (args.y_bins !== undefined) cmdArgs.push("--y-bins", String(args.y_bins));
    const result = await Bun.$`${cmdArgs}`.text();
    return result;
  },
});
