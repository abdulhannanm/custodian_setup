import { tool } from "@opencode-ai/plugin";

export default tool({
  description:
    "Analyze HPC metrics data to identify different phases of execution using K-means clustering. Returns a classification map of cluster labels and the time ranges when each cluster was active.",
  args: {
    filePath: tool.schema.string().describe("Path to the CSV file containing HPC metrics data"),
  },
  async execute(args) {
    const pythonEnv = process.env.CUSTODIAN_VENV_PATH ?? "/opt/venv/bin/python3";
    const script = (process.env.CUSTODIAN_APP_PATH ?? "/opt/custodian-agent/app") + "/app.py";
    const command = Bun.$`${pythonEnv} ${script} --inputFile ${args.filePath} --functionName get_cluster_analysis`;
    const result = await command.text();
    return result;
  },
});
