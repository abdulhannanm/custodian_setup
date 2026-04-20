import { tool } from "@opencode-ai/plugin";
import fs from "fs";
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.CBORG_API_KEY,
  baseURL: "https://api.cborg.lbl.gov",
});

export default tool({
  description:
    "Examines an image or plot (PNG/JPG) using CBorg Vision (Granite 3.2) and returns a text description of the visual data.",
  args: {
    imagePath: tool.schema.string().describe("The local filesystem path to the image file."),
    prompt: tool.schema.string().describe("What to look for in the image."),
  },
  async execute(args) {
    try {
      const imageBuffer = fs.readFileSync(args.imagePath);
      const base64Image = imageBuffer.toString("base64");

      const response = await client.chat.completions.create({
        model: "gpt",
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: args.prompt },
              { type: "image_url", image_url: { url: `data:image/png;base64,${base64Image}` } },
            ],
          },
        ],
      });

      return response.choices[0].message.content;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return `Vision Error: ${message}`;
    }
  },
});
