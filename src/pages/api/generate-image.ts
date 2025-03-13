import { NextApiRequest, NextApiResponse } from "next";
import { HfInference } from "@huggingface/inference";

// Initialize Hugging Face Inference from the API key
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method Not Allowed" });
  }
  const { story } = req.body;
  
  // This statement ensures that a story is provided and is not empty
  if (!story || story.trim().length === 0) {
    return res.status(400).json({ error: "No story provided" });
  }

  try {
    // This calls Hugging Face's FLUX.1-dev from the story
    const response = await hf.textToImage({
      model: "black-forest-labs/FLUX.1-dev",
      inputs: `Create a cartoon image based off this story: ${story}`,
      parameters: { num_inference_steps: 20 },
    });

    // Checks whether the response is an image data
    if (response instanceof Blob) {
      // Converts to Base64 string
      const buffer = await response.arrayBuffer();
      const base64Image = Buffer.from(buffer).toString("base64");
      const imageUrl = `data:image/png;base64,${base64Image}`;

      // Sends the image URL back
      res.status(200).json({ imageUrl });
    } else {
      throw new Error("No image returned.");
    }
  } catch (error) {
    console.error("Error generating image:", error);
    res.status(500).json({ error: "Failed to generate image" });
  }
}
