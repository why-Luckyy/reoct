require("dotenv").config();
const express = require("express");
const cors = require("cors");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { OpenAI } = require("openai");

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS for all routes
app.use(cors());

// Parse JSON payloads
app.use(express.json());

// Production-grade System Prompt enforcing React, TS, Vite, RTK, Hook Form, Yup, GraphQL, Axios
const SYSTEM_PROMPT = `You are a production-grade AI code generator and senior full-stack architect specializing in high-performance frontend and client applications.
Your job is to generate ready-to-run React, TypeScript, and Vite code bases based on the user's requirements.

You must write complete, production-ready, clean, and beautifully styled files.
DO NOT use any placeholders. DO NOT use TODOs. Write every single line of code completely.

CRITICAL TECH STACK REQUIREMENTS:
- Framework & Build Tool: React with TypeScript, structured for Vite.
- State Management: Redux Toolkit (@reduxjs/toolkit and react-redux).
- Form Handling: React Hook Form.
- Validation: Yup (for validation schemas).
- API Client / Networking: Axios.
- API Queries/Mutations: GraphQL (mock schema, queries, and mutations where applicable).

OUTPUT STRUCTURE & FORMAT REQUIREMENTS:
- You must output your entire response as a single, valid JSON object.
- The JSON object must strictly match the following schema:
{
  "folders": [
    "src",
    "src/components",
    "src/features",
    "src/graphql"
  ],
  "files": [
    {
      "path": "src/components/LoginForm.tsx",
      "content": "/* complete source code */"
    }
  ]
}

- Every folder in 'folders' must be listed before any files in those folders are created.
- Every file in 'files' must specify its relative path from the project root and its FULL, COMPLETE content.
- DO NOT put comments in the JSON structure itself.
- Code content must not contain any placeholders or omitted implementation.
- All code files must use standard, correct imports, TypeScript interfaces, and fully functional components.
- Absolutely NO conversational text, markdown wrapping (such as \`\`\`json ... \`\`\`), or explanations. Only raw JSON.`;

// Secure API Key authentication middleware
const authenticateKey = (req, res, next) => {
  const appKey = req.headers["x-app-key"];
  const expectedKey = process.env.APP_API_KEY;

  if (!expectedKey) {
    console.error("CRITICAL: APP_API_KEY is not defined in backend environment variables.");
    return res.status(500).json({ error: "Backend is misconfigured: APP_API_KEY is missing." });
  }

  if (!appKey || appKey !== expectedKey) {
    return res.status(401).json({ error: "Unauthorized: Invalid or missing x-app-key header." });
  }

  next();
};

// Route: Status check
app.get("/health", (req, res) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

// Helper: Secure JSON cleanup
function cleanJSONResponse(rawText) {
  let cleaned = rawText.trim();
  // Remove markdown codeblock wrapper if present
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```json\s*/i, "").replace(/```$/, "");
  }
  return cleaned.trim();
}

// Route: Main generation endpoint
app.post("/generate", authenticateKey, async (req, res) => {
  const { prompt, provider = "gemini" } = req.body;

  if (!prompt) {
    return res.status(400).json({ error: "Prompt is required." });
  }

  console.log(`[Generate Request] Provider: ${provider}, Prompt: "${prompt.substring(0, 50)}..."`);

  try {
    let resultJSON = null;

    if (provider === "gemini") {
      if (!process.env.GEMINI_API_KEY) {
        throw new Error("GEMINI_API_KEY is not configured on the backend server.");
      }

      const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
      const modelsToTry = [
        "gemini-2.5-flash",
        "gemini-3.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash"
      ];

      let lastError = null;
      let rawText = null;

      for (const modelName of modelsToTry) {
        try {
          console.log(`[Gemini] Attempting generation with model: ${modelName}`);
          const model = genAI.getGenerativeModel({
            model: modelName,
            systemInstruction: SYSTEM_PROMPT,
          });

          const response = await model.generateContent({
            contents: [{ role: "user", parts: [{ text: prompt }] }],
            generationConfig: {
              responseMimeType: "application/json",
              temperature: 0.2,
            },
          });

          rawText = response.response.text();
          if (rawText) {
            console.log(`[Gemini] Success using model: ${modelName}`);
            break;
          }
        } catch (err) {
          console.warn(`[Gemini] Failed with model ${modelName}:`, err.message);
          lastError = err;
        }
      }

      if (!rawText) {
        throw lastError || new Error("All Gemini models failed to generate content.");
      }

      const cleaned = cleanJSONResponse(rawText);
      resultJSON = JSON.parse(cleaned);

    } else if (provider === "openai") {
      if (!process.env.OPENAI_API_KEY) {
        throw new Error("OPENAI_API_KEY is not configured on the backend server.");
      }

      const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
      const completion = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: prompt }
        ],
        response_format: { type: "json_object" },
        temperature: 0.2,
      });

      const rawText = completion.choices[0].message.content;
      const cleaned = cleanJSONResponse(rawText);
      resultJSON = JSON.parse(cleaned);

    } else if (provider === "groq") {
      if (!process.env.GROQ_API_KEY) {
        throw new Error("GROQ_API_KEY is not configured on the backend server.");
      }

      // Groq provides an OpenAI-compatible endpoint
      const groq = new OpenAI({
        apiKey: process.env.GROQ_API_KEY,
        baseURL: "https://api.groq.com/openai/v1",
      });

      const groqModels = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "llama3-70b-8192"
      ];

      let lastError = null;
      let rawText = null;

      for (const modelName of groqModels) {
        try {
          console.log(`[Groq] Attempting generation with model: ${modelName}`);
          const completion = await groq.chat.completions.create({
            model: modelName,
            messages: [
              { role: "system", content: SYSTEM_PROMPT },
              { role: "user", content: prompt }
            ],
            response_format: { type: "json_object" },
            temperature: 0.2,
          });

          rawText = completion.choices[0].message.content;
          if (rawText) {
            console.log(`[Groq] Success using model: ${modelName}`);
            break;
          }
        } catch (err) {
          console.warn(`[Groq] Failed with model ${modelName}:`, err.message);
          lastError = err;
        }
      }

      if (!rawText) {
        throw lastError || new Error("All Groq models failed to generate content.");
      }

      const cleaned = cleanJSONResponse(rawText);
      resultJSON = JSON.parse(cleaned);

    } else {
      return res.status(400).json({ error: `Unsupported provider: ${provider}` });
    }

    // Safety Checks: Standardized format check
    if (!resultJSON.folders || !Array.isArray(resultJSON.folders)) {
      resultJSON.folders = [];
    }
    if (!resultJSON.files || !Array.isArray(resultJSON.files)) {
      resultJSON.files = [];
    }

    return res.json(resultJSON);

  } catch (error) {
    console.error("Generation Error:", error);
    return res.status(500).json({
      error: "LLM generation failed or returned invalid response.",
      details: error.message
    });
  }
});

// Start Server
app.listen(PORT, () => {
  console.log(`==================================================`);
  console.log(`Backend Server running on port ${PORT}`);
  console.log(`Health Check: http://localhost:${PORT}/health`);
  console.log(`API key protection enabled.`);
  console.log(`==================================================`);
});
