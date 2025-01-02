/**
 * server.js
 *
 * A production-ready API that:
 *   - Takes a "query" from the request body (but it's optional now)
 *   - If query is present:
 *       - Generates an OpenAI embedding
 *       - Retrieves top matches from Supabase (using match_documents RPC)
 *       - Reranks those matches with Cohere
 *       - Returns the top 3 results (title + content) from the reranker
 *     Otherwise:
 *       - Returns all documents
 */

import express from "express";
import { config } from "dotenv";
import { createClient } from "@supabase/supabase-js";
import OpenAI from "openai";
import { CohereClient } from "cohere-ai";
import cors from 'cors';

app.use(cors());

// 1. Load environment variables
config(); // loads .env

const {
  SUPABASE_URL,
  SUPABASE_SERVICE_ROLE_KEY,
  OPENAI_API_KEY,
  CO_API_KEY,
  PORT,
} = process.env;

// 2. Create Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
  auth: { persistSession: false },
});

// 3. Create OpenAI client
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// 4. Create Cohere client
const cohere = new CohereClient({ apiKey: CO_API_KEY });

// 5. Initialize Express
const app = express();
app.use(express.json());

/**
 * POST /search
 * Body: { "query"?: "some search term" }
 *
 * If query is provided, returns top 3 documents after reranking with Cohere.
 * If query is not provided, returns all documents.
 */
app.post("/search", async (req, res) => {
  try {
    const { query } = req.body;

    // If no query is provided, just return all documents
    if (!query) {
      const { data: allDocs, error: allDocsError } = await supabase
        .from("documents") // change to your actual table name
        .select("id, title, content");

      if (allDocsError) {
        console.error("Error retrieving all documents:", allDocsError.message);
        return res.status(500).json({ error: "Error retrieving all documents." });
      }

      // Return full JSON (all documents)
      return res.status(200).json({
        data: allDocs,
      });
    }

    //
    // If query is provided, do the usual process:
    //

    // 1. Generate embedding for the user's query with OpenAI
    const embeddingResponse = await openai.embeddings.create({
      input: query,
      model: "text-embedding-3-small", // or another OpenAI embedding model
    });
    const [{ embedding }] = embeddingResponse.data;

    // 2. Fetch matches from Supabase via RPC
    const { data: matches, error: rpcError } = await supabase
      .rpc("match_documents", {
        query_embedding: embedding,
        match_threshold: 0.3, // Adjust as needed
      })
      .select("id, title, content") // only need these fields for reranking
      .limit(10);

    if (rpcError) {
      console.error("match_documents RPC error:", rpcError.message);
      return res
        .status(500)
        .json({ error: "Error calling match_documents in Supabase." });
    }

    // 3. Rerank matches using Cohere
    const matchedDocs = matches.map((doc, index) => ({
      index,
      title: doc.title,
      content: doc.content,
    }));

    const cohereResponse = await cohere.v2.rerank({
      model: "rerank-v3.5", // or "rerank-english-v2.0", etc.
      query,
      documents: matchedDocs.map((m) => m.content),
      topN: 3, // We only want the top 3
    });

    // Construct final data for the top 3
    const rerankedResults = cohereResponse.results.map((r) => {
      const doc = matchedDocs[r.index];
      return {
        title: doc.title,
        content: doc.content,
        relevance_score: r.relevance_score,
      };
    });

    // 4. Return top 3 reranked documents
    return res.status(200).json({
      data: rerankedResults,
    });
  } catch (error) {
    console.error("Error in /search endpoint:", error);
    return res.status(500).json({ error: "An unexpected error occurred." });
  }
});

// 6. Start the server
const port = PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on port ${port}...`);
});
