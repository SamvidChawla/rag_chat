import { GoogleGenAI } from '@google/genai'
import { search } from './db.js'
import 'dotenv/config'

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

async function embedQuery(text) {
  const result = await ai.models.embedContent({
    model: 'gemini-embedding-2',
    contents: 'task:search query\n\n' + text
  })
  return result.embeddings[0].values
}

const Query = 'What are the Rules?'
const queryEmbedding= await embedQuery(Query)
const context = await search(queryEmbedding)

async function queryLLM(Query , context){
    const contextText = context.map(row => row.content).join('\n\n')
    const response = await ai.models.generateContent({
    model: "gemini-3.1-flash-lite",
    contents: "Query:" + Query + "\n\nContext:" + contextText,
    config: {
      systemInstruction: "You are an Internal Assistant , Answer Strictly on the basis of context , However Reply in Natural Language",
    },
  });
  console.log(response.text)
}

await queryLLM(Query , context)