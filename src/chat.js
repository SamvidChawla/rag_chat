import { GoogleGenAI } from '@google/genai'
import { search } from './db.js'
import 'dotenv/config'

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

export async function embedQuery(text) {
  const result = await ai.models.embedContent({
    model: 'gemini-embedding-2',
    contents: 'task:search query\n\n' + text
  })
  return result.embeddings[0].values
}

export async function queryLLM(Query , context){
    const contextText = context.map(row => row.content).join('\n\n')
    const chunk_index = context.map(row => row.chunk_index).join(',')
    const response = await ai.models.generateContent({
    model: "gemini-3.1-flash-lite",
    contents: "Query:" + Query + "\n\nContext:" + contextText,
    config: {
      systemInstruction: "You are an Internal Assistant , Answer Strictly on the basis of context , However Reply in Natural Language",
    },
  })
  return {response: response.text , chunk_index}
}