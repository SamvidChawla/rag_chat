import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs'
import { GoogleGenAI } from '@google/genai'
import { ingest } from './db.js'
import 'dotenv/config'

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

//1. PDF → raw text
const loadingTask = pdfjsLib.getDocument({url: './data.pdf'})
const pdf = await loadingTask.promise

const chunks = []

for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber++) {
  const page = await pdf.getPage(pageNumber)
  const content = await page.getTextContent()

  const pageText = content.items.map(item => item.str).join(' ')

  chunks.push(...chunkText(pageText, pageNumber))
}

//2. Text → chunks
function chunkText(text, pageNumber, chunkSize = 200, overlap = 20) {
  const words = text.split(/\s+/)
  const chunks = []
  const step = chunkSize - overlap

  for (let i = 0; i < words.length; i += step) {
    chunks.push({
      pageNumber,
      content: words.slice(i, i + chunkSize).join(' ')
    })
  }

  return chunks
}

//3. Embed each chunk → store in DB
async function embedText(text) {
  let attempts = 0
  const maxAttempts = 5
  let delay = 2000

  while (attempts < maxAttempts) {
    try {
      const result = await ai.models.embedContent({
        model: 'gemini-embedding-2',
        contents: 'task:search result\n\n' + text
      });
      return result.embeddings[0].values
    } catch (err) {
      if (err.status === 429) {
        attempts++
        console.log(`Rate limited. Waiting ${delay / 1000} seconds... (Attempt ${attempts}/${maxAttempts})`)
        await new Promise(r => setTimeout(r, delay))
        delay *= 2
      } else {
        console.error(err) 
      }
    }
  }
  console.error("Failed to get embedding after maximum retries.");
}


for (let i = 0; i < chunks.length; i++) {
  const chunk = chunks[i]

  const vector = await embedText(chunk.content)

  await ingest(
    'Data.pdf',
    chunk.pageNumber,
    i,
    chunk.content,
    vector
  )

  console.log('stored chunk:', chunk.content.slice(0, 60), '...')
}

console.log('done. total chunks:', chunks.length)