import * as pdfjsLib from 'pdfjs-dist/legacy/build/pdf.mjs'
import { GoogleGenAI } from '@google/genai'
import { ingest } from './db.js'
import 'dotenv/config'

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY })

//1. PDF → raw text
const loadingTask = pdfjsLib.getDocument({url: './data.pdf'})
const pdf = await loadingTask.promise

let fullText = ''
for (let i = 1; i <= pdf.numPages; i++) {
  const page = await pdf.getPage(i)
  const content = await page.getTextContent()
  const pageText = content.items.map(item => item.str).join(' ')
  fullText += pageText + ' '
}

//2. Text → chunks
function chunkText(text, chunkSize = 50 , overlap = 5) {
  const words = text.split(/\s+/)
  const chunks = []
  const step = chunkSize - overlap

  for (let i = 0; i < words.length; i += step) {
    const chunk = words.slice(i, i + chunkSize).join(' ')
    chunks.push(chunk)
  }

  return chunks
}

const chunks = chunkText(fullText)

//3. Embed each chunk → store in DB
async function embedText(text) {
  const result = await ai.models.embedContent({
    model: 'gemini-embedding-2',
    contents: 'task:search result\n\n' + text
  })
  return result.embeddings[0].values
}

for (const chunk of chunks) {
  const vector = await embedText(chunk)
  await ingest(chunk, vector)
  console.log('stored chunk:', chunk.slice(0, 60), '...')
}

console.log('done. total chunks:', chunks.length)