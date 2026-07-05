import 'dotenv/config'
import express from 'express'
import { search } from './src/db.js'
import { embedQuery , queryLLM } from './src/chat.js'

const app = express()
app.use(express.json({ limit: "10kb" }))

app.post('/query', async (req, res) => {
  const { query } = req.body
  if (!query) return res.status(400).json({ error: 'query required' })

  const embedding = await embedQuery(query)
  const context = await search(embedding)
  const answer = await queryLLM(query, context)

  res.json({ answer })
})

const startServer = async () => {
  const port = 3000;
  app.listen(port,"0.0.0.0", () => {
    console.log(`Server running on port ${port}`);
  });
};

startServer();