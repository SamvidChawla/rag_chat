import {Client} from 'pg'

const client = new Client({
    connectionString: process.env.PG,
})

await client.connect()

// Store a chunk
export async function ingest(chunkText, embeddingArray) {
    try{
        await client.query(
            'INSERT INTO chunks (content, embedding) VALUES ($1, $2)',
            [chunkText, JSON.stringify(embeddingArray)]
        )
    }catch(err){
        console.error(err)
    }
}

// Find similar chunks (cosine distance)
export async function search(queryEmbedding){
    try{
        const res = await client.query(
            'SELECT content FROM chunks ORDER BY embedding <=> $1 LIMIT 3',
            [JSON.stringify(queryEmbedding)]
        )
        return res.rows
    }catch(err){
        console.error(err)
    }

}