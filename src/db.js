import 'dotenv/config'
import {Client} from 'pg'

const client = new Client({
    connectionString: process.env.PG,
})

await client.connect()

// Store a chunk
export async function ingest(documentName, pageNumber, chunkIndex, chunkText, embeddingArray) {
    try {
        await client.query(
            `INSERT INTO chunks
            (document_name, page_number, chunk_index, content, embedding)
            VALUES ($1, $2, $3, $4, $5)`,
            [
                documentName,
                pageNumber,
                chunkIndex,
                chunkText,
                JSON.stringify(embeddingArray)
            ]
        )
    } catch (err) {
        console.error(err)
    }
}

// Find similar chunks (cosine distance)
export async function search(queryEmbedding){
    try{
        const res = await client.query(
            'SELECT content , chunk_index FROM chunks ORDER BY embedding <=> $1 LIMIT 10',
            [JSON.stringify(queryEmbedding)]
        )
        return res.rows
    }catch(err){
        console.error(err)
    }

}