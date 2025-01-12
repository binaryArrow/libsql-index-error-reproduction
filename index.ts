import {createClient} from "@libsql/client";
import path from "node:path";
import {LlamaCppEmbeddings} from "@langchain/community/embeddings/llama_cpp";
import {join} from "path";
import {PDFLoader} from "@langchain/community/document_loaders/fs/pdf";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters";
import {Document} from '@langchain/core/documents';
import {fileURLToPath} from "node:url";
import {LibSQLVectorStore} from "@langchain/community/vectorstores/libsql";

// we need this function to pad the embeddings to the correct size, somehow vectorstres addDocument function does not do this
function padOrTruncateEmbedding(embedding: number[], targetSize: number): number[] {
    if (embedding.length < targetSize) {
        return [...embedding, ...new Array(targetSize - embedding.length).fill(0)];
    }
    return embedding;
}
// create DB and indexes, the vector size for llamacpp is 4096
const client = createClient({
    url: 'file:local.db'
});
await client.batch(
    [
        `CREATE TABLE IF NOT EXISTS vectors
         (
             id        INTEGER PRIMARY KEY AUTOINCREMENT,
             content   TEXT,
             metadata  TEXT,
             embedding F32_BLOB(4096)
         );`,
        `CREATE INDEX IF NOT EXISTS vector_idx ON vectors (libsql_vector_idx(embedding));`,
    ],
    'write'
);

// create embeddingmodel
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const llamaEmbeddings = await LlamaCppEmbeddings.initialize({
    modelPath: path.join(join(__dirname, '/llm-model'), 'Mistral-7B-Instruct-v0.3.Q4_K_M.gguf')
});

// get pdfloader and split the pdf into chunks
const loader = new PDFLoader('testpdf');
const docs = await loader.load();
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 500
});
const splits: Document[] = await textSplitter.splitDocuments(docs);
console.log('Splits:', splits.length);

// create the vectorstore and embed the document
const testVectorStore = new LibSQLVectorStore(llamaEmbeddings, {
    db: client,
});
const embeddings = await llamaEmbeddings.embedDocuments(splits.map((doc) => doc.pageContent));
embeddings.forEach((embedding, i) => {
    embeddings[i] = padOrTruncateEmbedding(embedding, 4096);
});
await testVectorStore.addVectors(embeddings, splits);
const question = 'give me all words in the documents that relate to the word drugs'
console.log('Retrieving documents for question:', question);
const retrievedDocs = await testVectorStore.similaritySearch(question);
console.log(retrievedDocs)