### This is a minimal reproducible example of the ```vector index(search): failed to parse vector index parameters``` Error

if you want to test this out yourself, you have to download the mistral 7b model which u you can find here:
https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/blob/main/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf
download this gguf file and put it in the `llm-model` directory.

to start the application run `npm run develop` in the root directory of this project.
This will create a libsql DB and fill it with the chunks of the given pdf in this repository (testpdf) and 
the corresponding embeddings.