# Quiz do Programa Renda Extra Ton — IA Generativa

Este projeto implementa um **quiz automatizado de 10 perguntas** sobre o regulamento do programa **Renda Extra Ton**, utilizando:

- **Google Generative AI (Gemini)** para gerar perguntas objetivas de múltipla escolha.
- **FAISS** + **embeddings** para busca contextual precisa no PDF do regulamento.
- **Chunking inteligente** para dividir o documento em seções relevantes.
- Um **sistema interativo de quiz** com pontuação, justificativas e feedback educativo.

Este repositório contém todo o pipeline do desafio técnico solicitado.

---

## Funcionalidades

✔ Extração do conteúdo do regulamento  
✔ Chunking do texto
✔ Geração de embeddings e indexação FAISS  
✔ Recuperação de contexto relevante (RAG)  
✔ Criação de perguntas (MCQs) com 4 alternativas  
✔ Validação das perguntas (JSON estruturado)  
✔ Execução do quiz no notebook  
✔ Feedback de erros com explicações  
✔ Relatório final em JSON
