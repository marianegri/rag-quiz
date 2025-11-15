import sys
sys.dont_write_bytecode = True
import json
from guardrails import Guard
from pydantic import BaseModel, Field
from typing import List, Literal

class MCQ(BaseModel):
    question: str = Field(description="Pergunta direta, sem mencionar o texto")
    options: List[str] = Field(min_items=4, max_items=4)
    answer: Literal["A", "B", "C", "D"]
    explanation: str

mcq_guard = Guard(
    output_schema=MCQ.model_json_schema()
)

def generate_mcq_from_context(genai_client, context_text, model_name="gemini-2.5-flash"):
    """
    Gera 1 MCQ em JSON usando google.generativeai + Guardrails (Pydantic).
    Retorna dict com keys: question, options, answer, explanation.
    """
    prompt = f"""
    Crie UMA pergunta de múltipla escolha com base APENAS no contexto abaixo.
    IMPORTANTE:
    - Não invente nada além do que está no contexto.
    - Não use frases como "de acordo com o texto", "segundo o contexto", "conforme o texto" ou equivalentes.
    - A pergunta deve ser direta e clara, sem mencionar o texto ou o contexto.
    - Gere exatamente 4 alternativas.
    - NÃO coloque rótulos ou letras nas alternativas (não use "A)", "B.", etc). Apenas forneça o texto puro de cada alternativa.
    - Haverá exatamente 1 alternativa correta, suportada pelo contexto.

    Contexto:
    {context_text}

    Formato EXATO da resposta (somente JSON válido):
    {{
      "question": "pergunta direta e objetiva, sem mencionar o texto",
      "options": ["texto da opção 1", "texto da opção 2", "texto da opção 3", "texto da opção 4"],
      "answer": "D",
      "explanation": "justificativa curta baseada no contexto"
    }}
    """

    model = genai_client.GenerativeModel(model_name)
    resp = model.generate_content(prompt)

    # passar para Guardrails (auto-repair ocorre internamente)
    validated = mcq_guard.parse(resp.text)

    # extrair o dict validado do possível envelope
    parsed = validated.validated_output

    # validação mínima local (mantém compatibilidade)
    if not all(k in parsed for k in ("question", "options", "answer", "explanation")):
        raise ValueError("Resposta validada não contém todas as chaves esperadas: " + str(parsed))

    if not isinstance(parsed["options"], list) or len(parsed["options"]) != 4:
        raise ValueError("Número de opções != 4: " + str(parsed["options"]))

    parsed["answer"] = str(parsed["answer"]).strip().upper()
    return parsed