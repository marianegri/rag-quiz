"""
Módulo de Geração de Questões de Múltipla Escolha (MCQ)
============================================================

Este módulo usa LLM e Guardrails para gerar questões de múltipla escolha
validadas e estruturadas a partir de contextos textuais.

Autor: Maria Negri
"""

import sys
sys.dont_write_bytecode = True

from typing import List, Literal, Dict, Any
from guardrails import Guard
from pydantic import BaseModel, Field


# ==================== CONSTANTES ====================

DEFAULT_MODEL = "gemini-2.5-flash"
VALID_ANSWERS = ["A", "B", "C", "D"]

# ==================== MODELOS DE DADOS ====================

class MCQ(BaseModel):
    """
    Modelo de dados para questões de múltipla escolha.
    
    Attributes:
        question: Pergunta formulada de forma direta e objetiva
        options: Lista com exatamente 4 alternativas
        answer: Letra da resposta correta (A, B, C ou D)
        explanation: Justificativa da resposta correta
    """
    question: str = Field(
        description="Pergunta direta e objetiva, sem mencionar o texto"
    )
    options: List[str] = Field(
        min_items=len(VALID_ANSWERS),
        max_items=len(VALID_ANSWERS),
        description="Lista com exatamente 4 alternativas"
    )
    answer: str = Field(
        description="Letra da resposta correta (A, B, C ou D)"
    )
    explanation: str = Field(
        description="Justificativa da resposta com base no contexto"
    )

mcq_guard = Guard(
    output_schema=MCQ.model_json_schema()
)

def generate_mcq_from_context(
    genai_client,
    context_text: str,
    model_name: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Gera uma questão de múltipla escolha validada a partir de um contexto.
    
    Args:
        genai_client: Cliente da API Google Generative AI
        context_text: Texto de contexto para geração da questão
        model_name: Nome do modelo LLM a usar
        
    Returns:
        Dicionário com:
            - question: Texto da pergunta
            - options: Lista com 4 alternativas
            - answer: Letra da resposta correta (A-D)
            - explanation: Justificativa da resposta
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