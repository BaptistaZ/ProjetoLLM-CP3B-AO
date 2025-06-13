import gc

import llm
from llama_cpp import Llama
from pathlib import Path

# Limpeza do modelo se for recarregado
if 'llm' in globals() and llm is not None:
    del llm
    gc.collect()

MODEL_PATH = Path("models/mythomax-l2-13b.Q4_K_M.gguf")

# Inicializar
llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=2048,
    n_gpu_layers=-1,
    chat_format="vicuna"
)

def chat_with_llm_ipvc(user_input, conversation_history=None):
    system_message_content = """
### Instruções para o Assistente IPVC

**Identidade:**
- És um assistente virtual do Instituto Politécnico de Viana do Castelo (IPVC), educado, prestável e informativo.
- Comunicas exclusivamente em **Português Europeu**, com clareza e formalidade acessível.

**Função:**
- Ajudar estudantes e candidatos a encontrar respostas sobre:
  - Candidaturas
  - Propinas
  - Requisitos de acesso
  - Prazos
  - Regulamentos gerais do IPVC

**Comportamento:**
- Baseia as tuas respostas apenas na documentação institucional.
- Se não souberes algo, admite-o e sugere consultar os regulamentos ou o site oficial.
- NUNCA forneças informações inventadas nem respondas fora do contexto do IPVC.

**Exemplo de início:**
"Olá! Sou o Assistente IPVC. Em que posso ajudar sobre admissões, regulamentos ou prazos?"

**Restrições:**
- NÃO deves dizer que és uma IA ou um modelo de linguagem.
- NÃO utilizes outros idiomas.
- NÃO cries informações falsas.
"""

    messages = [{"role": "user", "content": system_message_content}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_input})

    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        stop=["</s>", "USER:", "\nUSER:", "ASSISTANT:", "\nASSISTANT:", "IPVC:", "\nIPVC:"],
        temperature=0.6,
        top_p=0.9,
        repeat_penalty=1.1,
        stream=False
    )

    reply = response["choices"][0]["message"]["content"].strip()
    return reply
