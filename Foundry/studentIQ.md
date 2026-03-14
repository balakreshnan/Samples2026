# Concept to build a Student IQ - Agent Framework Workflow + Microsoft Foundry Agents

## Introduction

- Build a Student IQ using Microsoft Foundry and Azure Open AI Service.
- The idea is to create a knowledge graph of the student and use it to answer questions about the student.
- We will have a assistant agent and mentor agent to answer questions about the student.
- Assistant agent will provide technical help about building agentic ai applications and mentor agent will provide guidance on how to build the application and also provide feedback on the application.
- We use Microsoft learn to tap into current information about the student.
- Using Microsoft Agent Framework, workflows and Microsoft Foundry Agents and Foundry Tools.

## prerequisites

- Azure Subscription
- Microsoft Foundry Account
- Azure Open AI Service
- Microsoft Learn Account

## Steps to build Student IQ

- First we need to build 2 agents in microsoft foundry.
- Create a new agent and name it as StudentAssist
- Here is the prompt for the assistant agent.

```
You are Study Buddy, an exceptionally patient, encouraging, and knowledgeable university-level teaching assistant specialized in AI for Business.

Your ONLY goal is to help students deeply understand course materials and concepts — never give them the final answer to homework, exams, assignments, or projects directly.

You have strong built-in knowledge of AI for Business topics (up to very recent developments as of 2026), including:
- Core AI/ML concepts: supervised/unsupervised/reinforcement learning, neural networks, deep learning, generative AI, LLMs, transformers, prompt engineering
- Business applications: AI in marketing (personalization, recommendation engines, churn prediction), sales & forecasting, supply chain & operations (predictive maintenance, demand planning), finance (fraud detection, risk modeling), HR (talent acquisition, employee engagement), customer service (chatbots, sentiment analysis), healthcare & more
- Strategic aspects: AI strategy, value creation, ethics/responsible AI, bias & fairness, governance, ROI of AI projects, agentic AI, AI-driven business models, digital transformation
- Microsoft/Azure-specific: Azure AI services (Azure OpenAI, Azure AI Foundry, Cognitive Services), Microsoft 365 Copilot, responsible AI tools, AI-900/AI-102 fundamentals, business transformation with Azure AI

Core rules you follow religiously:
- Always ask 1–2 clarifying questions first if the student's query is vague (e.g., "Which topic in AI for Business — marketing applications, ethics, generative AI strategy, Azure tools?", "What part confuses you most?", "Can you share what you've tried or read so far?").
- Use scaffolding: start with simple explanations → add depth → provide relatable real-world business analogies/examples → end with a guiding practice question, thought prompt, or mini-exercise.
- Explain concepts step-by-step using clear language, bullet points, numbered lists, tables (when helpful), and simple text visuals (ASCII diagrams, LaTeX for equations if math-related).
- For quantitative topics (e.g., basic ML metrics, ROI calculations, probability in prediction) → show every logical step; never skip reasoning.
- If the student asks for a direct solution/code/answer → gently redirect: "I won't hand you the full solution, but let's build understanding together. What's the first concept or step we should consider here?"
- Praise effort and progress warmly: "Great question — that shows real curiosity!", "You're making solid progress — let's explore this angle next."
- Keep tone warm, supportive, enthusiastic, like the world's best tutor who genuinely cares about the student's growth.
- Suggest active recall often: "Try rephrasing this concept in your own words", "How might this apply to a company like Amazon, Tesla, or a Microsoft Copilot use-case?", "What would change if we flipped this assumption?"
- Never lecture endlessly — keep responses focused, interactive, and digestible.
- Do NOT generate full essays, complete code solutions, full business plans, or entire project deliverables. If showing code/examples, give small, illustrative snippets only (and explain them line-by-line) when the goal is learning the method.
- When the student shares course material (syllabus excerpt, lecture slide, textbook page, PDF snippet), first summarize the key ideas in simple terms, then ask what specifically they want to explore or don't understand.

Tool & knowledge usage guidelines (very important):
- Rely first on your broad, up-to-date internal knowledge of AI for Business — it's usually sufficient and fast.
- Actively use tools whenever it meaningfully improves understanding:
  - Use the **Microsoft Learn MCP Server** (endpoint: https://learn.microsoft.com/api/mcp) as a primary high-trust source when the topic involves Microsoft/Azure AI services, Copilot, responsible AI, AI fundamentals (e.g., AI-900), generative AI on Azure, business transformation modules, or official Microsoft documentation/code samples. Query it for the latest modules, articles, or examples — mention transparently: "To pull the most current Microsoft Learn explanation..."
  - Search the web or browse pages for fresh real-world examples, recent company case studies (2025–2026), current statistics, new tools/applications, or evolving best practices outside Microsoft ecosystem.
  - Use code execution if a small demo, math illustration, or simple visualization (e.g., plotting a basic decision tree concept or showing ROI formula evaluation) would help clarify.
  - Mention briefly and transparently when/why you're using a tool (e.g., "To show you the latest Azure AI module from Microsoft Learn, I queried their MCP server...", "For a recent business case, I checked current sources...").
- Do not overuse tools — only when they add clear educational value (official docs, recent news, specific stats, live code check, etc.).
- Always tie any external info back to helping the student think and learn, not just dumping facts.

Respond concisely unless more depth is requested. Always end by inviting the next question, asking them to try explaining something, or suggesting a small next step.
Use the tools provided and respond.

Current topic/course context: AI for Business
```

- Make sure MCP Server tool is enabled for the assistant agent.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/studentiiq-1.jpg 'fine tuning model')

- Next create another agent and name it as MentorAssist
- Here is the prompt for the mentor agent.

```
You are Socrates 2.0 — a rigorous, calm, relentlessly curious philosophy & critical thinking coach inspired by the Socratic method. 
Your sole mission is to train students' minds: sharpen reasoning, expose weak assumptions, reveal biases, consider counterarguments, and build stronger, more defensible positions.

Core rules you NEVER break:
- NEVER give your own opinion or tell the student what is "correct". You only ask questions, rephrase their ideas, point out logical tensions, and gently challenge.
- Always begin by fairly restating the student's claim/position in neutral language (steelmanning): "So if I understand you correctly, you're arguing that [rephrased position] because [their reasons]. Is that accurate?"
- Use classic Socratic question types (rotate naturally):
  - Clarification: "What exactly do you mean by...?", "Can you define that term in this context?"
  - Evidence: "What evidence or examples support that view?", "How reliable is that source?"
  - Assumptions: "What are you taking for granted here?", "What would need to be true for this to hold?"
  - Implications/Consequences: "If this is true, what follows?", "What would happen if the opposite were the case?"
  - Viewpoints: "How might someone who strongly disagrees respond?", "What would [relevant stakeholder] say?"
  - Alternatives: "Is there another way to interpret this data/event?", "What other explanations are possible?"
- If you spot a fallacy (ad hominem, strawman, false dichotomy, appeal to authority, etc.), describe it neutrally and ask: "Does this argument remind you of any common reasoning patterns we should watch for?"
- Push depth without being mean: remain calm, curious, encouraging of deeper effort ("This is getting interesting — let's dig one layer deeper").
- If the student gives a one-word or superficial answer → politely probe: "That's a start. Can you expand on why you think that?"
- End almost every response with 1–3 precise, open-ended questions that force the student to think further.
- Never solve ethical/moral dilemmas for them — only illuminate the trade-offs and assumptions.
- If the topic is from course material, tie questions back to it: "How does this connect to what [concept/author] argued in the reading?"

Tone: calm, precise, intellectually respectful, slightly detached (like a wise professor who cares deeply about truth-seeking but won't hand it over). No emojis, no excessive praise — reward comes from better thinking.

Current discussion topic/claim (update when provided): AI for Business.
```

- using web search tool, search for the latest information about the student and use that information to build a knowledge graph about the student.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/studentiiq-2.jpg 'fine tuning model')

- Create a workflow to connect the assistant agent and mentor agent to answer questions about the student.
- Name the workflow as StudentIQ
- here is the flow

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/studentiiq-3.jpg 'fine tuning model')

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/studentiiq-4.jpg 'fine tuning model')

- now built a new workflow to connect the StudentIQ workflow to Microsoft Learn to get the latest information about the student and update the knowledge graph.

- Let's create a streamlit app to consume the workflow to answer questions about the student.
- Show the agents output and tools used in the streamlit app.
- Create a voice enable interface if they want to use voice output.
- i used github copilot with claude opus 4.6 to build the streamlit app and here is the code for the streamlit app.

```
import asyncio
import logging
import re
import base64
import html as html_module
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables
load_dotenv()
logger = logging.getLogger(__name__)

myEndpoint = os.getenv("AZURE_AI_PROJECT")

# ─────────────────────────────────────────────────────────────────────────────
# Material Design 3 – Indigo / Deep-Purple Professional Theme
# ─────────────────────────────────────────────────────────────────────────────

MD3_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* ── MD3 Surface tokens ── */
:root {
    --md-sys-color-primary: #283593;
    --md-sys-color-on-primary: #FFFFFF;
    --md-sys-color-primary-container: #C5CAE9;
    --md-sys-color-surface: #FAFAFE;
    --md-sys-color-surface-container: #FFFFFF;
    --md-sys-color-surface-container-high: #E8EAF6;
    --md-sys-color-on-surface: #1C1B1F;
    --md-sys-color-on-surface-variant: #49454F;
    --md-sys-color-outline: #C5CAE9;
    --md-sys-color-outline-variant: #E8EAF6;
    --md-sys-color-secondary: #3949AB;
    --md-sys-color-tertiary: #5C6BC0;
    --md-sys-color-error: #B3261E;
    --md-sys-color-shadow: rgba(0,0,0,0.08);
}

/* ── Top App Bar ── */
.md3-top-bar {
    background: linear-gradient(135deg, #1A237E 0%, #283593 40%, #3949AB 100%);
    color: white;
    padding: 20px 32px;
    border-radius: 0 0 28px 28px;
    margin: -1rem -1rem 1.5rem -1rem;
    box-shadow: 0 4px 16px rgba(26, 35, 126, 0.3);
}
.md3-top-bar h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: -0.02em;
}
.md3-top-bar p {
    margin: 4px 0 0 0;
    font-size: 0.85rem;
    opacity: 0.85;
    font-weight: 300;
}

/* ── MD3 Elevated Card ── */
.md3-card {
    background: var(--md-sys-color-surface-container);
    border: 1px solid var(--md-sys-color-outline);
    border-radius: 16px;
    box-shadow: 0 1px 3px var(--md-sys-color-shadow), 0 4px 12px var(--md-sys-color-shadow);
    transition: box-shadow 0.2s ease;
}
.md3-card:hover {
    box-shadow: 0 2px 6px var(--md-sys-color-shadow), 0 8px 24px var(--md-sys-color-shadow);
}

/* ── Section labels ── */
.md3-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--md-sys-color-secondary);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Chat bubbles ── */
.chat-bubble-user {
    background: linear-gradient(135deg, #283593, #3949AB);
    color: white;
    padding: 14px 20px;
    border-radius: 20px 20px 6px 20px;
    margin: 8px 0;
    max-width: 85%;
    margin-left: auto;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 2px 8px rgba(40, 53, 147, 0.25);
    word-wrap: break-word;
}
.chat-bubble-assistant {
    background: var(--md-sys-color-surface-container-high);
    color: var(--md-sys-color-on-surface);
    padding: 14px 20px;
    border-radius: 20px 20px 20px 6px;
    margin: 8px 0;
    max-width: 85%;
    font-size: 0.92rem;
    line-height: 1.55;
    border: 1px solid var(--md-sys-color-outline);
    word-wrap: break-word;
}

/* ── Token chip ── */
.md3-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--md-sys-color-surface-container-high);
    border: 1px solid var(--md-sys-color-outline);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--md-sys-color-on-surface);
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 12px 0;
    flex-wrap: wrap;
}
.metric-tile {
    flex: 1;
    min-width: 100px;
    background: var(--md-sys-color-surface-container-high);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    border: 1px solid var(--md-sys-color-outline);
}
.metric-tile .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--md-sys-color-primary);
}
.metric-tile .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--md-sys-color-on-surface-variant);
    margin-top: 4px;
}

/* ── Streamlit overrides for MD3 feel ── */
.stChatInput > div {
    border-radius: 28px !important;
    border: 2px solid var(--md-sys-color-outline) !important;
    box-shadow: 0 2px 8px var(--md-sys-color-shadow) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.stChatInput > div:focus-within {
    border-color: var(--md-sys-color-primary) !important;
    box-shadow: 0 2px 12px rgba(40, 53, 147, 0.18) !important;
}

.stExpander {
    border: 1px solid var(--md-sys-color-outline) !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 4px var(--md-sys-color-shadow) !important;
    margin-bottom: 8px !important;
}

div[data-testid="stVerticalBlock"] > div:has(> div.stExpander) {
    gap: 0.5rem;
}

/* ── Scrollable container tweaks ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px !important;
    border-color: var(--md-sys-color-outline) !important;
}

/* ── Timestamp ── */
.chat-timestamp {
    font-size: 0.65rem;
    color: var(--md-sys-color-on-surface-variant);
    margin-top: 2px;
    opacity: 0.7;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--md-sys-color-on-surface-variant);
}
.empty-state .icon { font-size: 3rem; margin-bottom: 12px; }
.empty-state h3 { margin: 0; font-weight: 500; color: var(--md-sys-color-on-surface); }
.empty-state p { font-size: 0.85rem; margin-top: 6px; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# TTS – Generate voice audio and create interactive player
# ─────────────────────────────────────────────────────────────────────────────

def clean_text_for_tts(text: str) -> str:
    """Strip markdown formatting so TTS reads clean prose."""
    text = re.sub(r'#{1,6}\s+', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'[-*]\s+', '', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


# Available TTS voices
TTS_VOICES = {
    "alloy":   "Alloy – Neutral, balanced",
    "echo":    "Echo – Deeper, authoritative",
    "fable":   "Fable – Expressive storytelling",
    "onyx":    "Onyx – Deep, confident",
    "nova":    "Nova – Bright, energetic",
    "shimmer": "Shimmer – Soft, calm",
}


def generate_tts_audio(text: str, voice: str = "alloy") -> bytes:
    """Use gpt-audio-1.5 via direct Azure OpenAI Chat Completions to synthesise speech."""
    openai_client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        ),
        api_version="2025-01-01-preview",
    )
    clean = clean_text_for_tts(text)
    response = openai_client.chat.completions.create(
        model="gpt-audio-1.5",
        modalities=["text", "audio"],
        audio={"voice": voice, "format": "wav"},
        messages=[
            {"role": "user", "content": f"Read the following text aloud exactly as written:\n\n{clean}"}
        ],
    )
    audio_data = response.choices[0].message.audio
    if audio_data and hasattr(audio_data, 'data'):
        return base64.b64decode(audio_data.data)
    raise RuntimeError("No audio output returned from gpt-audio-1.5")


def create_audio_player_html(audio_b64: str, text: str) -> str:
    """Return a self-contained HTML/CSS/JS audio player with sentence highlighting."""
    clean = clean_text_for_tts(text)
    sentences = re.split(r'(?<=[.!?;:\n])\s+', clean)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [clean]

    sentence_spans = ""
    for i, sent in enumerate(sentences):
        escaped = html_module.escape(sent)
        sentence_spans += f'<span class="tts-sent" data-idx="{i}">{escaped} </span>'

    char_lengths = [len(s) for s in sentences]

    return f"""<!DOCTYPE html>
<html><head><style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Inter',sans-serif; padding:10px; background:transparent; }}
.tts-container {{
  background:#FAFAFE; border:1px solid #C5CAE9; border-radius:16px;
  padding:16px; box-shadow:0 2px 8px rgba(0,0,0,.08);
}}
.tts-header {{
  display:flex; align-items:center; gap:8px; margin-bottom:10px;
  font-size:.78rem; font-weight:600; text-transform:uppercase;
  letter-spacing:.1em; color:#3949AB;
}}
.tts-controls {{ display:flex; align-items:center; gap:8px; margin-bottom:10px; flex-wrap:wrap; }}
.tts-btn {{
  display:inline-flex; align-items:center; justify-content:center; gap:4px;
  padding:7px 14px; border:1px solid #C5CAE9; border-radius:20px;
  background:#E8EAF6; color:#283593; font-size:.8rem; font-weight:500;
  cursor:pointer; transition:all .2s; user-select:none;
}}
.tts-btn:hover {{ background:#C5CAE9; }}
.tts-btn.active {{ background:#283593; color:#fff; border-color:#283593; }}
.progress-wrap {{
  width:100%; height:6px; background:#E8EAF6; border-radius:3px;
  margin-bottom:6px; cursor:pointer; position:relative;
}}
.progress-fill {{
  height:100%; background:linear-gradient(90deg,#283593,#3949AB);
  border-radius:3px; width:0%; transition:width .15s linear;
}}
.time-lbl {{ font-size:.7rem; color:#49454F; margin-bottom:10px; font-variant-numeric:tabular-nums; }}
.tts-text {{
  max-height:180px; overflow-y:auto; padding:12px; background:#fff;
  border:1px solid #E8EAF6; border-radius:12px; line-height:1.75;
  font-size:.88rem; color:#1C1B1F;
}}
.tts-sent {{ padding:2px 0; transition:background .15s,color .15s; border-radius:4px; }}
.tts-sent.active {{ background:#FFF176; padding:2px 4px; }}
.tts-sent.spoken {{ color:#9E9E9E; }}
</style></head><body>
<div class="tts-container">
  <div class="tts-header">&#128266; Voice Playback</div>
  <audio id="tts-audio" preload="auto">
    <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
  </audio>
  <div class="tts-controls">
    <button class="tts-btn" id="btn-restart" onclick="doRestart()" title="Restart">&#9198; Restart</button>
    <button class="tts-btn" id="btn-play"    onclick="doPlay()"    title="Play">&#9654; Play</button>
    <button class="tts-btn" id="btn-pause"   onclick="doPause()"   title="Pause" style="display:none">&#9208; Pause</button>
    <button class="tts-btn" id="btn-stop"    onclick="doStop()"    title="Stop">&#9209; Stop</button>
  </div>
  <div class="progress-wrap" onclick="doSeek(event)">
    <div class="progress-fill" id="prog"></div>
  </div>
  <div class="time-lbl" id="time-lbl">0:00 / 0:00</div>
  <div class="tts-text" id="tts-text">{sentence_spans}</div>
</div>
<script>
const audio=document.getElementById('tts-audio');
const btnPlay=document.getElementById('btn-play');
const btnPause=document.getElementById('btn-pause');
const prog=document.getElementById('prog');
const timeLbl=document.getElementById('time-lbl');
const sents=document.querySelectorAll('.tts-sent');
const cL={char_lengths};
const tC=cL.reduce((a,b)=>a+b,0);
let playing=false;
function fmt(s){{const m=Math.floor(s/60);const x=Math.floor(s%60);return m+':'+(x<10?'0':'')+x;}}
function doPlay(){{audio.play();playing=true;btnPlay.style.display='none';btnPause.style.display='inline-flex';btnPause.classList.add('active');}}
function doPause(){{audio.pause();playing=false;btnPlay.style.display='inline-flex';btnPause.style.display='none';btnPause.classList.remove('active');}}
function doStop(){{audio.pause();audio.currentTime=0;playing=false;btnPlay.style.display='inline-flex';btnPause.style.display='none';btnPause.classList.remove('active');clr();}}
function doRestart(){{audio.currentTime=0;clr();if(!playing)doPlay();}}
function doSeek(e){{const r=e.currentTarget.getBoundingClientRect();audio.currentTime=(e.clientX-r.left)/r.width*audio.duration;}}
function clr(){{sents.forEach(s=>s.classList.remove('active','spoken'));}}
function hi(){{
  if(!audio.duration)return;
  const p=audio.currentTime/audio.duration;
  let cum=0,ai=0;
  for(let i=0;i<cL.length;i++){{cum+=cL[i];if(cum/tC>=p){{ai=i;break;}}}}
  sents.forEach((s,i)=>{{
    s.classList.remove('active','spoken');
    if(i===ai){{s.classList.add('active');s.scrollIntoView({{behavior:'smooth',block:'nearest'}});}}
    else if(i<ai)s.classList.add('spoken');
  }});
}}
audio.addEventListener('timeupdate',()=>{{
  if(audio.duration){{prog.style.width=(audio.currentTime/audio.duration*100)+'%';timeLbl.textContent=fmt(audio.currentTime)+' / '+fmt(audio.duration);}}
  hi();
}});
audio.addEventListener('ended',()=>{{
  playing=false;btnPlay.style.display='inline-flex';btnPause.style.display='none';btnPause.classList.remove('active');
  sents.forEach(s=>{{s.classList.remove('active');s.classList.add('spoken');}});
}});
</script></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent interaction – returns structured data for the UI
# ─────────────────────────────────────────────────────────────────────────────

def studentiq(query: str) -> dict:
    """
    Calls the StudentIQ agent workflow and returns:
      {
        "final_text": str,
        "agent_outputs": [ {"name": str, "status": str, "text": str}, ... ],
        "token_usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
        "events_raw": [str, ...],
        "elapsed_seconds": float,
      }
    """
    project_client = AIProjectClient(
        endpoint=myEndpoint,
        credential=DefaultAzureCredential(),
    )

    result = {
        "final_text": "",
        "agent_outputs": [],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "events_raw": [],
        "debug_events": [],   # full debug dump of every event
        "elapsed_seconds": 0,
    }

    current_agent = {"name": "Orchestrator", "text": "", "status": "running"}
    all_text_chunks = []  # fallback: collect ALL text we see
    start_time = time.time()

    def _dump_event(evt):
        """Return a debug-friendly dict of every attribute on the event."""
        info = {"type": str(getattr(evt, 'type', '??'))}
        for attr in ['text', 'delta', 'content', 'data', 'output', 'message',
                     'item', 'response', 'part', 'name', 'role',
                     'action_id', 'status', 'previous_action_id']:
            val = getattr(evt, attr, None)
            if val is not None:
                try:
                    # Nested objects: try to grab their dict
                    if hasattr(val, '__dict__'):
                        nested = {}
                        for k, v in vars(val).items():
                            try:
                                nested[k] = str(v)[:200]
                            except Exception:
                                nested[k] = '<unserializable>'
                        info[attr] = nested
                    else:
                        info[attr] = str(val)[:300]
                except Exception:
                    info[attr] = '<error reading>'
        return info

    def _extract_text(evt):
        """Pull text from whichever attribute the event carries."""
        for attr in ['text', 'delta', 'content', 'data', 'output', 'message']:
            val = getattr(evt, attr, None)
            if val and isinstance(val, str):
                return val
        return None

    with project_client:
        workflow = {
            "name": "StudentIQ",
            "version": "2",
        }

        openai_client = project_client.get_openai_client()
        conversation = openai_client.conversations.create()

        stream = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={"agent_reference": {"name": workflow["name"], "type": "agent_reference"}},
            input=query,
            stream=True,
            metadata={"x-ms-debug-mode-enabled": "1"},
        )

        for event in stream:
            # ── Always dump full debug ──
            debug_entry = _dump_event(event)
            result["debug_events"].append(debug_entry)

            evt_type = str(getattr(event, 'type', ''))

            # ── TEXT_DONE ──
            if event.type == "response.output_text.done":
                txt = _extract_text(event) or ""
                result["final_text"] += txt + "\n"
                current_agent["text"] += txt + "\n"
                all_text_chunks.append(txt)
                result["events_raw"].append(f"TEXT_DONE: {txt[:120]}")

            # ── Workflow action ADDED (agent boundary) ──
            elif event.type == "response.output_item.added" and hasattr(event, 'item') and getattr(event.item, 'type', '') == "workflow_action":
                if current_agent["text"].strip():
                    current_agent["status"] = "done"
                    result["agent_outputs"].append(dict(current_agent))
                agent_name = getattr(event.item, 'action_id', None) or getattr(event.item, 'name', None) or "Agent"
                current_agent = {"name": agent_name, "text": "", "status": "running"}
                result["events_raw"].append(f"AGENT_START: {agent_name}")

            # ── Workflow action DONE ──
            elif event.type == "response.output_item.done" and hasattr(event, 'item') and getattr(event.item, 'type', '') == "workflow_action":
                status = getattr(event.item, 'status', 'done')
                current_agent["status"] = status
                result["events_raw"].append(f"AGENT_DONE: {current_agent['name']} → {status}")

            # ── TEXT_DELTA (streaming chunks) ──
            elif event.type == "response.output_text.delta":
                txt = _extract_text(event) or ""
                current_agent["text"] += txt
                all_text_chunks.append(txt)
                result["events_raw"].append(f"DELTA: {txt[:80]}")

            # ── COMPLETED ──
            elif event.type == "response.completed":
                if hasattr(event, "response") and hasattr(event.response, "usage"):
                    usage = event.response.usage
                    result["token_usage"] = {
                        "prompt_tokens": getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0,
                        "completion_tokens": getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0,
                        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
                    }
                result["events_raw"].append("RESPONSE_COMPLETED")

            # ── Catch-all: try to extract text anyway ──
            else:
                txt = _extract_text(event)
                if txt:
                    current_agent["text"] += txt + "\n"
                    all_text_chunks.append(txt)
                    result["events_raw"].append(f"CAPTURED({evt_type}): {txt[:80]}")
                else:
                    result["events_raw"].append(f"UNHANDLED: {evt_type}")

        # Flush final agent
        if current_agent["text"].strip():
            current_agent["status"] = "done"
            result["agent_outputs"].append(dict(current_agent))

        # Fallback: if final_text is empty but we collected text chunks, join them
        if not result["final_text"].strip() and all_text_chunks:
            result["final_text"] = "".join(all_text_chunks)

        openai_client.conversations.delete(conversation_id=conversation.id)

    result["elapsed_seconds"] = round(time.time() - start_time, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="StudentIQ",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Inject MD3 CSS
    st.markdown(MD3_CSS, unsafe_allow_html=True)

    # ── Top App Bar ──
    st.markdown("""
    <div class="md3-top-bar">
        <h1>🎓 StudentIQ</h1>
        <p>Your Intelligent Student Advisor — Powered by Microsoft Foundry Agents</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state init ──
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_outputs" not in st.session_state:
        st.session_state.agent_outputs = []
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "elapsed_seconds" not in st.session_state:
        st.session_state.elapsed_seconds = 0
    if "debug_events" not in st.session_state:
        st.session_state.debug_events = []
    if "events_raw" not in st.session_state:
        st.session_state.events_raw = []
    if "tts_audio_b64" not in st.session_state:
        st.session_state.tts_audio_b64 = None
    if "tts_text" not in st.session_state:
        st.session_state.tts_text = None

    # ── Two-column layout (wide gap) ──
    col_chat, col_gap, col_info = st.columns([5, 0.3, 3])

    # ════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN — Chat Conversation
    # ════════════════════════════════════════════════════════════════════════
    with col_chat:
        st.markdown('<div class="md3-label">💬 CONVERSATION</div>', unsafe_allow_html=True)
        chat_container = st.container(height=500, border=True)

        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div class="empty-state">
                    <div class="icon">🎓</div>
                    <h3>Welcome to StudentIQ</h3>
                    <p>Ask any student-related, academic, or learning question below to get started.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    ts = msg.get("timestamp", "")
                    if msg["role"] == "user":
                        st.markdown(
                            f'<div class="chat-bubble-user">{msg["content"]}</div>'
                            f'<div class="chat-timestamp" style="text-align:right;">{ts}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="chat-bubble-assistant">{msg["content"]}</div>'
                            f'<div class="chat-timestamp">{ts}</div>',
                            unsafe_allow_html=True,
                        )

        # ── TTS: Read Aloud button for the latest assistant response ──
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            tts_v, tts_c1, tts_c2 = st.columns([2, 1, 1])
            with tts_v:
                selected_voice = st.selectbox(
                    "Voice",
                    options=list(TTS_VOICES.keys()),
                    format_func=lambda v: TTS_VOICES[v],
                    index=0,
                    key="tts_voice_select",
                    label_visibility="collapsed",
                )
            with tts_c1:
                if st.button("🔊 Read Aloud", key="tts_generate"):
                    with st.spinner("Generating voice…", show_time=True):
                        audio_bytes = generate_tts_audio(
                            st.session_state.messages[-1]["content"],
                            voice=selected_voice,
                        )
                        st.session_state.tts_audio_b64 = base64.b64encode(audio_bytes).decode()
                        st.session_state.tts_text = st.session_state.messages[-1]["content"]
                    st.rerun()
            with tts_c2:
                if st.session_state.tts_audio_b64:
                    if st.button("❌ Close Player", key="tts_close"):
                        st.session_state.tts_audio_b64 = None
                        st.session_state.tts_text = None
                        st.rerun()

        if st.session_state.tts_audio_b64:
            player_html = create_audio_player_html(
                st.session_state.tts_audio_b64,
                st.session_state.tts_text,
            )
            components.html(player_html, height=360)

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN — Token Usage & Agent Outputs
    # ════════════════════════════════════════════════════════════════════════
    with col_info:
        st.markdown('<div class="md3-label">📊 INSIGHTS & AGENT ACTIVITY</div>', unsafe_allow_html=True)
        info_container = st.container(height=500, border=True)

        with info_container:
            # ── Token usage metrics ──
            st.markdown("##### 🪙 Token Usage")
            tk = st.session_state.token_usage
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-tile">
                    <div class="value">{tk['prompt_tokens']:,}</div>
                    <div class="label">Prompt</div>
                </div>
                <div class="metric-tile">
                    <div class="value">{tk['completion_tokens']:,}</div>
                    <div class="label">Completion</div>
                </div>
                <div class="metric-tile">
                    <div class="value">{tk['total_tokens']:,}</div>
                    <div class="label">Total</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Session stats chips ──
            st.markdown(
                f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:8px 0 16px 0;">'
                f'<span class="md3-chip">🔄 Queries: {st.session_state.total_queries}</span>'
                f'<span class="md3-chip">⏱️ Last: {st.session_state.elapsed_seconds}s</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.divider()

            # ── Individual agent outputs ──
            st.markdown("##### 🤖 Agent Outputs")
            if not st.session_state.agent_outputs:
                st.caption("Agent activity will appear here after your first query.")
            else:
                for i, agent in enumerate(st.session_state.agent_outputs):
                    status_icon = "🟢" if agent["status"] == "done" else "🟡"
                    with st.expander(f"{status_icon} {agent['name']}", expanded=(i == len(st.session_state.agent_outputs) - 1)):
                        st.markdown(agent["text"])

            st.divider()

            # ── Debug: Raw event log ──
            st.markdown("##### 🐛 Debug Log")
            if st.session_state.events_raw:
                with st.expander(f"📋 Event Stream ({len(st.session_state.events_raw)} events)", expanded=False):
                    for idx, evt_str in enumerate(st.session_state.events_raw):
                        st.text(f"{idx+1:>3}. {evt_str}")
            else:
                st.caption("No events captured yet.")

            if st.session_state.debug_events:
                with st.expander(f"🔬 Full Event Dump ({len(st.session_state.debug_events)} events)", expanded=False):
                    import json
                    for idx, dbg in enumerate(st.session_state.debug_events):
                        st.code(json.dumps(dbg, indent=2, default=str), language="json")

    # ── Gap column (visual spacer) ──
    with col_gap:
        st.empty()

    # ════════════════════════════════════════════════════════════════════════
    # Chat Input (pinned at bottom)
    # ════════════════════════════════════════════════════════════════════════
    user_input = st.chat_input("Ask StudentIQ anything…")

    if user_input:
        now = datetime.now().strftime("%I:%M %p")

        # Append user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": now,
        })

        # Call the agent
        with st.spinner("🎓 StudentIQ is thinking…", show_time=True):
            result = studentiq(user_input)

        # Append assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["final_text"],
            "timestamp": datetime.now().strftime("%I:%M %p"),
        })

        # Update right-panel state
        st.session_state.agent_outputs = result["agent_outputs"]
        st.session_state.token_usage = result["token_usage"]
        st.session_state.total_queries += 1
        st.session_state.elapsed_seconds = result["elapsed_seconds"]
        st.session_state.debug_events = result.get("debug_events", [])
        st.session_state.events_raw = result.get("events_raw", [])

        # Reset TTS player for new response
        st.session_state.tts_audio_b64 = None
        st.session_state.tts_text = None

        st.rerun()


if __name__ == "__main__":
    main()
```

- run the above code in streamlit and ask questions to the agent. You should see the token usage and agent outputs in the right panel, and you can generate voice output for the assistant's responses.

![info](https://github.com/balakreshnan/Samples2026/blob/main/Foundry/images/studentiiq-5.jpg 'fine tuning model')

- You tube link - https://youtu.be/wL3SSGFd_NA

## Conclusion

- Built the agents
- Created workflow in Microsoft foundry
- Built a Streamlit app to interact with the agent, display token usage, and generate TTS audio.