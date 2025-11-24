// api/ask.js  — Vercel serverless (Node 18+)
// Gera embeddings automaticamente em memória (opção A).
// Requisitos: definir OPENAI_API_KEY nas env vars do Vercel.

import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_DIR = path.join(process.cwd(), 'data');

// util: cosine similarity
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

// Carrega e prepara docs (gera embeddings se null). Mantém em memória (global).
async function prepareDocs() {
  // reutiliza se já existe (mantém embeddings em memória enquanto a instância estiver warm)
  if (global.__PDQ_DOCS && Array.isArray(global.__PDQ_DOCS) && global.__PDQ_DOCS.length) return global.__PDQ_DOCS;

  const docs = [];
  if (!fs.existsSync(DATA_DIR)) {
    console.warn('data/ não existe — cria a pasta e insere JSONs com {id,title,url,text,embedding:null}');
    global.__PDQ_DOCS = [];
    return global.__PDQ_DOCS;
  }

  const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.json'));
  for (const f of files) {
    try {
      const content = JSON.parse(fs.readFileSync(path.join(DATA_DIR, f), 'utf8'));
      // valida campos mínimos
      if (!content.id || !content.title || !content.url || !content.text) continue;
      docs.push(content);
    } catch (e) {
      console.warn('ERRO ao ler', f, e.message);
    }
  }

  // gerar embeddings para os docs que têm embedding null ou muito curtas
  const docsNeeding = docs.filter(d => !Array.isArray(d.embedding) || d.embedding.length < 10);
  if (docsNeeding.length > 0) {
    // Atenção: se tens muitos docs, isto pode demorar. Mantém o número reduzido.
    for (const d of docsNeeding) {
      try {
        const embResp = await openai.embeddings.create({
          model: 'text-embedding-3-small',
          input: d.text
        });
        d.embedding = embResp.data[0].embedding;
      } catch (err) {
        console.error('Erro a gerar embedding para', d.id, err.message || err);
        d.embedding = null;
      }
    }
  }

  // guarda em memória global (persistirá enquanto a instancia Vercel estiver warm)
  global.__PDQ_DOCS = docs;
  return global.__PDQ_DOCS;
}

export default async function handler(req, res) {
  try {
    if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed (use POST)' });
    const { question } = req.body || {};
    if (!question || typeof question !== 'string' || !question.trim()) return res.status(400).json({ error: 'Pergunta vazia' });

    // prepara docs (gera embeddings se necessário)
    const docs = await prepareDocs();
    if (!docs || docs.length === 0) return res.status(500).json({ error: 'Sem documentos indexados em /data' });

    // criar embedding da pergunta
    const embResp = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: question
    });
    const qEmb = embResp.data[0].embedding;

    // similaridade
    const scored = [];
    for (const d of docs) {
      if (!d.embedding || !Array.isArray(d.embedding)) continue;
      const s = cosine(qEmb, d.embedding);
      scored.push({ d, score: s });
    }
    scored.sort((a, b) => b.score - a.score);
    const top = scored.slice(0, 4).filter(x => x.score > 0.12); // threshold mínimo

    if (top.length === 0) {
      return res.json({ answer: 'Não encontrei resposta nas fontes oficiais indexadas.', sources: [] });
    }

    // construir contexto para o LLM
    const contextText = top.map(t => `URL: ${t.d.url}\nTITLE: ${t.d.title}\nEXCERPT: ${t.d.text}`).join('\n\n----\n\n');

    const systemPrompt = `
És um assistente que responde exclusivamente em Português de Portugal.
Só podes usar as informações textuais fornecidas nos excertos abaixo — não inventes.
Se a resposta não estiver nos excertos, responde exactamente: "Não encontrei resposta nas fontes oficiais indexadas."
No fim da resposta inclui uma secção "Fontes:" com as URLs utilizadas.
Mantém linguagem clara, concisa e sem jargão desnecessário.
`.trim();

    const userPrompt = `Pergunta: ${question}\n\nContexto:\n${contextText}\n\nResponde em pt-PT.`;

    // chamada ao modelo
    const chatResp = await openai.chat.completions.create({
      model: 'gpt-4o-mini', // podes ajustar se preferires outro modelo
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt }
      ],
      temperature: 0.0,
      max_tokens: 600
    });

    const answer = chatResp.choices[0].message.content;
    const sources = top.map(t => ({ title: t.d.title, url: t.d.url }));

    return res.json({ answer, sources });
  } catch (err) {
    console.error('Erro /api/ask', err);
    return res.status(500).json({ error: err.message || 'Erro no servidor' });
  }
}
