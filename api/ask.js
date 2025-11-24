// api/ask.js - Vercel Serverless (Node 18+)
// Versão minimalista para o MVP DIY.
// Requisitos: definir OPENAI_API_KEY nas env vars do Vercel.
// Antes do deploy, garante que os ficheiros em /data têm embeddings (ver script gen-embeddings.js).

import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_DIR = path.join(process.cwd(), 'data');

// util: carregar docs JSON da pasta data
function loadDocsSync() {
  if(!fs.existsSync(DATA_DIR)) return [];
  const files = fs.readdirSync(DATA_DIR).filter(f=>f.endsWith('.json'));
  const docs = files.map(f=>{
    try {
      const content = JSON.parse(fs.readFileSync(path.join(DATA_DIR, f), 'utf8'));
      return content;
    } catch(e) { return null; }
  }).filter(Boolean);
  return docs;
}

// cosine similarity
function cosine(a,b) {
  let dot=0, na=0, nb=0;
  for(let i=0;i<a.length;i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
  return dot / (Math.sqrt(na)*Math.sqrt(nb) + 1e-12);
}

export default async function handler(req, res) {
  try {
    if(req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
    const body = req.body || {};
    const question = (body.question || '').trim();
    if(!question) return res.status(400).json({ error: 'Pergunta vazia' });

    // load docs
    const docs = loadDocsSync();
    if(docs.length === 0) return res.status(500).json({ error: 'Sem documentos indexados em /data' });

    // criar embedding da pergunta
    const embResp = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: question
    });
    const qEmb = embResp.data[0].embedding;

    // calcular similaridade com cada doc
    const scored = docs.map(d => {
      if(!d.embedding) return { d, score: -1 };
      const s = cosine(qEmb, d.embedding);
      return { d, score: s };
    }).filter(x=>x.score>=0);

    scored.sort((a,b)=>b.score - a.score);
    const top = scored.slice(0,4).filter(s => s.score > 0.12); // threshold mínimo

    if(top.length === 0) {
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

    // Chamada ao modelo (ajusta o model se necessário)
    const chatResp = await openai.chat.completions.create({
      model: 'gpt-4o-mini', // podes alterar para outro modelo se preferires
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
  } catch(err) {
    console.error('Error /api/ask', err);
    return res.status(500).json({ error: err.message || 'Erro no servidor' });
  }
}

