// api/ask.js — com CORS integrado
import fs from 'fs';
import path from 'path';
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_DIR = path.join(process.cwd(), 'data');

// CORS --------------------------------------------------------
function setCors(res) {
  res.setHeader("Access-Control-Allow-Origin", "https://politicadebolso.pt");  
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
}

// --------------------------------------------------------------
export default async function handler(req, res) {

  // Responder pré-flight (browser manda sempre antes do POST)
  if (req.method === "OPTIONS") {
    setCors(res);
    return res.status(204).end();
  }

  // Aplica CORS a TODAS as respostas
  setCors(
