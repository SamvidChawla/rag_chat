import { GoogleGenAI } from '@google/genai'
import { dataset } from './dataset.js'
import 'dotenv/config'

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

async function getRagAnswer(question) {
  const res = await fetch('http://localhost:3000/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: question }),
  });
  const data = await res.json();
  return data.answer;
}

async function judge(question, expected, predicted) {
  const prompt = `You are an eval judge. Decide if the predicted answer is correct.

Rules:
- Out-of-scope questions (expected contains "does not mention" / "does not specify"): predicted MUST say the info is unavailable. Any fabricated detail = FAIL.
- Factual questions: must be correct AND complete. Answering "Yes" without a required condition (e.g. supervisor awareness) = FAIL.
- Exact wording not required. Semantically equivalent = PASS.

Question: ${question}
Expected: ${expected}
Predicted: ${predicted}

Reply with exactly one word: PASS or FAIL.`;

  const response = await ai.models.generateContent({
    model: 'gemini-3.1-flash-lite',
    contents: prompt,
  });

  return response.text.trim().toUpperCase().startsWith('PASS');
}

async function runEval() {
  let passed = 0;
  const failures = [];

  for (const item of dataset) {
    const predicted = await getRagAnswer(item.question);
    const pass = await judge(item.question, item.expected, predicted);

    if (pass) {
      passed++;
      console.log(`[PASS] Q${item.id}`);
    } else {
      failures.push(item.id);
      console.log(`[FAIL] Q${item.id}`);
      console.log(`  Expected:  ${item.expected}`);
      console.log(`  Predicted: ${predicted}`);
    }
    await new Promise(r => setTimeout(r, 9000));
  }

  console.log(`\nFinal Score: ${passed}/${dataset.length}`);
  if (failures.length) {
    console.log(`Failed IDs: ${failures.join(', ')}`);
  }
}

runEval();