/**
 * server/server.js
 * --------------------------------------------
 * Express proxy for OpenAI-powered AI bidding (Responses API).
 *
 * Endpoints:
 *  - GET  /health
 *  - POST /api/ai/bid     : returns ai_bid + expected range + thought
 *  - POST /api/ai/report  : returns end-of-game report (text)
 *
 * Setup:
 *  1) cd server
 *  2) npm init -y
 *  3) npm i express cors dotenv openai
 *  4) create server/.env with:
 *       OPENAI_API_KEY=sk-...
 *       PORT=3001
 *       OPENAI_MODEL=gpt-4o-mini
 *       REQUEST_TIMEOUT_MS=4500
 *
 * Run:
 *   node server.js
 */

import "dotenv/config";
import express from "express";
import cors from "cors";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(express.json({ limit: "1mb" }));

const PORT = Number(process.env.PORT || 3001);
const MODEL = process.env.OPENAI_MODEL || "gpt-4o-mini";
const TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS || 4500);

if (!process.env.OPENAI_API_KEY) {
  console.error("❌ Missing OPENAI_API_KEY. Put it in server/.env or env vars.");
  process.exit(1);
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ------------------ helpers ------------------
function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function roundInt(x, fallback = 0) {
  const n = Number(x);
  return Number.isFinite(n) ? Math.round(n) : fallback;
}

/**
 * Robustly extract text from Responses API result.
 * Different SDK versions/models may populate different fields.
 */
function extractOutputText(resp) {
  if (!resp) return "";
  if (typeof resp.output_text === "string" && resp.output_text.trim()) return resp.output_text.trim();

  const out = resp.output;
  if (Array.isArray(out)) {
    let buf = "";
    for (const item of out) {
      const content = item && item.content;
      if (!Array.isArray(content)) continue;
      for (const c of content) {
        if (c && typeof c.text === "string") buf += c.text;
      }
    }
    return buf.trim();
  }
  return "";
}

/**
 * Promise timeout (no AbortController param leakage).
 */
async function withTimeout(promise, ms) {
  let t;
  const timeout = new Promise((_, rej) => {
    t = setTimeout(() => rej(new Error("Request was aborted.")), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    clearTimeout(t);
  }
}

/**
 * Basic state sanitizer: keep payload small & predictable.
 * We only pass what the AI needs, and cap history length.
 */
function sanitizeState(raw) {
  const state = raw?.state || raw;

  const round = roundInt(state?.round, 1);
  const totalRounds = roundInt(state?.totalRounds, 10);

  const playerBudget = roundInt(state?.player?.budget, 0);
  const playerScore = roundInt(state?.player?.score, 0);
  const lockedBid = state?.player?.lockedBid ?? null;

  const aiBudget = roundInt(state?.ai?.budget, 0);
  const aiScore = roundInt(state?.ai?.score, 0);

  const stageId = roundInt(state?.stage?.id, 1);
  const stageDiff = Number(state?.stage?.diff ?? 1.0);
  const stageName = String(state?.stage?.name || "");
  const stageProfileRaw = state?.stage?.profile || {};
  const stageProfile = {
    style: String(stageProfileRaw.style || ""),
    aggression: Number.isFinite(stageProfileRaw.aggression) ? stageProfileRaw.aggression : 1,
    bluff: Number.isFinite(stageProfileRaw.bluff) ? stageProfileRaw.bluff : 0.1,
    conserve: Number.isFinite(stageProfileRaw.conserve) ? stageProfileRaw.conserve : 0.5,
    spendCeil: Number.isFinite(stageProfileRaw.spendCeil) ? stageProfileRaw.spendCeil : 0.25,
  };

  const history = Array.isArray(state?.history) ? state.history.slice(-8) : [];

  return {
    round,
    totalRounds,
    stage: { id: stageId, diff: stageDiff, name: stageName, profile: stageProfile },
    player: {
      budget: playerBudget,
      score: playerScore,
      lockedBid: lockedBid === null ? null : roundInt(lockedBid, null),
    },
    ai: { budget: aiBudget, score: aiScore },
    history: history.map((h) => ({
      round: roundInt(h?.round, 0),
      playerBid: roundInt(h?.playerBid, 0),
      aiBid: roundInt(h?.aiBid, 0),
      result: String(h?.result ?? ""),
      expectedCenter: roundInt(h?.expectedCenter, 0),
      expectedLow: roundInt(h?.expectedLow, 0),
      expectedHigh: roundInt(h?.expectedHigh, 0),
    })),
  };
}

// ---------- adaptive AI helpers ----------
function makeRng(seed) {
  let v = seed % 2147483647;
  if (v <= 0) v += 2147483646;
  return () => {
    v = (v * 16807) % 2147483647;
    return (v - 1) / 2147483646;
  };
}

function computePlayerModel(clean, rng) {
  const N = 8;
  const history = Array.isArray(clean.history) ? clean.history.slice(-N) : [];
  const bids = history.map((h) => roundInt(h?.playerBid, 0)).filter((b) => Number.isFinite(b));
  const locks = history.map((h) => roundInt(h?.lockedBid ?? h?.playerBid, 0)).filter((b) => Number.isFinite(b));

  const arr = bids.length ? bids : locks;
  if (!arr.length) {
    const fallback = Math.max(4, Math.round(clean.player.budget * 0.15));
    return { mu: fallback, sigma: 8, conf: 0.25, samples: 0, pred: fallback };
  }

  // EMA
  const alpha = 2 / (arr.length + 1);
  let mu = arr[0];
  for (let i = 1; i < arr.length; i++) mu = alpha * arr[i] + (1 - alpha) * mu;

  const meanVal = arr.reduce((a, b) => a + b, 0) / arr.length;
  const variance = arr.reduce((a, b) => a + Math.pow(b - meanVal, 2), 0) / arr.length;
  const sigma = Math.max(3, Math.sqrt(variance));

  const predNoise = rng();
  const sample = mu + (predNoise - 0.5) * sigma * 1.2;
  const pred = clamp(Math.round(sample), 0, clean.player.budget);

  const sizeFactor = Math.min(1, arr.length / 8);
  const spreadFactor = clamp(1 - sigma / 30, 0, 1);
  const conf = clamp(0.3 * sizeFactor + 0.7 * spreadFactor, 0, 1);

  return { mu, sigma, conf, samples: arr.length, pred };
}

function pickMode(clean, model, rng) {
  const history = Array.isArray(clean.history) ? clean.history : [];
  const last2 = history.slice(-2).map((h) => h?.result);
  const aiLostTwice = last2.length === 2 && last2.every((r) => r === "player");

  const lowBudget = clean.ai.budget < Math.max(15, clean.player.budget * 0.6);
  const stablePlayer = model.sigma < 6 && model.conf > 0.45;

  let wSaver = 0.4,
    wSniper = 0.3,
    wBully = 0.3;
  if (aiLostTwice) wBully += 0.2;
  if (lowBudget) {
    wSaver += 0.3;
    wBully -= 0.1;
  }
  if (stablePlayer) {
    wSniper += 0.2;
    wSaver -= 0.1;
  }

  wSaver = Math.max(0.05, wSaver);
  wSniper = Math.max(0.05, wSniper);
  wBully = Math.max(0.05, wBully);

  const total = wSaver + wSniper + wBully;
  const r = rng() * total;
  if (r < wSaver) return "saver";
  if (r < wSaver + wSniper) return "sniper";
  return "bully";
}

// V2 Utility AI: predictive, opponent-aware, variety
function computeAdaptiveBid(clean) {
  const seed =
    clean.round * 7919 +
    clean.ai.budget * 37 +
    clean.player.budget * 17 +
    clean.ai.score * 131 +
    clean.player.score * 191;
  const rng = makeRng(seed);

  const history = Array.isArray(clean.history) ? clean.history.slice(-8) : [];
  const bids = history.map((h) => roundInt(h?.playerBid, 0)).filter((b) => Number.isFinite(b));

  const mu = bids.length ? bids.reduce((a, b) => a + b, 0) / bids.length : Math.max(6, Math.round(clean.player.budget * 0.15));
  const variance = bids.length ? bids.reduce((a, b) => a + Math.pow(b - mu, 2), 0) / bids.length : 64;
  const sigma = Math.max(3, Math.sqrt(variance));

  // Normal CDF approx (Abramowitz-Stegun)
  function normCdf(x, m, s) {
    if (s <= 0) return x > m ? 1 : 0;
    const z = (x - m) / (s * Math.SQRT2);
    const t = 1 / (1 + 0.3275911 * Math.abs(z));
    const a1 = 0.254829592,
      a2 = -0.284496736,
      a3 = 1.421413741,
      a4 = -1.453152027,
      a5 = 1.061405429;
    const erf = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-z * z);
    const cdf = 0.5 * (1 + (z >= 0 ? erf : -erf));
    return cdf;
  }

  function pWin(b) {
    return normCdf(b - 0.0001, mu, sigma);
  }

  const roundsLeftLocal = Math.max(1, clean.totalRounds - clean.round + 1);
  const aiBudget = Math.max(0, clean.ai.budget);
  const pBudget = Math.max(0, clean.player.budget);
  const scoreGap = clean.ai.score - clean.player.score;

  const lockedWinIfAIWins = isLockWinIfAIWins(clean);
  const lockedWinIfPlayerWins = isLockWinIfPlayerWins(clean);
  const late = roundsLeftLocal <= 2;
  const behind = scoreGap < 0;

  let valueWin = 10;
  if (lockedWinIfAIWins) valueWin = 30;
  else if (lockedWinIfPlayerWins && roundsLeftLocal <= 3) valueWin = 24;
  else if (late && behind) valueWin = 20;
  else if (behind) valueWin = 14;

  const stageAgg = clean?.stage?.profile?.aggression ?? 1;
  const stageCons = clean?.stage?.profile?.conserve ?? 0.5;

  let costWeight = 0.1 + stageCons * 0.35 - (behind ? 0.04 : 0) + (scoreGap > 1 ? 0.04 : 0);
  costWeight *= 1 / Math.max(0.75, Math.min(1.35, stageAgg));
  costWeight = clamp(costWeight, 0.04, 0.28);

  const pacing = aiBudget / roundsLeftLocal;
  const candidates = new Set();
  for (let k = -2; k <= 2; k++) candidates.add(clamp(Math.round(mu + k * sigma), 0, aiBudget));
  [0, 1, 2, 3, 4, 5].forEach((b) => candidates.add(b));
  [pacing, pacing * 1.5, pacing * 2.0].forEach((v) => candidates.add(clamp(Math.round(v), 0, aiBudget)));
  candidates.add(clamp(Math.round(mu + 1), 0, aiBudget));

  let best = { bid: 0, util: -Infinity, p: 0 };
  for (const b of candidates) {
    const winP = pWin(b);
    const util = winP * valueWin - costWeight * b;
    if (util > best.util) best = { bid: b, util, p: winP };
  }

  let bid = best.bid;
  let winProb = best.p;

  // variety
  const predictable = sigma < 6;
  if (predictable && rng() < 0.35) bid = clamp(Math.round(mu + 1), 0, aiBudget);

  const historyAI = history.map((h) => roundInt(h?.aiBid, 0)).filter((n) => Number.isFinite(n));
  if (historyAI.length >= 2) {
    const last = historyAI[historyAI.length - 1];
    const prev = historyAI[historyAI.length - 2];
    if (last === prev && !lockedWinIfAIWins && !late) {
      const delta = Math.round((rng() - 0.5) * 4); // ±2
      bid = clamp(bid + delta, 0, aiBudget);
    }
  }

  // bluff
  const lastRes = history.length ? history[history.length - 1]?.result : null;
  const lastWasOverbidLoss =
    lastRes === "player" && history.length
      ? (history[history.length - 1].aiBid ?? 0) > (history[history.length - 1].playerBid ?? 0)
      : false;

  const allowBluff = !lastWasOverbidLoss;
  const bluffRoll = rng();

  if (allowBluff && bluffRoll < 0.08) {
    bid = clamp(Math.round(mu * 0.35), 0, aiBudget);
  } else if (allowBluff && bluffRoll < 0.16 && valueWin >= 14) {
    const spike = Math.max(bid, mu + 4, pacing * 1.4);
    bid = clamp(Math.round(spike), 0, aiBudget);
  }

  const band = Math.max(4, Math.round(sigma));
  const expectedLow = clamp(Math.round(mu - band), 0, pBudget);
  const expectedHigh = clamp(Math.round(mu + band), expectedLow, pBudget);

  bid = clamp(bid, 0, aiBudget);

  return {
    bid,
    expectedLow,
    expectedHigh,
    mode: "V2-utility",
    pred: Math.round(mu),
    sigma: Number(sigma.toFixed(1)),
    valueWin,
    costWeight: Number(costWeight.toFixed(3)),
    pWin: Number(winProb.toFixed(2)),
  };
}

// ---------- thought generation (B option) ----------
function bucketBid(bid, budget) {
  if (!Number.isFinite(bid) || !Number.isFinite(budget) || budget <= 0) return "mid";
  const r = bid / budget;
  if (r <= 0.2) return "low";
  if (r <= 0.45) return "mid";
  return "high";
}

function summarizePattern(clean) {
  const hist = Array.isArray(clean.history) ? clean.history.slice(-3) : [];
  if (!hist.length) return "no clear pattern yet";

  const buckets = hist.map((h) => bucketBid(h?.playerBid, clean.player.budget));
  const uniq = new Set(buckets);
  let pattern = "mixed signals";

  const bids = hist.map((h) => roundInt(h?.playerBid, 0));
  const monotoneUp = bids.every((b, i, arr) => i === 0 || b >= arr[i - 1]);
  const monotoneDown = bids.every((b, i, arr) => i === 0 || b <= arr[i - 1]);

  if (uniq.size === 1) pattern = `player sticks to ${buckets[0]} bids`;
  else if (monotoneUp) pattern = "player keeps edging up";
  else if (monotoneDown) pattern = "player keeps easing off";
  else pattern = "player looks volatile lately";

  const lastRes = hist[hist.length - 1]?.result;
  if (lastRes === "player") pattern += ", they took the last round";
  else if (lastRes === "ai") pattern += ", I took the last round";
  else if (lastRes === "tie") pattern += ", last round was a wash";

  return pattern;
}

function localThought(clean) {
  const seed = clean.round * 104729 + clean.ai.budget * 37 + clean.player.budget * 17;
  const rng = makeRng(seed);
  const model = computePlayerModel(clean, rng);
  const intent = pickMode(clean, model, rng);
  const pattern = summarizePattern(clean);

  const uncertainties = [
    "hard to tell if they swing again",
    "not sure they stay steady",
    "unsure they bluff or settle",
    "can't read their next push",
    "unsure if they tighten up",
  ];
  const intents = {
    saver: "play it saver-leaning",
    sniper: "try a sniper-like edge",
    bully: "lean bully-style pressure",
  };
  const styles = [
    (p, u, i) => `Seeing ${p}; ${u}, so I'll ${i}.`,
    (p, u, i) => `${p}. ${u} - time to ${i}.`,
    (p, u, i) => `${p}; ${u}, I should ${i}.`,
    (p, u, i) => `${p}; ${u}. I'll ${i}.`,
  ];

  const style = styles[Math.floor(rng() * styles.length)];
  const u = uncertainties[Math.floor(rng() * uncertainties.length)];
  const i = intents[intent] || "stay flexible";

  let thought = style(pattern, u, i);
  if (thought.length > 180) thought = thought.slice(0, 180).trim();
  return thought;
}

function sanitizeThought(text) {
  if (!text) return "";
  let t = String(text).replace(/\r?\n/g, "\n").trim();
  t = t.replace(/\s{2,}/g, " ").replace(/\s([,.;:!?])/g, "$1").trim();
  if (!t) return "";
  if (t.length > 220) t = t.slice(0, 220).trim();
  return t;
}

async function generateThoughtWithLLM(clean, adaptive, bucketLabel) {
  // B안: LLM thought는 "되면 좋고 아니면 말고"
  // -> 짧게 기다리고(soft timeout), 실패하면 바로 fallback.
  const SOFT_TIMEOUT_MS = Math.min(Math.max(1200, TIMEOUT_MS), 5000);

  const messages = [
    {
      role: "system",
      content:
        `You are the AI in a sealed-bid game. Write an "AI thought" that feels like real reasoning.\n` +
        `Rules:\n` +
        `- 2~4 short lines max.\n` +
        `- Structure MUST be:\n` +
        `  1) Read: what you inferred from player behavior\n` +
        `  2) Pressure: what matters now (scores/budgets/rounds)\n` +
        `  3) Plan: your intent this round (probe/defend/swing/conserve) WITHOUT revealing exact bid\n` +
        `- Avoid generic phrases.\n` +
        `- No exact bid amount. No precise math dumps.\n`
    },
    {
      role: "user",
      content:
        `Round ${clean.round}/${clean.totalRounds}\n` +
        `Budgets: AI=${clean.ai.budget}, Player=${clean.player.budget}\n` +
        `Scores: AI=${clean.ai.score}, Player=${clean.player.score}\n` +
        `Expected band: ${adaptive.expectedLow}-${adaptive.expectedHigh}\n` +
        `Volatility: ${adaptive.sigma}\n` +
        `My intensity: ${bucketLabel} (low/mid/high). Explain without numbers.\n` +
        `Recent history: ${JSON.stringify((clean.history || []).slice(-4))}\n`
    }
  ];

  const resp = await withTimeout(
    client.responses.create({
      model: MODEL,
      input: messages,
      max_output_tokens: 140,
    }),
    SOFT_TIMEOUT_MS
  );

  const text = extractOutputText(resp).trim();
  return text ? text.slice(0, 220) : "";
}

async function generateAiThought(clean, adaptive, bucketLabel) {
  // 1) primary: LLM
  try {
    const raw = await generateThoughtWithLLM(clean, adaptive, bucketLabel);
    const cleaned = sanitizeThought(raw);
    if (cleaned) return cleaned;
  } catch (err) {
    // 429/quota/timeout 모두 여기로 올 수 있음. 그냥 fallback.
    console.warn("AI thought generation failed, using fallback:", err?.message || err);
  }

  // 2) fallback: local
  return sanitizeThought(localThought(clean));
}

// ------------------ routes ------------------
app.get("/health", (_req, res) =>
  res.json({ ok: true, model: MODEL, timeout_ms: TIMEOUT_MS })
);

/**
 * POST /api/ai/bid
 * B option:
 * - bid is ALWAYS computed locally (fast, reliable)
 * - thought tries LLM quickly; fallback to local thought
 * - ALWAYS returns ok:true to prevent client OFFLINE state
 */
app.post("/api/ai/bid", async (req, res) => {
  const clean = sanitizeState(req.body);

  try {
    const adaptive = computeAdaptiveBid(clean);

    // 1) bid always local
    let ai_bid = applyBidGuardrails(clean, adaptive.bid);

    // wiggle
    const wiggle = Math.round((makeRng(clean.round + ai_bid + adaptive.expectedLow)() - 0.5) * 2);
    ai_bid = clamp(ai_bid + wiggle, 0, clean.ai.budget);

    const low = adaptive.expectedLow;
    const high = adaptive.expectedHigh;

    // bucket label for "intent intensity"
    const bucketLabel = bucketBid(ai_bid, clean.ai.budget);

    // 2) thought: LLM (fast) -> fallback local
    const thought = await generateAiThought(clean, adaptive, bucketLabel);

    console.log("[AI BID]", {
      round: clean.round,
      mode: adaptive.mode,
      pred: adaptive.pred,
      sigma: adaptive.sigma,
      pWin: adaptive.pWin,
      valueWin: adaptive.valueWin,
      costWeight: adaptive.costWeight,
      finalBid: ai_bid,
      thought: thought.slice(0, 500),
    });

    return res.json({
      ok: true, // ✅ critical for client: do NOT go OFFLINE
      ai_bid,
      expected_player_bid_low: low,
      expected_player_bid_high: high,
      thought,
      model: MODEL,
      debug: {
        mode: adaptive.mode,
        mu: adaptive.pred,
        sigma: adaptive.sigma,
        pWin: adaptive.pWin,
        valueWin: adaptive.valueWin,
        costWeight: adaptive.costWeight,
      },
    });
  } catch (err) {
    // Even on errors, keep ok:true and provide safe fallbacks.
    const msg = err?.message || String(err);
    console.error("❌ /api/ai/bid hard-failed (forcing safe response):", msg);

    const fallbackBid = clamp(Math.floor(clean.ai.budget * 0.12), 0, clean.ai.budget);
    const low = clamp(Math.floor(clean.player.budget * 0.08), 0, clean.player.budget);
    const high = clamp(Math.floor(clean.player.budget * 0.2), low, clean.player.budget);

    return res.json({
      ok: true,
      ai_bid: fallbackBid,
      expected_player_bid_low: low,
      expected_player_bid_high: high,
      thought: "System lag detected. Using safe fallback estimate and pacing.",
      model: "fallback",
      debug: { error: msg },
    });
  }
});

/**
 * POST /api/ai/report
 * Uses LLM. If API quota is blocked (429), returns ok:false with message.
 */
app.post("/api/ai/report", async (req, res) => {
  const clean = sanitizeState(req.body);

  try {
    const messages = [
      {
        role: "system",
        content:
          "You write concise, game-like post-match reports. Keep it punchy, insightful, and grounded in the provided history. No personal data claims.",
      },
      {
        role: "user",
        content:
          `Generate a post-game report (max 12 lines) for the human player.\n` +
          `Include: playstyle, predictability, budget efficiency, and 2 concrete tips.\n` +
          `State JSON:\n` +
          JSON.stringify(clean),
      },
    ];

    const resp = await withTimeout(
      client.responses.create({
        model: MODEL,
        input: messages,
        max_output_tokens: 300,
      }),
      Math.max(TIMEOUT_MS, 6000)
    );

    const report = extractOutputText(resp);
    if (!report) throw new Error("Empty model output (no text).");

    return res.json({ ok: true, report, model: MODEL });
  } catch (err) {
    const msg = err?.message || String(err);
    console.error("❌ /api/ai/report failed:", { name: err?.name || "Error", msg });
    return res.json({ ok: false, error: `${err?.name || "Error"}: ${msg}` });
  }
});

app.listen(PORT, () => {
  console.log(`✅ AI server running at http://localhost:${PORT}`);
  console.log(`   Model: ${MODEL}`);
  console.log(`   Timeout: ${TIMEOUT_MS}ms`);
});

// ------------------ guardrails ------------------
function roundsLeft(clean) {
  return Math.max(0, clean.totalRounds - clean.round + 1);
}

function isLockWinIfAIWins(clean) {
  const left = roundsLeft(clean);
  const after = Math.max(0, left - 1);
  const aS = clean.ai.score;
  const pS = clean.player.score;
  return aS + 1 - pS > after;
}

function isLockWinIfPlayerWins(clean) {
  const left = roundsLeft(clean);
  const after = Math.max(0, left - 1);
  const aS = clean.ai.score;
  const pS = clean.player.score;
  return pS + 1 - aS > after;
}

function shouldAllowSwing(clean) {
  const left = roundsLeft(clean);
  const aS = clean.ai.score;
  const pS = clean.player.score;

  if (isLockWinIfAIWins(clean)) return true;
  if (left <= 3 && isLockWinIfPlayerWins(clean)) return true;
  if (left <= 2 && aS < pS) return true;

  return false;
}

function applyBidGuardrails(clean, proposedBid) {
  let bid = clamp(proposedBid, 0, clean.ai.budget);

  const left = roundsLeft(clean);
  const base = left > 0 ? clean.ai.budget / left : clean.ai.budget;
  const stageId = Number(clean.stage?.id ?? 1);
  const swing = shouldAllowSwing(clean);

  let normalMul = stageId === 1 ? 1.05 : stageId === 2 ? 1.18 : 1.35;
  if (clean.round <= 3) normalMul -= 0.05;
  const normalCap = Math.max(0, Math.round(base * normalMul));

  let swingMul = stageId === 1 ? 1.4 : stageId === 2 ? 2.2 : 3.0;
  if (left <= 2) swingMul += 0.4;
  const swingCap = Math.max(0, Math.round(base * swingMul));

  const cap = swing ? swingCap : normalCap;
  bid = Math.min(bid, cap);

  if (clean.round === 1) {
    bid = Math.min(bid, Math.floor(clean.ai.budget * 0.2));
  }

  return clamp(bid, 0, clean.ai.budget);
}
