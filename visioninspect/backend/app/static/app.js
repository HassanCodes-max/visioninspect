const MAX_PHOTOS = 30;
const MAX_SIDE = 1600;
const HEAT_FLOOR = 0.2;
const PASS_WAIT = 1500;
const ERRORS = { 413: "tooLarge", 415: "notImage", 429: "busy" };
const HISTORY_KEY = "visioninspect-history";
const DECISIONS = ["PASS", "REVIEW", "REJECT"];
const REPORT_KEY = "visioninspect-report";
const USER_KEY = "visioninspect-user";
const Z_95 = 1.96;
const MIN_BATCH = 2;
const OUTLINE_GRID = 256;
const SEVERITIES = ["minor", "major", "critical"];
const STEPS = ["stepRecognise", "stepDefects", "stepOutline", "stepFix"];
const STEP_TIME = 1800;
const RING = 326.7;
const CALM = matchMedia("(prefers-reduced-motion: reduce)");
const LANGUAGE_FADE = 170;
const VIEW_OUT = 220;
const SIGN_IN_PAUSE = 650;
const SKELETONS = 8;
const VIEWS = { "": "inspect", start: "inspect", inspect: "inspect", report: "report", history: "report", supported: "products", products: "products", guide: "guide", account: "account" };

const $ = (id) => document.getElementById(id);
const state = {
  language: localStorage.getItem("visioninspect-language") || "en",
  results: [],
  running: false,
  filter: "all",
  batchKey: null,
  open: null,
  spot: null,
  fresh: false,
  scanUrl: null,
  stepTimer: null,
  user: null
};

function say(key, values = {}) {
  const text = (TEXT[state.language] && TEXT[state.language][key]) || TEXT.en[key] || key;
  return text.replace(/\{(\w+)\}/g, (_, name) => values[name]);
}

function applyLanguage() {
  document.documentElement.lang = state.language;
  document.documentElement.dir = state.language === "ur" ? "rtl" : "ltr";
  document.querySelectorAll("[data-text]").forEach((node) => { node.textContent = say(node.dataset.text); });
  document.querySelectorAll("[data-label]").forEach((node) => node.setAttribute("aria-label", say(node.dataset.label)));
  const other = state.language === "ur" ? "en" : "ur";
  $("language").textContent = other === "ur" ? "اردو" : "English";
  $("language").lang = other;
  $("language").dir = other === "ur" ? "rtl" : "ltr";
  document.querySelectorAll("[data-hint]").forEach((node) => { node.placeholder = say(node.dataset.hint); });
  renderHistory();
  renderReport();
  renderAccount();
  if (state.results.length) renderResults();
  document.dispatchEvent(new CustomEvent("vi:language"));
}

async function loadProducts() {
  const { products } = await (await fetch("/api/products")).json();
  $("productList").replaceChildren(...products.map((name, index) => {
    const chip = document.createElement("span");
    chip.className = "chip plain pop-in";
    chip.style.setProperty("--i", index);
    chip.textContent = name;
    return chip;
  }));
}

async function shrink(blob) {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  if (scale === 1) {
    const aspect = bitmap.width / bitmap.height;
    bitmap.close();
    return { small: blob, aspect };
  }
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  const context = canvas.getContext("2d");
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const small = await new Promise((done) => canvas.toBlob(done, "image/jpeg", 0.92));
  return { small, aspect: canvas.width / canvas.height };
}

function visitorPass() {
  if (window.parent === window) return Promise.resolve({});
  return new Promise((done) => {
    const channel = new MessageChannel();
    const timer = setTimeout(() => done({}), PASS_WAIT);
    channel.port1.onmessage = ({ data }) => {
      clearTimeout(timer);
      channel.port1.close();
      done(data instanceof Map ? Object.fromEntries(data) : data || {});
    };
    window.parent.postMessage("zerogpu-headers", "https://huggingface.co", [channel.port2]);
  });
}

async function preparePhotos(list) {
  const ready = [];
  const refused = [];
  for (const [index, photo] of list.entries()) {
    setWorking(say("preparing", { i: index + 1, n: list.length }), index, list.length);
    try {
      ready.push({ ...(await shrink(photo.blob)), name: photo.name });
    } catch (error) {
      console.error(error);
      refused.push({ name: photo.name, error: "notImage" });
    }
  }
  return { ready, refused };
}

async function inspectAll(ready) {
  const form = new FormData();
  for (const photo of ready) form.append("photos", photo.small, "photo.jpg");
  if (Number($("limit").value) >= 0 && $("limit").value !== "") form.append("limit", $("limit").value);
  if (Number($("stock").value) > 0) form.append("stock", Math.round(Number($("stock").value)));
  const response = await fetch("/api/inspect-batch", { method: "POST", body: form, headers: await visitorPass() });
  if (!response.ok) throw new Error(ERRORS[response.status] || "failed");
  const { results, analysis, fallback } = await response.json();
  state.analysis = analysis;
  state.fallback = Boolean(fallback);
  return results.map((result, index) => ({ ...result, name: ready[index].name, small: ready[index].small, aspect: ready[index].aspect }));
}

function clearResults() {
  state.results.forEach((r) => URL.revokeObjectURL(r.url));
  state.results = [];
  state.filter = "all";
  state.analysis = null;
  state.spot = null;
  state.fallback = false;
  $("problem").hidden = true;
  $("notice").hidden = true;
  renderResults();
}

function addResult(result, small) {
  state.results.unshift({ ...result, url: URL.createObjectURL(small) });
}

function setWorking(text, done, total) {
  $("workingText").textContent = text;
  $("workingNumber").textContent = total ? `${done}/${total}` : "";
  $("workingBar").style.width = `${total ? 100 * done / total : 0}%`;
}

async function runPhotos(photos) {
  if (state.running || !photos.length) return;
  clearResults();
  state.running = true;
  $("problem").hidden = true;
  $("working").hidden = false;
  state.batchKey = Date.now();
  const list = photos.slice(0, MAX_PHOTOS);
  const group = $("group").value.trim();
  const user = $("inspector").value.trim();
  let trouble = photos.length > MAX_PHOTOS ? say("tooMany", { n: MAX_PHOTOS }) : "";
  const { ready, refused } = await preparePhotos(list);
  setWorking(say("waking"), list.length, list.length);
  if (ready.length) startScan(ready[0], ready.length);
  if (refused.length) trouble = `${refused[0].name}: ${say("notImage")}`;
  try {
    if (ready.length) {
      for (const result of await inspectAll(ready)) {
        if (result.error) {
          trouble = `${result.name}: ${say(result.error)}`;
        } else {
          addResult(result, result.small);
          recordResult(result, group, user);
        }
      }
    }
    stopScan();
    state.fresh = true;
    $("notice").hidden = !state.fallback;
    renderResults();
    renderReport();
  } catch (error) {
    console.error(error);
    stopScan();
    trouble = say(TEXT.en[error.message] ? error.message : "failed");
  }
  setWorking("", list.length, list.length);
  $("working").hidden = true;
  state.running = false;
  if (trouble) showProblem(trouble);
  if (state.results.length) saveHistory();
}

function notify(key) {
  if (window.toast) window.toast(say(key));
}

function showProblem(text) {
  $("problem").textContent = text;
  $("problem").hidden = false;
}

function heatCanvas(heat, aspect, width = 320) {
  const small = document.createElement("canvas");
  small.width = small.height = 32;
  const context = small.getContext("2d");
  const paint = context.createImageData(32, 32);
  heat.forEach((value, i) => {
    const level = Math.max(0, (value - HEAT_FLOOR) / (1 - HEAT_FLOOR));
    paint.data[4 * i] = 255;
    paint.data[4 * i + 1] = Math.round(182 - 150 * level);
    paint.data[4 * i + 2] = Math.round(58 - 20 * level);
    paint.data[4 * i + 3] = Math.round(225 * Math.sqrt(level));
  });
  context.putImageData(paint, 0, 0);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = Math.round(width / aspect);
  const big = canvas.getContext("2d");
  big.imageSmoothingQuality = "high";
  big.drawImage(small, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function drawBoxes(canvas, regions) {
  const context = canvas.getContext("2d");
  context.strokeStyle = "#ffffff";
  context.lineWidth = Math.max(2, canvas.width / 320);
  for (const region of regions) {
    context.strokeRect(region.x * canvas.width, region.y * canvas.height, region.width * canvas.width, region.height * canvas.height);
  }
}

function outlineCanvas(hex) {
  const bit = (x, y) => {
    if (x < 0 || y < 0 || x >= OUTLINE_GRID || y >= OUTLINE_GRID) return 0;
    const index = y * OUTLINE_GRID + x;
    return (parseInt(hex.substr(Math.floor(index / 8) * 2, 2), 16) >> (7 - index % 8)) & 1;
  };
  const small = document.createElement("canvas");
  small.width = small.height = OUTLINE_GRID;
  const context = small.getContext("2d");
  const paint = context.createImageData(OUTLINE_GRID, OUTLINE_GRID);
  for (let y = 0; y < OUTLINE_GRID; y += 1) {
    for (let x = 0; x < OUTLINE_GRID; x += 1) {
      if (!bit(x, y)) continue;
      const edge = !(bit(x - 1, y) && bit(x + 1, y) && bit(x, y - 1) && bit(x, y + 1));
      const at = 4 * (y * OUTLINE_GRID + x);
      paint.data.set(edge ? [255, 255, 255, 255] : [255, 60, 44, 130], at);
    }
  }
  context.putImageData(paint, 0, 0);
  return small;
}

function markCanvas(result, width) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = Math.round(width / result.aspect);
  const context = canvas.getContext("2d");
  if (result.outline) {
    context.imageSmoothingEnabled = false;
    context.drawImage(outlineCanvas(result.outline), 0, 0, canvas.width, canvas.height);
    return canvas;
  }
  if (result.heat) {
    const heat = heatCanvas(result.heat, result.aspect, width);
    if (result.decision === "REJECT") drawBoxes(heat, result.regions);
    return heat;
  }
  if (!result.regions.length) return null;
  context.fillStyle = "rgba(255, 82, 56, .32)";
  for (const region of result.regions) {
    context.fillRect(region.x * canvas.width, region.y * canvas.height, region.width * canvas.width, region.height * canvas.height);
  }
  drawBoxes(canvas, result.regions);
  return canvas;
}

function itemName(result) {
  return result.product || result.explanation?.item || say("notRecognised");
}

function defectLine(result) {
  const found = result.explanation;
  if (!found?.defect) return "";
  return found.severity ? `${found.defect} (${say(`sev_${found.severity}`)})` : found.defect;
}

function sizeLine(result) {
  if (result.decision === "PASS" || !result.regions.length) return "";
  const n = result.regions.length;
  return say(n > 1 ? "faultSizeMany" : "faultSize", { area: result.affected_percent, n });
}

function defectName(result) {
  return result.explanation?.defect || say("defectUnnamed");
}

function fixText(result) {
  if (result.decision === "PASS") return "";
  return result.explanation?.fix || say("fixGeneral");
}

function fixNote(result) {
  return result.explanation?.fix ? say("fixNote") : say("fixGeneralNote");
}

function severityText(result) {
  const severity = result.explanation?.severity;
  return severity ? say(`sev_${severity}`) : say("notRated");
}

function areaText(result) {
  return result.regions.length ? `${result.affected_percent}%` : say("notMeasured");
}

function sizeWord(area) {
  return say(area < 1 ? "sizeSmall" : area < 5 ? "sizeMedium" : "sizeLarge");
}

function attention(result) {
  const rank = { REJECT: 2000, REVIEW: 1000, PASS: 0 }[result.decision];
  return rank + 100 * (SEVERITIES.indexOf(result.explanation?.severity) + 1) + (result.affected_percent || 0);
}

function ranked() {
  return [...state.results].sort((a, b) => attention(b) - attention(a));
}

function countUp(node, value, suffix = "", decimals = 0) {
  if (CALM.matches || !value) {
    node.textContent = `${value.toFixed(decimals)}${suffix}`;
    return;
  }
  const began = performance.now();
  const step = (now) => {
    const done = Math.min(1, (now - began) / 900);
    node.textContent = `${(value * (1 - (1 - done) ** 3)).toFixed(decimals)}${suffix}`;
    if (done < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function showSkeletons(count) {
  const box = $("skeletons");
  box.replaceChildren(...Array.from({ length: Math.min(count, SKELETONS) }, (_, index) => {
    const card = document.createElement("div");
    card.className = "card skeleton-card";
    card.style.setProperty("--i", index);
    card.append(...["sk-photo", "sk-line", "sk-line short"].map((name) => {
      const part = document.createElement("span");
      part.className = name;
      return part;
    }));
    return card;
  }));
  box.hidden = false;
}

function hideSkeletons() {
  $("skeletons").hidden = true;
  $("skeletons").replaceChildren();
}

function startScan(photo, total) {
  const spot = $("spotlight");
  stopScan();
  state.scanUrl = URL.createObjectURL(photo.small);
  spot.hidden = false;
  spot.classList.remove("revealed");
  spot.classList.add("scanning");
  $("spotImage").src = state.scanUrl;
  $("spotMark").getContext("2d").clearRect(0, 0, $("spotMark").width, $("spotMark").height);
  $("spotBadge").className = "verdict-pill";
  $("spotBadge").textContent = say("checking");
  $("spotCount").textContent = total > 1 ? say("scanCount", { n: total }) : "";
  $("spotTitle").textContent = photo.name;
  $("spotName").textContent = "";
  $("spotBody").hidden = true;
  document.querySelector(".spot-actions").hidden = true;
  let step = 0;
  $("spotStep").textContent = say(STEPS[0]);
  showSkeletons(total);
  state.stepTimer = setInterval(() => {
    step = Math.min(step + 1, STEPS.length - 1);
    $("spotStep").textContent = say(STEPS[step]);
  }, STEP_TIME);
  spot.scrollIntoView({ behavior: CALM.matches ? "auto" : "smooth", block: "center" });
}

function stopScan() {
  hideSkeletons();
  clearInterval(state.stepTimer);
  state.stepTimer = null;
  $("spotlight").classList.remove("scanning");
  $("spotStep").textContent = "";
  if (state.scanUrl) URL.revokeObjectURL(state.scanUrl);
  state.scanUrl = null;
  if (!state.results.length) $("spotlight").hidden = true;
}

function showSpotlight(result, animate = true) {
  const spot = $("spotlight");
  if (!result) {
    if (!state.stepTimer) spot.hidden = true;
    return;
  }
  state.spot = result;
  const order = ranked();
  spot.hidden = false;
  spot.classList.remove("scanning");
  document.querySelector(".spot-actions").hidden = false;
  $("spotBody").hidden = false;
  $("spotStep").textContent = "";
  $("spotCount").textContent = order.length > 1 ? say("spotCount", { i: order.indexOf(result) + 1, n: order.length }) : "";
  $("spotPrev").disabled = $("spotNext").disabled = order.length < 2;
  $("spotBadge").className = `verdict-pill ${result.decision.toLowerCase()}`;
  $("spotBadge").textContent = say(result.decision);
  $("spotTitle").textContent = itemName(result);
  $("spotName").textContent = result.name;
  $("spotWhy").textContent = [say(`why_${result.reason}`), sizeLine(result)].filter(Boolean).join(" ");
  const flagged = result.decision !== "PASS";
  $("spotGauges").hidden = !flagged;
  $("spotFixBox").hidden = !flagged;
  const area = result.regions.length ? result.affected_percent : 0;
  $("spotRing").style.strokeDashoffset = `${RING}`;
  if (flagged) {
    $("spotFault").textContent = defectName(result);
    $("spotWhere").textContent = result.explanation?.where || "";
    $("spotFix").textContent = fixText(result);
    $("spotFixNote").textContent = fixNote(result);
    $("spotSeverity").textContent = severityText(result);
    $("spotSize").textContent = result.regions.length ? sizeWord(area) : "";
    const level = SEVERITIES.indexOf(result.explanation?.severity) + 1;
    [...$("spotMeter").children].forEach((bar, index) => bar.classList.toggle("on", index < level));
    if (result.regions.length) countUp($("spotArea"), area, "%", 2);
    else $("spotArea").textContent = "-";
    const shown = result.regions.length ? Math.max(1.5, Math.min(100, area)) : 0;
    requestAnimationFrame(() => requestAnimationFrame(() => { $("spotRing").style.strokeDashoffset = `${RING * (1 - shown / 100)}`; }));
  }
  $("spotTested").textContent = result.tested_share != null ? say("spotTested", { n: Math.round(100 * result.tested_share) }) : "";
  const image = $("spotImage");
  image.onload = () => {
    const canvas = $("spotMark");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    const mark = markCanvas(result, canvas.width);
    if (mark) context.drawImage(mark, 0, 0, canvas.width, canvas.height);
  };
  image.src = result.url;
  image.alt = result.name;
  if (animate) {
    spot.classList.remove("revealed");
    void spot.offsetWidth;
    spot.classList.add("revealed");
  }
}

function moveSpotlight(step) {
  const order = ranked();
  const at = order.indexOf(state.spot);
  showSpotlight(order[(at + step + order.length) % order.length]);
}

function tableRow(cells, tag, className = "") {
  const row = document.createElement("tr");
  if (className) row.className = className;
  row.append(...cells.map((value) => {
    const cell = document.createElement(tag);
    cell.textContent = value;
    return cell;
  }));
  return row;
}

function renderFindings() {
  const rows = ranked().filter((r) => r.decision !== "PASS");
  $("findings").hidden = !rows.length;
  const head = [say("csvItem"), say("csvDecision"), say("fFault"), say("detailSeverity"), say("detailAffected"), say("fWhere"), say("detailFix")];
  $("findingsTable").replaceChildren(tableRow(head, "th"), ...rows.map((r) => tableRow(
    [r.name, say(r.decision), defectName(r), severityText(r), areaText(r), r.explanation?.where || "-", fixText(r)], "td", r.decision.toLowerCase())));
}

function cardLine(result) {
  const defect = defectLine(result);
  const parts = [defect || say(`why_${result.reason}`), sizeLine(result)].filter(Boolean);
  return parts.join(defect ? " · " : " ");
}

function pill(decision) {
  const tag = document.createElement("span");
  tag.className = `verdict-pill ${decision.toLowerCase()}`;
  tag.textContent = say(decision);
  return tag;
}

function renderTally() {
  const counts = Object.fromEntries(DECISIONS.map((d) => [d, state.results.filter((r) => r.decision === d).length]));
  $("tally").replaceChildren(...[["all", state.results.length], ...DECISIONS.map((d) => [d, counts[d]])].map(([key, count]) => {
    const item = document.createElement("div");
    item.className = `tally-item ${key.toLowerCase()}`;
    const number = document.createElement("strong");
    if (state.fresh) countUp(number, count);
    else number.textContent = count;
    const label = document.createElement("span");
    label.textContent = key === "all" ? say("tallyItems", { n: "" }).trim() : say(key);
    item.append(number, label);
    return item;
  }));
  $("filters").replaceChildren(...["all", ...DECISIONS].map((key) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chip${state.filter === key ? " on" : ""}`;
    button.textContent = key === "all" ? say("all") : say(key);
    button.addEventListener("click", () => {
      state.filter = key;
      renderResults();
    });
    return button;
  }));
}

function listInto(list, items) {
  list.replaceChildren(...items.map((text) => {
    const item = document.createElement("li");
    item.textContent = text;
    return item;
  }));
}

function renderAnalysis() {
  const stats = state.analysis;
  const show = Boolean(stats) && stats.photos >= MIN_BATCH;
  $("analysis").hidden = !show;
  if (!show) return;
  $("analysisHead").textContent = say("batchHead", { rate: stats.defect_rate, rejected: stats.rejected, photos: stats.photos });
  const parts = [say("batchRange", { low: stats.range_low, high: stats.range_high })];
  if (stats.review) parts.push(say("batchReview", { n: stats.review }));
  parts.push(say("batchMissed"));
  $("analysisRange").textContent = parts.join(" ");
  $("analysisStock").hidden = !stats.stock;
  if (stats.stock) $("analysisStock").textContent = say("batchStock", stats.stock);
  const how = SEVERITIES.filter((name) => stats.severity?.[name]).reverse().map((name) => `${stats.severity[name]} ${say(`sev_${name}`)}`);
  const howParts = [how.length ? say("batchSeverity", { list: how.join(", ") }) : "", stats.mean_area ? say("batchArea", { area: stats.mean_area }) : ""].filter(Boolean);
  $("analysisHow").hidden = !howParts.length;
  $("analysisHow").textContent = howParts.join(" ");
  const advice = $("analysisAdvice");
  advice.className = `lot-advice ${stats.advice}`;
  advice.textContent = say(`lot_${stats.advice}`, { limit: stats.limit });
  $("analysisPareto").replaceChildren(...stats.pareto.map((entry) => {
    const row = document.createElement("div");
    row.className = "pareto-row";
    const line = document.createElement("div");
    const name = document.createElement("span");
    name.textContent = entry.defect;
    const count = document.createElement("strong");
    count.textContent = `${entry.count} (${entry.share}%)`;
    line.append(name, count);
    const track = document.createElement("div");
    track.className = "limit-track";
    const fill = document.createElement("span");
    fill.style.width = `${entry.share}%`;
    track.append(fill);
    row.append(line, track);
    return row;
  }));
  const hasAdvice = stats.causes.length || stats.actions.length;
  $("analysisAi").hidden = !hasAdvice;
  listInto($("analysisCauses"), stats.causes);
  listInto($("analysisActions"), stats.actions);
}

function renderResults() {
  $("results").hidden = !state.results.length;
  renderTally();
  renderAnalysis();
  $("printTitle").textContent = `${say("reportTitle")}, ${new Date().toLocaleDateString()}`;
  const shown = state.results.filter((r) => state.filter === "all" || r.decision === state.filter);
  $("grid").replaceChildren(...shown.map((result, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `card shown ${result.decision.toLowerCase()}${state.fresh ? " enter" : ""}`;
    card.style.setProperty("--i", Math.min(index, 20));
    const photo = document.createElement("div");
    photo.className = "card-photo";
    const image = new Image();
    image.src = result.url;
    image.alt = result.name;
    photo.append(image);
    const mark = markCanvas(result, 320);
    if (mark) photo.append(mark);
    const meta = document.createElement("div");
    meta.className = "card-meta";
    const name = document.createElement("span");
    name.className = "card-name";
    name.textContent = itemName(result);
    meta.append(name, pill(result.decision));
    const why = document.createElement("p");
    why.className = "card-why";
    why.textContent = cardLine(result);
    card.append(photo, meta, why);
    if (result.decision !== "PASS") {
      const fix = document.createElement("p");
      fix.className = "card-fix";
      fix.textContent = say("cardFix", { fix: fixText(result) });
      card.append(fix);
    }
    card.addEventListener("click", () => {
      showSpotlight(result);
      $("spotlight").scrollIntoView({ behavior: CALM.matches ? "auto" : "smooth", block: "center" });
    });
    return card;
  }));
  renderFindings();
  const keep = state.results.includes(state.spot) ? state.spot : ranked()[0];
  showSpotlight(keep, state.fresh);
  state.fresh = false;
}

function hottest(result) {
  if (!result.heat) return result.regions.length ? Math.round(Math.min(90, Math.max(35, 100 * (result.regions[0].x + result.regions[0].width) + 8))) : 55;
  const index = result.heat.indexOf(Math.max(...result.heat));
  return Math.round(Math.min(90, Math.max(35, 100 * ((index % 32) + 0.5) / 32 + 18)));
}

function setCompare(value) {
  $("compareTop").style.clipPath = `inset(0 ${100 - value}% 0 0)`;
  $("compareLine").style.left = `${value}%`;
}

function openDetail(result) {
  state.open = result;
  const badge = $("detailBadge");
  badge.className = `verdict-pill ${result.decision.toLowerCase()}`;
  badge.textContent = say(result.decision);
  $("detailTitle").textContent = result.name;
  $("detailWhy").textContent = say(`why_${result.reason}`);
  $("detailSize").textContent = sizeLine(result);
  $("detailSize").hidden = !sizeLine(result);
  $("detailProduct").textContent = itemName(result);
  $("detailAffected").textContent = result.regions.length ? `${result.affected_percent}%` : say("notMeasured");
  $("detailRegions").textContent = result.regions.length || say("notMeasured");
  const found = result.explanation;
  $("detailSeverity").textContent = found?.severity ? say(`sev_${found.severity}`) : say("notMeasured");
  $("detailAdvice").hidden = result.decision === "PASS";
  $("detailDefect").textContent = defectLine(result) || defectName(result);
  $("detailWhere").textContent = found?.where || "";
  $("detailFix").textContent = fixText(result);
  $("detailFixNote").textContent = fixNote(result);
  $("detailTested").textContent = result.tested_share != null ? `${Math.round(100 * result.tested_share)}%` : say("notMeasured");
  $("detailVersion").textContent = say("modelVersion", { v: result.model_version });
  const image = $("compareBase");
  image.onload = () => {
    const canvas = $("compareHeat");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    const mark = markCanvas(result, canvas.width);
    if (mark) {
      if (result.heat && !result.outline) drawBoxes(mark, result.regions);
      context.drawImage(mark, 0, 0, canvas.width, canvas.height);
    }
  };
  image.src = result.url;
  image.alt = result.name;
  const start = hottest(result);
  $("compareRange").value = start;
  setCompare(start);
  if (!$("detail").open) $("detail").showModal();
}

function saveHistory() {
  const count = (decision) => state.results.filter((r) => r.decision === decision).length;
  const entry = { key: state.batchKey, date: new Date().toISOString(), items: state.results.length, pass: count("PASS"), review: count("REVIEW"), reject: count("REJECT") };
  const past = readHistory().filter((old) => old.key !== entry.key);
  localStorage.setItem(HISTORY_KEY, JSON.stringify([entry, ...past].slice(0, 12)));
  renderHistory();
}

function readHistory() {
  try {
    const past = JSON.parse(localStorage.getItem(HISTORY_KEY));
    return Array.isArray(past) ? past : [];
  } catch {
    return [];
  }
}

function updateEmpty() {
  $("reportEmpty").hidden = !($("report").hidden && $("history").hidden);
}

function renderHistory() {
  const past = readHistory();
  $("history").hidden = !past.length;
  updateEmpty();
  $("historyList").replaceChildren(...past.map((entry) => {
    const line = document.createElement("p");
    line.textContent = say("historyLine", { ...entry, date: new Date(entry.date).toLocaleString(state.language === "ur" ? "ur-PK" : "en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) });
    return line;
  }));
}

function readReport() {
  try {
    const report = JSON.parse(localStorage.getItem(REPORT_KEY));
    return report && typeof report === "object" && !Array.isArray(report) ? report : {};
  } catch {
    return {};
  }
}

function recordResult(result, group, user) {
  const name = group || result.explanation?.item || result.product || say("notRecognised");
  const key = (user ? `${user}|${name}` : name).toLowerCase();
  const report = readReport();
  const entry = report[key] || { name, user, photos: 0, PASS: 0, REVIEW: 0, REJECT: 0, defects: {}, severity: {} };
  entry.photos += 1;
  entry[result.decision] += 1;
  if (result.decision === "REJECT" && result.regions.length) {
    entry.area = (entry.area || 0) + result.affected_percent;
    entry.marked = (entry.marked || 0) + 1;
  }
  const found = result.explanation;
  if (found?.defect) {
    const defect = found.defect.toLowerCase();
    entry.defects[defect] = (entry.defects[defect] || 0) + 1;
    entry.fixes = entry.fixes || {};
    if (found.fix && !entry.fixes[defect]) entry.fixes[defect] = found.fix;
  }
  if (found?.severity) entry.severity[found.severity] = (entry.severity[found.severity] || 0) + 1;
  report[key] = entry;
  localStorage.setItem(REPORT_KEY, JSON.stringify(report));
}

function mostCommon(counts) {
  const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  return ranked.length ? ranked[0][0] : "";
}

function wilson(hits, total) {
  const share = hits / total;
  const scale = 1 + Z_95 ** 2 / total;
  const middle = (share + Z_95 ** 2 / (2 * total)) / scale;
  const spread = Z_95 * Math.sqrt(share * (1 - share) / total + Z_95 ** 2 / (4 * total ** 2)) / scale;
  return [Math.max(0, middle - spread), Math.min(1, middle + spread)].map((value) => Math.round(1000 * value) / 10);
}

function userName(entry) {
  return entry.user || say("rptNoUser");
}

function reportRows() {
  const entries = Object.values(readReport()).sort((a, b) => (a.user || "").localeCompare(b.user || "") || b.photos - a.photos);
  return entries.map((entry) => {
    const severity = mostCommon(entry.severity);
    const common = mostCommon(entry.defects);
    const [low, high] = wilson(entry.REJECT, entry.photos);
    return [userName(entry), entry.name, entry.photos, entry.PASS, entry.REVIEW, entry.REJECT, `${Math.round(100 * entry.REJECT / entry.photos)}%`, `${low}% - ${high}%`,
      common || "-", severity ? say(`sev_${severity}`) : "-", entry.marked ? `${(entry.area / entry.marked).toFixed(1)}%` : "-",
      (common && entry.fixes?.[common]) || (entry.REJECT ? say("fixGeneral") : "-")];
  });
}

function reportHead() {
  return [say("rptUser"), say("rptProduct"), say("rptPhotos"), say("PASS"), say("REVIEW"), say("REJECT"), say("rptRate"), say("rptRange"), say("rptCommon"), say("rptSeverity"), say("rptArea"), say("rptFix")];
}

function userTotals() {
  const totals = new Map();
  for (const entry of Object.values(readReport())) {
    const key = (entry.user || "").toLowerCase();
    const total = totals.get(key) || { name: userName(entry), photos: 0, rejected: 0 };
    total.photos += entry.photos;
    total.rejected += entry.REJECT;
    totals.set(key, total);
  }
  return [...totals.values()].sort((a, b) => b.photos - a.photos);
}

function renderUsers() {
  $("reportUsers").replaceChildren(...userTotals().map((total) => {
    const rate = Math.round(100 * total.rejected / total.photos);
    const [low, high] = wilson(total.rejected, total.photos);
    const row = document.createElement("div");
    row.className = "pareto-row";
    const line = document.createElement("div");
    const name = document.createElement("span");
    name.textContent = total.name;
    const count = document.createElement("strong");
    count.textContent = say("rptUserLine", { rejected: total.rejected, photos: total.photos, rate, low, high });
    line.append(name, count);
    const track = document.createElement("div");
    track.className = "limit-track";
    const fill = document.createElement("span");
    fill.style.width = `${rate}%`;
    track.append(fill);
    row.append(line, track);
    return row;
  }));
}

function renderReport() {
  const rows = reportRows();
  $("report").hidden = !rows.length;
  updateEmpty();
  renderUsers();
  $("reportTable").replaceChildren(tableRow(reportHead(), "th"), ...rows.map((row) => tableRow(row, "td")));
}

function safeCell(value) {
  const text = String(value ?? "");
  const guarded = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return `"${guarded.replace(/"/g, '""')}"`;
}

function saveSheet(head, rows, name) {
  const csv = [head, ...rows].map((row) => row.map(safeCell).join(",")).join("\r\n");
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob(["﻿", csv], { type: "text/csv;charset=utf-8" }));
  link.download = `${name}-${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 2000);
}

function downloadCsv() {
  const head = ["csvItem", "csvDecision", "csvReason", "csvProduct", "csvDefect", "csvSeverity", "csvWhere", "csvFix", "csvScore", "csvArea", "csvRegions", "csvModel"].map((key) => say(key));
  const rows = state.results.map((r) => [r.name, r.decision, say(`why_${r.reason}`), itemName(r), r.explanation?.defect, r.explanation?.severity && say(`sev_${r.explanation.severity}`),
    r.explanation?.where, fixText(r), r.score, r.affected_percent, r.regions.length, r.model_version]);
  saveSheet(head, rows, "visioninspect");
}

async function loadSample(name) {
  if (state.running) return;
  state.running = true;
  $("working").hidden = false;
  $("inspect").scrollIntoView({ behavior: "smooth" });
  setWorking(say("loading"), 0, 0);
  try {
    showSkeletons(SKELETONS);
    const sample = await (await fetch(`/static/samples/${name}.json`)).json();
    clearResults();
    state.analysis = sample.analysis || null;
    state.batchKey = Date.now();
    for (const [index, item] of sample.items.entries()) {
      setWorking(say("preparing", { i: index + 1, n: sample.items.length }), index, sample.items.length);
      const { small, aspect } = await shrink(await (await fetch(item.src)).blob());
      addResult({ ...item.result, name: `${name}-${index + 1}.jpg`, aspect }, small);
    }
    $("problem").hidden = true;
    hideSkeletons();
    state.fresh = true;
    renderResults();
    saveHistory();
    $("spotlight").scrollIntoView({ behavior: CALM.matches ? "auto" : "smooth", block: "center" });
  } catch (error) {
    console.error(error);
    hideSkeletons();
    showProblem(say("failed"));
  }
  $("working").hidden = true;
  state.running = false;
}

function viewOf(hash) {
  return VIEWS[hash.replace("#", "")] || "inspect";
}

function showView(name, target = "") {
  const next = document.querySelector(`.view[data-view="${name}"]`);
  const current = document.querySelector(".view:not([hidden])");
  document.querySelectorAll("[data-nav]").forEach((link) => link.classList.toggle("active", link.dataset.nav === name));
  closeMenu();
  const land = () => {
    const spot = target && target !== "start" && name === "inspect" ? document.getElementById(target) : null;
    if (spot) spot.scrollIntoView({ behavior: CALM.matches ? "auto" : "smooth" });
    else scrollTo({ top: 0 });
  };
  if (current === next) {
    land();
    return;
  }
  const swapIn = () => {
    if (current) current.hidden = true;
    next.hidden = false;
    next.classList.remove("view-in");
    void next.offsetWidth;
    next.classList.add("view-in");
    land();
    document.dispatchEvent(new CustomEvent("vi:view", { detail: name }));
  };
  if (!current || CALM.matches) {
    swapIn();
    return;
  }
  current.classList.add("view-out");
  setTimeout(() => {
    current.classList.remove("view-out");
    swapIn();
  }, VIEW_OUT);
}

function closeMenu() {
  $("links").classList.remove("open");
  $("menu").setAttribute("aria-expanded", "false");
}

function initials(name) {
  return name.split(/\s+/).filter(Boolean).slice(0, 2).map((part) => part[0].toUpperCase()).join("");
}

function renderAccount() {
  const user = state.user;
  $("accountLink").classList.toggle("signed", Boolean(user));
  $("avatar").textContent = user ? initials(user.name) : "";
  $("accountText").textContent = user ? user.name.split(/\s+/)[0] : say("authSignIn");
  $("authTabs").hidden = Boolean(user);
  if (user) {
    $("meAvatar").textContent = initials(user.name);
    $("meTitle").textContent = say("authMeTitle", { name: user.name });
    $("meUser").textContent = `@${user.username}`;
    showPane("paneMe");
  } else if (!$("paneMe").hidden) {
    showPane("paneIn");
  }
}

function showPane(id) {
  const panes = ["paneIn", "paneUp", "paneMe"];
  const current = panes.find((pane) => !$(pane).hidden);
  $("tabIn").setAttribute("aria-selected", String(id === "paneIn"));
  $("tabUp").setAttribute("aria-selected", String(id === "paneUp"));
  $("authTabs").classList.toggle("second", id === "paneUp");
  if (current === id) return;
  panes.forEach((pane) => { $(pane).hidden = pane !== id; });
  const pane = $(id);
  pane.classList.remove("slide-left", "slide-right");
  void pane.offsetWidth;
  pane.classList.add(panes.indexOf(id) > panes.indexOf(current) ? "slide-left" : "slide-right");
}

function signIn() {
  const button = $("signIn");
  button.classList.add("loading");
  button.disabled = true;
  $("signInText").textContent = say("authConnecting");
  const url = `/login/huggingface?_target_url=${encodeURIComponent("/?signed=1")}`;
  setTimeout(() => {
    if (window.parent === window) {
      location.href = url;
      return;
    }
    window.open(location.origin + url, "_blank", "noopener");
    button.classList.remove("loading");
    button.disabled = false;
    $("signInText").textContent = say("authInButton");
    notify("authNewTab");
  }, CALM.matches ? 0 : SIGN_IN_PAUSE);
}

async function loadAccount() {
  const params = new URLSearchParams(location.search);
  try {
    const me = await (await fetch("/api/me")).json();
    state.user = me.signed_in ? { name: me.name, username: me.username } : null;
  } catch (error) {
    console.error(error);
    state.user = null;
  }
  $("accountLink").classList.remove("loading");
  renderAccount();
  if (state.user && !$("inspector").value) {
    $("inspector").value = state.user.name;
    localStorage.setItem(USER_KEY, state.user.name);
  }
  if (params.has("signed") || params.has("signedout")) {
    history.replaceState(null, "", location.pathname + location.hash);
    if (params.has("signedout")) {
      if (window.toast) window.toast(say("authOut"));
    } else if (state.user) {
      if (window.toast) window.toast(say("authOk", { name: state.user.name }), "success");
      showView("account");
    } else {
      if (window.toast) window.toast(say("authFail"), "error");
      showView("account");
    }
  }
}

function connect() {
  const drop = $("drop");
  const input = $("file");
  const open = () => { if (!state.running) input.click(); };
  drop.addEventListener("click", (event) => { if (event.target !== input) open(); });
  input.addEventListener("click", (event) => event.stopPropagation());
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
  });
  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    runPhotos([...event.dataTransfer.files].map((file) => ({ blob: file, name: file.name })));
  });
  input.addEventListener("change", () => {
    const photos = [...input.files].map((file) => ({ blob: file, name: file.name }));
    input.value = "";
    runPhotos(photos);
  });
  document.querySelectorAll("[data-sample]").forEach((chip) => chip.addEventListener("click", () => loadSample(chip.dataset.sample)));
  $("csv").addEventListener("click", () => { downloadCsv(); notify("toastSheet"); });
  $("spotPrev").addEventListener("click", () => moveSpotlight(-1));
  $("spotNext").addEventListener("click", () => moveSpotlight(1));
  $("spotOpen").addEventListener("click", () => { if (state.spot) openDetail(state.spot); });
  $("print").addEventListener("click", () => window.print());
  $("clear").addEventListener("click", () => { clearResults(); notify("toastCleared"); });
  $("reportSave").addEventListener("click", () => { saveSheet(reportHead(), reportRows(), "visioninspect-report"); notify("toastReport"); });
  $("reportPrint").addEventListener("click", () => {
    $("reportStamp").textContent = `${say("rptTitle")}, ${new Date().toLocaleDateString()}`;
    document.body.classList.add("print-report");
    window.print();
  });
  addEventListener("afterprint", () => document.body.classList.remove("print-report"));
  $("inspector").value = localStorage.getItem(USER_KEY) || "";
  $("inspector").addEventListener("change", () => localStorage.setItem(USER_KEY, $("inspector").value.trim()));
  $("reportClear").addEventListener("click", () => {
    localStorage.removeItem(REPORT_KEY);
    renderReport();
    notify("toastReportCleared");
  });
  $("forget").addEventListener("click", () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
    notify("toastHistory");
  });
  $("language").addEventListener("click", () => {
    state.language = state.language === "ur" ? "en" : "ur";
    localStorage.setItem("visioninspect-language", state.language);
    document.body.classList.add("switching");
    setTimeout(() => {
      applyLanguage();
      document.body.classList.remove("switching");
    }, CALM.matches ? 0 : LANGUAGE_FADE);
  });
  $("menu").addEventListener("click", () => {
    const open = $("links").classList.toggle("open");
    $("menu").setAttribute("aria-expanded", String(open));
  });
  $("tabIn").addEventListener("click", () => showPane("paneIn"));
  $("tabUp").addEventListener("click", () => showPane("paneUp"));
  $("toUp").addEventListener("click", () => showPane("paneUp"));
  $("toIn").addEventListener("click", () => showPane("paneIn"));
  $("signIn").addEventListener("click", signIn);
  addEventListener("hashchange", () => showView(viewOf(location.hash), location.hash.slice(1)));
  addEventListener("keydown", (event) => { if (event.key === "Escape") closeMenu(); });
  $("compareRange").addEventListener("input", (event) => setCompare(Number(event.target.value)));
  $("detailClose").addEventListener("click", () => $("detail").close());
  $("detail").addEventListener("click", (event) => { if (event.target === $("detail")) $("detail").close(); });
  addEventListener("scroll", () => document.querySelector(".top").classList.toggle("scrolled", scrollY > 12), { passive: true });
}

connect();
applyLanguage();
showView(viewOf(location.hash), location.hash.slice(1));
loadProducts().catch(console.error);
loadAccount();
