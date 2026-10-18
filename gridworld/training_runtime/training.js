const $ = id => document.getElementById(id);
let result = null, job = null, selectedMap = null, busy = false, initialized = false;
let heatmapMap = null, selectedHeatCell = null, policyRequest = 0, policyAnchor = null;
let lastCompletedJob = null, lastDrawn = null, animationHandle = null;
const ppoMode = () => ['ppo', 'grag'].includes($('mode').value);
const matches = data => data && data.map.name === $('map-list').value && data.mode === $('mode').value;
const interactive = () => job && job.execution !== 'batch' && ['running', 'paused'].includes(job.status) && matches(result) && job.session_id === result.session_id;
const integer = (value, min, max) => Number.isSafeInteger(value) && value >= min && value <= max;

async function api(url, body) {
  const response = await fetch(url, body === undefined ? undefined : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '処理に失敗しました。');
  return data;
}
function message(text, error = false) {
  $('status').textContent = text; $('status').classList.toggle('error', error);
}
function syncControls() {
  const active = job?.status === 'running';
  const current = interactive(), watching = current && job.execution === 'observe' && active;
  const skipping = current && job.execution === 'fast' && active;
  document.querySelectorAll('.q-setting').forEach(el => el.hidden = $('mode').value !== 'q_learning');
  document.querySelectorAll('.ppo-setting').forEach(el => el.hidden = !ppoMode());
  $('map-list').disabled = busy || active; $('mode').disabled = busy || active;
  $('new-training').disabled = !initialized || !selectedMap || busy || active;
  $('observe-training').disabled = !initialized || !selectedMap || busy || watching || (active && !current);
  $('pause-training').disabled = busy || !current || job.status === 'paused';
  $('skip-training').disabled = !initialized || !selectedMap || busy || skipping || (active && !current);
  $('stop-fast').disabled = busy;
  const count = Number($('batch-episodes').value);
  $('skip-training').textContent = integer(count, 1, 1000000) ? count.toLocaleString() + '試行を高速学習 →' : '指定回数を高速学習 →';
  $('fast-progress').hidden = !skipping;
  $('skip-overlay').hidden = !skipping;
  $('animation').parentElement.classList.toggle('skipping', skipping);
  $('live-indicator').classList.toggle('watching', watching);
  $('live-indicator').textContent = skipping ? '高速学習' : watching ? 'LIVE · 学習中' : current ? '一時停止' : '待機';
  $('phase-badge').textContent = skipping ? '高速学習中' : watching ? '観察しながら学習中' : current ? '一時停止' : '学習0回・開始前';
  if (skipping) {
    $('skip-progress').max = job.total; $('skip-progress').value = Math.min(job.completed, job.total);
    $('fast-progress-label').textContent = '高速学習 ' + job.completed.toLocaleString() + ' / ' + job.total.toLocaleString() + '試行';
  }
}
function trainingSettings() {
  const value = {
    max_steps: Number($('max-steps').value),
    seed: $('seed').value === '' ? null : Number($('seed').value),
    learning_rate: Number($('learning-rate').value),
    discount_factor: Number($('discount-factor').value),
    epsilon: Number($('epsilon').value),
    epsilon_decay: Number($('epsilon-decay').value),
    min_epsilon: Number($('min-epsilon').value)
  };
  if (!integer(value.max_steps, 1, 1000000)) throw Error('最大行動回数は1〜1,000,000の整数にしてください。');
  if (ppoMode()) {
    value.ppo_learning_rate = Number($('ppo-learning-rate').value);
    value.ppo_hidden_size = Number($('ppo-hidden-size').value);
  }
  return value;
}
function stats(data) {
  const n = data?.completed_episodes ?? data?.settings?.episodes ?? 0;
  const values = [[n.toLocaleString(), '完了した試行数'],
    [((data?.recent_success_rate ?? 0) * 100).toFixed(0) + '%', '直近100試行の到達率'],
    [(data?.recent_mean_steps ?? 0).toFixed(1), '直近100試行の平均行動数'],
    [(data?.total_steps ?? 0).toLocaleString(), '累計の行動数']];
  $('stats').replaceChildren(...values.map(([value, label]) => {
    const box = document.createElement('div'), caption = document.createElement('small'), number = document.createElement('strong');
    box.className = 'lab-stat'; caption.textContent = label; number.textContent = value;
    box.append(caption, number); return box;
  }));
  $('curve-count').textContent = n.toLocaleString() + '試行';
}
function drawCurve(history, total) {
  const top = 18, bottom = 230, left = 65, right = 740, n = Math.max(1, total);
  const x = v => left + v / n * (right - left), y = v => bottom - v * (bottom - top);
  const parts = ['<title>直近100試行の平均報酬</title>'];
  for (const v of [0, .25, .5, .75, 1]) parts.push('<line x1="' + left + '" y1="' + y(v) + '" x2="' + right + '" y2="' + y(v) + '" stroke="#e2eaee"/><text x="' + (left - 10) + '" y="' + (y(v) + 4) + '" text-anchor="end" fill="#738892" font-size="12">' + v.toFixed(2) + '</text>');
  for (const v of new Set([0, 1, 2, 3, 4].map(i => Math.round(total * i / 4)))) parts.push('<text x="' + x(v) + '" y="250" text-anchor="middle" fill="#738892" font-size="12">' + v.toLocaleString() + '</text>');
  const points = [left + ',' + bottom, ...history.map(item => x(item.episode).toFixed(2) + ',' + y(item.moving_average_reward).toFixed(2))];
  parts.push('<polyline points="' + points.join(' ') + '" fill="none" stroke="#078878" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>');
  parts.push('<text x="400" y="274" text-anchor="middle" fill="#71858d" font-size="12">完了した試行数</text>');
  if (history.length) {
    const last = history[history.length - 1];
    parts.push('<circle cx="' + x(last.episode) + '" cy="' + y(last.moving_average_reward) + '" r="4.5" fill="#078878"/>');
  }
  $('learning-curve').innerHTML = parts.join('');
}
function drawWorld(map, position, collision = false) {
  const canvas = $('animation'), context = canvas.getContext('2d'), unit = 700 / Math.max(map.rows, map.cols);
  canvas.width = Math.round(map.cols * unit); canvas.height = Math.round(map.rows * unit);
  canvas.style.width = (map.cols / Math.max(map.rows, map.cols) * 100) + "%";
  const walls = new Set(map.obstacles.map(cell => cell.join(',')));
  context.fillStyle = '#e2eaee'; context.fillRect(0, 0, canvas.width, canvas.height);
  for (let r = 0; r < map.rows; r++) for (let c = 0; c < map.cols; c++) {
    const wall = walls.has(r + ',' + c), start = r === map.start[0] && c === map.start[1], goal = r === map.goal[0] && c === map.goal[1];
    context.fillStyle = wall ? '#354f60' : goal ? '#bde5d7' : start ? '#e0ebf5' : '#fff';
    context.fillRect(c * unit + 1, r * unit + 1, unit - 2, unit - 2);
    if ((start || goal) && unit >= 18) {
      context.fillStyle = goal ? '#178267' : '#527d9c'; context.font = 'bold ' + Math.min(17, unit * .4) + 'px sans-serif';
      context.textAlign = 'center'; context.textBaseline = 'middle'; context.fillText(goal ? 'G' : 'S', (c + .5) * unit, (r + .5) * unit);
    }
  }
  if (position) {
    context.fillStyle = collision ? '#d55449' : '#ed8b35';
    context.beginPath(); context.arc((position[1] + .5) * unit, (position[0] + .5) * unit, unit * .29, 0, Math.PI * 2); context.fill();
    context.strokeStyle = '#fff'; context.lineWidth = Math.max(1, unit * .06); context.stroke();
  }
}
function drawLive(data) {
  const state = data.live || { position: data.map.start, episode: (data.settings.episodes || 0) + 1, step: 0 };
  const limit = state.max_steps || data.settings.max_steps;
  $('live-caption').textContent = '学習 ' + (data.completed_episodes ?? data.settings.episodes).toLocaleString() + '回 · ' + state.episode.toLocaleString() + '試行目 · ' + state.step + ' / ' + limit + '行動' + (state.reached_goal ? ' · ゴール到達' : state.collision ? ' · 衝突' : '');
  if (interactive() && job.execution === 'fast' && job.status === 'running') { lastDrawn = null; return; }
  if (animationHandle) cancelAnimationFrame(animationHandle);
  const previous = lastDrawn;
  lastDrawn = { ...state, position: state.position.slice() };
  const distance = previous ? Math.abs(previous.position[0] - state.position[0]) + Math.abs(previous.position[1] - state.position[1]) : 0;
  const watching = interactive() && job.execution === 'observe' && job.status === 'running';
  if (!watching || !previous || previous.episode !== state.episode || distance !== 1) {
    drawWorld(data.map, state.position, state.collision); return;
  }
  const start = performance.now(), duration = Math.min(Number($('observation-speed').value), 140);
  function animate(now) {
    const t = Math.min(1, (now - start) / duration);
    const position = previous.position.map((value, index) => value + (state.position[index] - value) * t);
    drawWorld(data.map, position, state.collision);
    if (t < 1) animationHandle = requestAnimationFrame(animate); else animationHandle = null;
  }
  animationHandle = requestAnimationFrame(animate);
}
function drawHeat(map, visits) {
  heatmapMap = map;
  const canvas = $('heatmap'), context = canvas.getContext('2d');
  const zoom = Number($('heat-zoom').value || 100) / 100;
  const available = Math.max(320, canvas.parentElement.clientWidth - 32);
  const unit = Math.max(44, available / Math.max(map.rows, map.cols)) * zoom;
  const width = map.cols * unit, height = map.rows * unit;
  // 拡大時にもキャンバスのメモリ使用量を抑え、同じサイズでは再確保しない。
  const density = Math.min(window.devicePixelRatio || 1, 2, 4096 / Math.max(width, height));
  const pixelWidth = Math.round(width * density), pixelHeight = Math.round(height * density);
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  canvas.style.width = width + 'px'; canvas.style.height = height + 'px';
  context.setTransform(density, 0, 0, density, 0, 0);
  const max = Math.max(0, ...visits.flat()), total = visits.flat().reduce((a, b) => a + b, 0);
  const walls = new Set(map.obstacles.map(cell => cell.join(',')));
  context.fillStyle = '#dce6eb'; context.fillRect(0, 0, width, height);
  for (let r = 0; r < map.rows; r++) for (let c = 0; c < map.cols; c++) {
    const wall = walls.has(r + ',' + c), ratio = max ? visits[r][c] / max : 0;
    context.fillStyle = wall ? '#354f60' : 'rgb(' + Math.round(255 - 35 * ratio) + ',' + Math.round(255 - 158 * ratio) + ',' + Math.round(255 - 180 * ratio) + ')';
    context.fillRect(c * unit + 1, r * unit + 1, unit - 2, unit - 2);
    if (!wall) {
      context.fillStyle = ratio > .7 ? '#fff' : '#344e5b';
      context.font = '600 ' + Math.min(38 * zoom, Math.max(16 * zoom, unit * .32)) + 'px sans-serif';
      context.textAlign = 'center'; context.textBaseline = 'middle';
      context.fillText(visits[r][c].toLocaleString(), (c + .5) * unit, (r + .5) * unit, unit - 8);
    }
  }
  if (selectedHeatCell && selectedHeatCell.mapName === map.name) {
    context.strokeStyle = '#078878'; context.lineWidth = Math.max(3, unit * .045);
    context.strokeRect(selectedHeatCell.col * unit + 3, selectedHeatCell.row * unit + 3, unit - 6, unit - 6);
  }
  $('heat-zoom-value').value = Math.round(zoom * 100) + '%';
  $('visit-count').textContent = total.toLocaleString() + '訪問'; $('heat-max').textContent = max.toLocaleString() + '回';
}
function redrawHeat() {
  const data = matches(result) ? result : null, map = data?.map || selectedMap;
  if (map) drawHeat(map, data?.visits || Array.from({ length: map.rows }, () => Array(map.cols).fill(0)));
}
function heatCellAt(event) {
  const map = heatmapMap;
  if (!map || map.name !== $('map-list').value) return null;
  const rect = $('heatmap').getBoundingClientRect();
  const row = Math.floor((event.clientY - rect.top) / rect.height * map.rows);
  const col = Math.floor((event.clientX - rect.left) / rect.width * map.cols);
  return row >= 0 && row < map.rows && col >= 0 && col < map.cols ? { row, col } : null;
}
function positionPolicyPopup() {
  if (!policyAnchor || $('policy-popup').hidden) return;
  const popup = $('policy-popup'), margin = 12, gap = 14;
  const rect = popup.getBoundingClientRect();
  const maxLeft = Math.max(margin, window.innerWidth - rect.width - margin);
  const maxTop = Math.max(margin, window.innerHeight - rect.height - margin);
  let left = policyAnchor.x + gap, top = policyAnchor.y + gap;
  if (left > maxLeft) left = policyAnchor.x - rect.width - gap;
  if (top > maxTop) top = policyAnchor.y - rect.height - gap;
  popup.style.left = Math.max(margin, Math.min(left, maxLeft)) + 'px';
  popup.style.top = Math.max(margin, Math.min(top, maxTop)) + 'px';
}
function closePolicyPopup(redraw = true, returnFocus = false) {
  policyRequest++;
  $('policy-popup').hidden = true;
  selectedHeatCell = null; policyAnchor = null;
  if (redraw) redrawHeat();
  if (returnFocus) $('heatmap').focus({ preventScroll: true });
}
function showPolicyDistribution(data) {
  $('policy-snapshot').textContent = '学習 ' + data.completed_episodes.toLocaleString() + '回 / ' + data.total_steps.toLocaleString() + '行動の時点'
    + ' · 学習時の選択確率（探索を含む）';
  $('policy-visits').textContent = 'このマス：' + data.visits.toLocaleString() + '訪問';
  $('policy-note').textContent = data.note;
  $('policy-distribution').replaceChildren();
  if (!data.supported) {
    $('policy-distribution').textContent = data.wall ? '障害物のマス' : '行動確率を表示できません';
    $('policy-refresh').disabled = true;
    positionPolicyPopup(); return;
  }
  const directions = [['up', '↑ 上'], ['down', '↓ 下'], ['left', '← 左'], ['right', '→ 右']];
  const best = Math.max(...directions.map(([action]) => data.probabilities[action]));
  for (const [action, title] of directions) {
    const probability = data.probabilities[action];
    const row = document.createElement('div'), label = document.createElement('span'), number = document.createElement('strong');
    const track = document.createElement('div'), fill = document.createElement('div');
    row.className = 'action-probability' + (probability === best ? ' best' : '');
    label.className = 'action-label'; label.textContent = title;
    if (data.blocked_actions.includes(action)) {
      const blocked = document.createElement('small'); blocked.textContent = '衝突'; label.append(blocked);
    }
    track.className = 'action-track'; fill.className = 'action-fill'; fill.style.width = (probability * 100) + '%';
    number.textContent = (probability * 100).toFixed(1) + '%';
    track.append(fill); row.append(label, track, number); $('policy-distribution').append(row);
  }
  $('policy-refresh').disabled = false; positionPolicyPopup();
}
async function openPolicyCell(row, col, anchor) {
  const token = ++policyRequest;
  selectedHeatCell = { row, col, mapName: heatmapMap.name };
  policyAnchor = anchor || policyAnchor;
  $('policy-title').textContent = 'マス [' + row + ', ' + col + '] の行動確率';
  $('policy-popup').hidden = false;
  $('policy-snapshot').textContent = '現在の方策を確認しています…';
  $('policy-distribution').textContent = ''; $('policy-note').textContent = ''; $('policy-visits').textContent = '';
  $('policy-refresh').disabled = true;
  redrawHeat(); positionPolicyPopup();
  if (!matches(result) || !result.session_id) {
    $('policy-snapshot').textContent = '「新しい学習」でAIを準備すると確認できます。';
    positionPolicyPopup(); return;
  }
  const requested = { session_id: result.session_id, mode: result.mode, map_name: result.map.name, position: [row, col] };
  try {
    const data = await api('/api/train/policy', requested);
    if (token !== policyRequest || $('policy-popup').hidden) return;
    showPolicyDistribution(data);
  } catch (error) {
    if (token !== policyRequest || $('policy-popup').hidden) return;
    $('policy-snapshot').textContent = '行動確率を取得できませんでした。';
    $('policy-note').textContent = error.message;
    $('policy-refresh').disabled = false; positionPolicyPopup();
  }
}
function render(data) {
  if (result && (result.session_id !== data.session_id || result.mode !== data.mode || result.map.name !== data.map.name)) closePolicyPopup(false);
  result = data; syncControls();
  if (!matches(data)) return;
  selectedMap = data.map;
  stats(data); drawCurve(data.history, data.completed_episodes ?? data.settings.episodes);
  drawHeat(data.map, data.visits); drawLive(data);
}
async function chooseMap() {
  closePolicyPopup(false);
  if (!$('map-list').value) return;
  selectedMap = await api('/api/maps/' + encodeURIComponent($('map-list').value));
  lastDrawn = null;
  if (matches(result)) render(result);
  else {
    stats(null); drawCurve([], 0); drawWorld(selectedMap, selectedMap.start);
    drawHeat(selectedMap, Array.from({ length: selectedMap.rows }, () => Array(selectedMap.cols).fill(0)));
    $('live-caption').textContent = '学習0回 · 開始前'; syncControls();
  }
}
async function control(action, fields = {}) {
  job = await api('/api/train/control', { action, session_id: job.session_id, ...fields });
  syncControls();
}
async function ensureInteractive(fresh = false) {
  if (fresh) closePolicyPopup(false);
  if (!fresh && interactive()) return;
  const settings = trainingSettings();
  const body = { map_name: $('map-list').value, mode: $('mode').value, execution: 'paused',
    step_delay_ms: Number($('observation-speed').value), training_settings: settings };
  if (!body.map_name) throw Error('マップ作成アプリでマップを保存してください。');
  if (!fresh && matches(result) && result.can_continue) body.session_id = result.session_id;
  job = await api('/api/train', body);
  // 操作に使う新しいIDを即座に保持する。実際の描画状態は次のポーリングで受け取る。
  if (!body.session_id) result = { session_id: job.session_id, mode: body.mode, map: selectedMap,
    settings: { ...settings, episodes: 0 }, completed_episodes: 0, history: [], visits: Array.from({ length: selectedMap.rows }, () => Array(selectedMap.cols).fill(0)),
    live: { episode: 1, step: 0, position: selectedMap.start }, can_continue: true };
  await control('limit', { max_steps: settings.max_steps });
  if (matches(result)) render(result);
}
async function action(kind) {
  if (busy) return;
  busy = true; syncControls();
  try {
    if (kind === 'new') { await ensureInteractive(true); message('学習0回のAIを準備しました。観察または高速学習を始められます。'); }
    else if (kind === 'observe') { await ensureInteractive(); await control('observe'); message('エージェントの試行錯誤を観察しています。'); }
    else if (kind === 'skip') {
      const count = Number($('batch-episodes').value);
      if (!integer(count, 1, 1000000)) throw Error('高速学習する試行回数は1〜1,000,000の整数にしてください。');
      await ensureInteractive(); await control('skip', { episodes: count });
      message(count.toLocaleString() + '試行を高速学習します。途中でも停止できます。');
    } else { await control('pause'); message(kind === 'stop-fast' ? '高速学習を停止しています…' : '一時停止しています…'); }
  } catch (error) { message(error.message, true); }
  finally { busy = false; syncControls(); }
}
async function poll() {
  try {
    const state = await api('/api/train/status');
    if (state.job_id) job = state;
    if (state.preview) render(state.preview);
    else if (state.status === 'complete' && state.job_id !== lastCompletedJob) {
      lastCompletedJob = state.job_id;
      const data = await api('/api/result'); render(data);
    }
    if (state.status === 'error') message(state.error || '学習中にエラーが発生しました。', true);
    syncControls();
  } catch (error) { if (initialized) message('接続を確認しています… ' + error.message, true); }
  finally { setTimeout(poll, job?.status === 'paused' ? 250 : 150); }
}
syncControls();
$('new-training').onclick = () => action('new');
$('observe-training').onclick = () => action('observe');
$('pause-training').onclick = () => action('pause');
$('skip-training').onclick = () => action('skip');
$('stop-fast').onclick = () => action('stop-fast');
$('batch-episodes').oninput = syncControls;
$('map-list').onchange = () => chooseMap().catch(error => message(error.message, true));
$('mode').onchange = () => { syncControls(); chooseMap().catch(error => message(error.message, true)); };
$('observation-speed').onchange = () => { if (interactive()) control('speed', { step_delay_ms: Number($('observation-speed').value) }).catch(error => message(error.message, true)); };
$('max-steps').onchange = () => {
  const value = Number($('max-steps').value);
  if (!integer(value, 1, 1000000)) { message('最大行動回数は1〜1,000,000の整数にしてください。', true); return; }
  if (interactive()) control('limit', { max_steps: value }).then(() => message('最大行動回数を変更しました。次の試行から適用します。')).catch(error => message(error.message, true));
};
$('heatmap').onmousemove = event => {
  const cell = heatCellAt(event);
  if (cell && matches(result)) $('heat-detail').textContent = '[' + cell.row + ', ' + cell.col + '] · ' + result.visits[cell.row][cell.col].toLocaleString() + '訪問 · クリックで行動確率';
};
$('heatmap').onclick = event => {
  const cell = heatCellAt(event);
  if (cell) openPolicyCell(cell.row, cell.col, { x: event.clientX, y: event.clientY });
};
$('heatmap').onkeydown = event => {
  if (event.key !== 'Enter' && event.key !== ' ') return;
  event.preventDefault();
  const map = heatmapMap;
  if (!map) return;
  const position = matches(result) ? result.live?.position || map.start : map.start;
  const rect = $('heatmap').getBoundingClientRect();
  openPolicyCell(position[0], position[1], { x: Math.max(12, Math.min(window.innerWidth - 12, rect.left + rect.width / 2)), y: Math.max(12, Math.min(window.innerHeight - 12, rect.top + rect.height / 2)) });
};
$('heat-zoom').oninput = redrawHeat;
$('policy-close').onclick = () => closePolicyPopup(true, true);
$('policy-refresh').onclick = () => {
  if (selectedHeatCell) openPolicyCell(selectedHeatCell.row, selectedHeatCell.col, policyAnchor);
};
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && !$('policy-popup').hidden) closePolicyPopup(true, true);
});
document.addEventListener('pointerdown', event => {
  if (!$('policy-popup').hidden && !$('policy-popup').contains(event.target) && event.target !== $('heatmap')) closePolicyPopup();
});
window.addEventListener('resize', () => { redrawHeat(); positionPolicyPopup(); });
if (typeof ResizeObserver !== 'undefined') new ResizeObserver(() => redrawHeat()).observe($('heatmap').parentElement);
(async function initialize() {
  try {
    const [maps, agents, config] = await Promise.all([api('/api/maps'), api('/api/agents'), api('/api/train/config')]);
    $('map-list').replaceChildren(...maps.map(map => new Option(map.name + ' (' + map.rows + '×' + map.cols + ')', map.name)));
    for (const agent of agents) if (![...$('mode').options].some(option => option.value === agent.key)) $('mode').add(new Option(agent.name, agent.key));
    const ids = { max_steps: 'max-steps', seed: 'seed', discount_factor: 'discount-factor', learning_rate: 'learning-rate',
      epsilon: 'epsilon', epsilon_decay: 'epsilon-decay', min_epsilon: 'min-epsilon', ppo_learning_rate: 'ppo-learning-rate', ppo_hidden_size: 'ppo-hidden-size' };
    for (const [key, id] of Object.entries(ids)) if (config[key] !== undefined) $(id).value = config[key] ?? '';
    try {
      const saved = await api('/api/result');
      if (maps.some(map => map.name === saved.map.name) && [...$('mode').options].some(option => option.value === saved.mode)) {
        $('map-list').value = saved.map.name; $('mode').value = saved.mode;
        for (const [key, id] of Object.entries(ids)) if (saved.settings[key] !== undefined) $(id).value = saved.settings[key] ?? '';
        render(saved);
        message('前回の学習状態を読み込みました。観察または高速学習を続けられます。');
      }
    } catch (_) { /* 保存結果がなければ学習0回から始める。 */ }
    await chooseMap(); if (!maps.length) message('マップ作成アプリでマップを保存してください。');
    initialized = true; syncControls(); poll();
  } catch (error) { message(error.message, true); }
})();
