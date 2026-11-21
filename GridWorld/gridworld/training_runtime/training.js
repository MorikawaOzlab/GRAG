const $ = id => document.getElementById(id);
let result = null, job = null, selectedMap = null, busy = false, initialized = false;
let heatmapMap = null, selectedHeatCell = null, policyRequest = 0, policyAnchor = null;
let lastCompletedJob = null, lastDrawn = null, animationHandle = null;
let lastPreviewKey = null;
let initialJobAdopted = false, programStopped = false;
let liveTrainingClock = null, liveClockReceivedAt = 0;
const formatDuration = seconds => seconds < 60 ? seconds.toFixed(2) + ' 秒'
  : Math.floor(seconds / 60).toLocaleString() + ' 分 ' + (seconds % 60).toFixed(1) + ' 秒';
const ppoMode = () => $('mode').value === 'ppo';
const gragMode = () => ['grag', 'sgcrl'].includes($('mode').value);
const cppoMode = () => $('mode').value === 'cppo';
const cppoFields = { learning_rate: 'cppo-learning-rate', hidden_size: 'cppo-hidden-size',
  gamma: 'cppo-gamma', rollout_steps: 'cppo-rollout-steps', batch_size: 'cppo-batch-size', epochs: 'cppo-epochs',
  entropy_coef: 'cppo-entropy-coef', normalize_advantage: 'cppo-normalize-advantage' };
const matches = data => data && data.map.name === $('map-list').value && data.mode === $('mode').value;
const interactive = () => job && job.execution !== 'batch' && ['running', 'paused'].includes(job.status) && matches(result) && job.session_id === result.session_id;
const integer = (value, min, max) => Number.isSafeInteger(value) && value >= min && value <= max;

async function api(url, body) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  let response;
  try {
    response = await fetch(url, body === undefined ? {signal: controller.signal} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: controller.signal
    });
  } catch (error) {
    if (error.name === 'AbortError') throw Error('処理の応答待ちが長くなっています。進捗を確認してから再操作してください。');
    throw error;
  } finally { clearTimeout(timeout); }
  const data = await response.json();
  if (!response.ok) throw Error(data.error || '処理に失敗しました。');
  return data;
}
function message(text, error = false) {
  $('status').textContent = text; $('status').classList.toggle('error', error);
}
function syncControls() {
  const active = job?.status === 'running' || job?.saving;
  const stopping = active && job?.control_pending;
  const current = interactive(), watching = current && job.execution === 'observe' && active;
  const measuring = current && job.benchmark && active;
  const skipping = current && (job.execution === 'fast' || measuring) && active;
  document.querySelectorAll('.q-setting').forEach(el => el.hidden = $('mode').value !== 'q_learning');
  document.querySelectorAll('.ppo-setting').forEach(el => el.hidden = !ppoMode());
  document.querySelectorAll('.crl-setting').forEach(el => el.hidden = !gragMode());
  document.querySelectorAll('.grag-setting').forEach(el => el.hidden = $('mode').value !== 'grag');
  document.querySelectorAll('.sgcrl-setting').forEach(el => el.hidden = $('mode').value !== 'sgcrl');
  document.querySelectorAll('.cppo-setting').forEach(el => el.hidden = !cppoMode());
  $('discount-factor').parentElement.hidden = gragMode() || cppoMode();
  $('map-list').disabled = busy || active; $('mode').disabled = busy || active;
  $('new-training').disabled = !initialized || !selectedMap || busy || active;
  $('observe-training').disabled = !initialized || !selectedMap || busy || watching || measuring || stopping || job?.saving || (active && !current);
  $('pause-training').disabled = busy || stopping || job?.saving || !current || job.status === 'paused';
  $('skip-training').disabled = !initialized || !selectedMap || busy || stopping || job?.saving || skipping || (active && !current);
  $('start-measurement').disabled = !initialized || !selectedMap || busy || active;
  $('measurement-limit').disabled = busy || active;
  $('max-steps').disabled = measuring || stopping || job?.saving;
  $('grag-contrastive-epochs').disabled = busy || measuring || stopping || job?.saving;
  $('pause-training').textContent = measuring ? '計測を停止' : 'Ⅱ 一時停止';
  $('stop-fast').disabled = busy || stopping || job?.saving;
  const count = Number($('batch-episodes').value);
  $('skip-training').textContent = integer(count, 1, 1000000) ? count.toLocaleString() + '試行を高速学習 →' : '指定回数を高速学習 →';
  $('fast-progress').hidden = !skipping;
  $('skip-overlay').hidden = !skipping;
  $('animation').parentElement.classList.toggle('skipping', skipping);
  $('animation').classList.toggle('goal-selectable', initialized && Boolean(selectedMap) &&
    !busy && !stopping && !job?.saving && (!active || (current && !skipping)));
  $('live-indicator').classList.toggle('watching', watching);
  $('live-indicator').textContent = skipping ? '高速学習' : watching ? 'LIVE · 学習中' : current ? '一時停止' : '待機';
  $('phase-badge').textContent = measuring ? '100%到達まで計測中' : skipping ? '高速学習中' : watching ? '観察しながら学習中' : current ? '一時停止' : result?.benchmark ? ({reached: '100%到達・計測完了', limit: '時間上限・未達成', cancelled: '計測を停止'}[result.benchmark.status] || '計測終了') : '学習0回・開始前';
  if (job?.saving) $('phase-badge').textContent = '学習停止済み・保存中';
  else if (stopping) $('phase-badge').textContent = '停止受付済み・更新完了待ち';
  else if (active && job.snapshot_age_seconds > 1) $('phase-badge').textContent = '学習更新中・直前の進捗を表示';
  else if (active && job.runtime?.phase === 'updating') $('phase-badge').textContent =
    '重み更新 ' + job.runtime.completed + ' / ' + job.runtime.total;
  if (skipping) {
    $('skip-progress').max = measuring ? job.max_hours * 3600 : job.total;
    $('skip-progress').value = measuring ? Math.min(job.elapsed_seconds || 0, job.max_hours * 3600) : Math.min(job.completed, job.total);
    $('fast-progress-label').textContent = measuring ? '計測 ' + ((job.elapsed_seconds || 0) / 3600).toFixed(3) + ' / ' + job.max_hours + '時間 · ' + job.completed.toLocaleString() + '試行' : '高速学習 ' + job.completed.toLocaleString() + ' / ' + job.total.toLocaleString() + '試行';
    if (job?.saving || stopping) $('fast-progress-label').textContent = job.saving ? '学習は停止済みです。保存しています…' : '停止要求を受け付けました。実行中の更新を待っています…';
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
  if (gragMode()) {
    if ($('mode').value === 'grag') {
      const gammaInput = $('hindsight-gamma').value;
      const gamma = Number(gammaInput);
      if (gammaInput.trim() === '' || !Number.isFinite(gamma) || gamma <= 0 || gamma >= 1)
        throw Error('将来ゴール割引率は0より大きく1未満にしてください。');
      value.hindsight_gamma = gamma;
    }
    if ($('mode').value === 'sgcrl') {
      const gammaInput = $('crl-gamma').value;
      const gamma = Number(gammaInput);
      if (gammaInput.trim() === '' || !Number.isFinite(gamma) || gamma <= 0 || gamma >= 1)
        throw Error('割引率は0より大きく1未満にしてください。');
      value.crl_gamma = gamma;
    }
    const learningRate = Number($('crl-learning-rate').value);
    if (!Number.isFinite(learningRate) || learningRate <= 0) throw Error('対照学習の学習率は正の数にしてください。');
    if ($('mode').value === 'grag') value.contrastive_learning_rate = learningRate;
    else {
      value.crl_learning_rate = learningRate;
      value.crl_actor_learning_rate = learningRate;
      value.crl_hidden_size = Number($('crl-hidden-size').value);
      value.crl_min_replay_size = Number($('crl-min-replay').value);
    }
  }
  if ($('mode').value === 'grag') {
    value.grag_device = $('grag-device').value;
    value.grag_update_interval = Number($('grag-update-interval').value);
    if (!integer(value.grag_update_interval, 1, 1000000)) throw Error('更新間隔は1〜1,000,000の整数にしてください。');
    value.grag_contrastive_epochs = Number($('grag-contrastive-epochs').value);
    if (!integer(value.grag_contrastive_epochs, 1, 1000)) throw Error('対照学習のエポック数は1〜1000の整数にしてください。');
  }
  if ($('mode').value === 'sgcrl') {
    value.crl_device = $('sgcrl-device').value;
    value.crl_batch_size = Number($('crl-batch-size').value);
    value.crl_replay_cache_mb = Number($('crl-cache-mb').value);
    if (!integer(value.crl_batch_size, 2, 4096)) throw Error('バッチサイズは2〜4096にしてください。');
    if (!integer(value.crl_replay_cache_mb, 0, 8192)) throw Error('キャッシュ上限は0〜8192MiBにしてください。');
  }
  if (cppoMode()) for (const [key, id] of Object.entries(cppoFields)) value['cppo_' + key] =
    key === 'normalize_advantage' ? $(id).value === 'true' : Number($(id).value);
  return value;
}
function stats(data) {
  const n = data?.completed_episodes ?? data?.settings?.episodes ?? 0;
  const milestone = data?.success_100_milestone;
  const duration = formatDuration;
  const values = [[n.toLocaleString(), '完了した試行数'],
    [((data?.recent_success_rate ?? 0) * 100).toFixed(0) + '%', '直近100試行の到達率'],
    [(data?.recent_mean_steps ?? 0).toFixed(1), '直近100試行の平均行動数'],
    [(data?.total_steps ?? 0).toLocaleString(), '累計の行動数'],
    [duration(data?.training_seconds ?? 0), data?.timing_complete === false ? '学習実行時間（再開後）' : '学習実行時間'],
    [milestone ? duration(milestone.seconds) : data?.timing_complete === false ? '記録なし（旧データ）' : '未達成',
      '直近100試行が100%になるまで'],
    [milestone ? milestone.episode.toLocaleString() + ' 試行 / ' + milestone.steps.toLocaleString() + ' 行動' : '—',
      '初めて100%になった時点']];
  $('stats').replaceChildren(...values.map(([value, label], index) => {
    const box = document.createElement('div'), caption = document.createElement('small'), number = document.createElement('strong');
    box.className = 'lab-stat'; caption.textContent = label; number.textContent = value;
    if (index === 4) number.id = 'training-clock';
    box.append(caption, number); return box;
  }));
  $('curve-count').textContent = n.toLocaleString() + '試行';
  refreshTrainingClock();
}
function refreshTrainingClock() {
  const clock = liveTrainingClock;
  if (!clock || !result || clock.session_id !== result.session_id) return;
  const number = $('training-clock');
  if (number) number.textContent = formatDuration(clock.seconds +
    (clock.running ? Math.max(0, performance.now() - liveClockReceivedAt) / 1000 : 0));
}
setInterval(refreshTrainingClock, 100);
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
function drawCountHeatmap(canvas, map, counts, zoom) {
  const context = canvas.getContext('2d');
  const stage = canvas.parentElement, style = getComputedStyle(stage);
  const paddingX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  const paddingY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
  const availableWidth = Math.max(1, stage.clientWidth - paddingX);
  const heightLimit = parseFloat(style.maxHeight);
  const availableHeight = Number.isFinite(heightLimit)
    ? Math.max(1, heightLimit - paddingY - (stage.offsetHeight - stage.clientHeight))
    : availableWidth;
  // 100%は縦横比を保って枠内に全マスが収まる倍率。
  const unit = Math.min(availableWidth / map.cols, availableHeight / map.rows) * zoom;
  const width = map.cols * unit, height = map.rows * unit;
  // 拡大時にもキャンバスのメモリ使用量を抑え、同じサイズでは再確保しない。
  const density = Math.min(window.devicePixelRatio || 1, 2, 4096 / Math.max(width, height));
  const pixelWidth = Math.round(width * density), pixelHeight = Math.round(height * density);
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  canvas.style.width = width + 'px'; canvas.style.height = height + 'px';
  context.setTransform(density, 0, 0, density, 0, 0);
  const values = counts.flat();
  const max = Math.max(0, ...values), total = values.reduce((a, b) => a + b, 0);
  const walls = new Set(map.obstacles.map(cell => cell.join(',')));
  context.fillStyle = '#dce6eb'; context.fillRect(0, 0, width, height);
  for (let r = 0; r < map.rows; r++) for (let c = 0; c < map.cols; c++) {
    const wall = walls.has(r + ',' + c), ratio = max ? counts[r][c] / max : 0;
    context.fillStyle = wall ? '#354f60' : 'rgb(' + Math.round(255 - 35 * ratio) + ',' + Math.round(255 - 158 * ratio) + ',' + Math.round(255 - 180 * ratio) + ')';
    context.fillRect(c * unit + 1, r * unit + 1, unit - 2, unit - 2);
    if (!wall) {
      context.fillStyle = ratio > .7 ? '#fff' : '#344e5b';
      context.font = '600 ' + Math.max(1, Math.min(38 * zoom, unit * .32)) + 'px sans-serif';
      context.textAlign = 'center'; context.textBaseline = 'middle';
      context.fillText(counts[r][c].toLocaleString(), (c + .5) * unit, (r + .5) * unit, Math.max(1, unit - 8));
    }
  }
  return {context, unit, max, total};
}
function drawHeat(map, visits) {
  heatmapMap = map;
  const zoom = Number($('heat-zoom').value || 100) / 100;
  const {context, unit, max, total} = drawCountHeatmap($('heatmap'), map, visits, zoom);
  if (matches(result) && result.mode === 'grag' && valueGoal) {
    const [r, c] = valueGoal;
    context.fillStyle = '#078878'; context.beginPath();
    context.arc((c + .84) * unit, (r + .16) * unit, Math.max(2, unit * .08), 0, Math.PI * 2);
    context.fill();
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
    + (data.mode === 'grag' ? ' · 目的地 [' + data.selected_goal.join(', ') + '] に対する方策'
                            : ' · 学習時の選択確率（探索を含む）');
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
  const map = heatmapMap;
  const wall = map.obstacles.some(([r, c]) => r === row && c === col);
  $('policy-set-goal').hidden = !matches(result) || result.mode !== 'grag';
  $('policy-set-goal').disabled = wall;
  redrawHeat(); positionPolicyPopup();
  if (!matches(result) || !result.session_id) {
    $('policy-snapshot').textContent = '「新しい学習」でAIを準備すると確認できます。';
    positionPolicyPopup(); return;
  }
  const requested = { session_id: result.session_id, mode: result.mode, map_name: result.map.name, position: [row, col] };
  if (result.mode === 'grag') requested.goal = [...(valueGoal || result.map.goal)];
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
  if (result && (result.session_id !== data.session_id || result.mode !== data.mode || result.map.name !== data.map.name ||
      result.map.goal.join(',') !== data.map.goal.join(','))) {
    lastDrawn = null;
    closePolicyPopup(false);
    closeValuePopup();
    valueToken++; valueGrid = null; valueKey = null; valueGoal = null; valueSnapshotGoal = null; $('value-warning').hidden = true;
    drawValueMap();
  }
  result = data; syncControls();
  if (!matches(data)) return;
  selectedMap = data.map;
  scheduleValueMap(data);
  stats(data); drawCurve(data.history, data.completed_episodes ?? data.settings.episodes);
  drawHeat(data.map, data.visits); drawLive(data);
}
async function chooseMap() {
  valueToken++; valueGrid = null; valueKey = null; valueGoal = null; valueSnapshotGoal = null; $('value-section').hidden = true;
  $('value-warning').hidden = true;
  $('value-goal-error').hidden = true;
  closeValuePopup();
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
  if (body.session_id && body.mode === 'grag') {
    await control('contrastive_epochs', { grag_contrastive_epochs: settings.grag_contrastive_epochs });
  }
  if (matches(result)) render(result);
}

$('animation').onclick = async event => {
  const canvas = $('animation');
  if (!canvas.classList.contains('goal-selectable')) return;
  const map = matches(result) ? result.map : selectedMap;
  const rect = canvas.getBoundingClientRect();
  const row = Math.floor((event.clientY - rect.top) / rect.height * map.rows);
  const col = Math.floor((event.clientX - rect.left) / rect.width * map.cols);
  if (!integer(row, 0, map.rows - 1) || !integer(col, 0, map.cols - 1)) return;
  const goal = [row, col];
  if ([map.start, ...map.obstacles].some(cell => cell[0] === row && cell[1] === col)) {
    message('スタートと障害物を除いたマスを選んでください。', true);
    return;
  }
  if (map.goal[0] === row && map.goal[1] === col) return;
  busy = true; syncControls();
  try {
    await ensureInteractive();
    await control('goal', { goal });
    message(`最終ゴールを [${row}, ${col}] に変更します。訪問回数はゼロから集計します。`);
  } catch (error) { message(error.message, true); }
  finally { busy = false; syncControls(); }
};
async function action(kind) {
  if (busy) return;
  busy = true; syncControls();
  try {
    if (kind === 'new') { await ensureInteractive(true); message('学習0回のAIを準備しました。観察または高速学習を始められます。'); }
    else if (kind === 'measure') {
      const limit = Number($('measurement-limit').value), settings = trainingSettings();
      if (!Number.isFinite(limit) || limit <= 0 || limit > 100000) throw Error('計測の最大時間は0より大きく100,000以下にしてください。');
      if (!$('map-list').value) throw Error('計測するマップを選択してください。');
      closePolicyPopup(false);
      job = await api('/api/train', {map_name: $('map-list').value, mode: $('mode').value,
        execution: 'benchmark', max_hours: limit, training_settings: settings});
      result = {session_id: job.session_id, mode: $('mode').value, map: selectedMap,
        settings: {...settings, episodes: 0}, completed_episodes: 0, history: [],
        visits: Array.from({length: selectedMap.rows}, () => Array(selectedMap.cols).fill(0)),
        live: {episode: 1, step: 0, position: selectedMap.start}, can_continue: true,
        benchmark: {status: 'running'}};
      render(result);
      message('新しいAIで計測を開始しました。直近100試行が100%になったら自動終了します。');
    }
    else if (kind === 'observe') { await ensureInteractive(); await control('observe'); message('エージェントの試行錯誤を観察しています。'); }
    else if (kind === 'skip') {
      const count = Number($('batch-episodes').value);
      if (!integer(count, 1, 1000000)) throw Error('高速学習する試行回数は1〜1,000,000の整数にしてください。');
      await ensureInteractive(); await control('skip', { episodes: count });
      message(count.toLocaleString() + '試行を高速学習します。途中でも停止できます。');
    } else { const measuring = job?.benchmark; await control('pause'); message(measuring ? '計測を停止しています…' : kind === 'stop-fast' ? '高速学習を停止しています…' : '一時停止しています…'); }
  } catch (error) { message(error.message, true); }
  finally { busy = false; syncControls(); }
}
async function poll() {
  if (programStopped) return;
  try {
    const state = await api('/api/train/status');
    if (state.job_id) job = state;
    if (state.training_clock) {
      liveTrainingClock = {...state.training_clock, session_id: state.session_id};
      liveClockReceivedAt = performance.now();
      refreshTrainingClock();
    }
    else if (['idle', 'error', 'complete', 'initializing'].includes(state.status)) liveTrainingClock = null;
    const runtime = state.runtime;
    $('training-runtime').textContent = runtime ? '実行デバイス: ' + runtime.device +
      ' / 重み更新: ' + runtime.updates.toLocaleString() + '回' +
      (runtime.phase === 'updating' ? '（今回 ' + runtime.completed + '/' + runtime.total + '）' : '') +
      ' / バッチ: ' + runtime.batch_size + ' / 更新速度: ' + runtime.updates_per_second.toFixed(1) + '回/秒' +
      (runtime.cuda_graph ? ' / CUDA Graph有効' : runtime.cuda_graph_error ? ' / CUDA Graph利用不可' : '') +
      ' / リプレイ: ' + runtime.replay_size.toLocaleString() + '件' +
      (runtime.replay_cached ? '（' + (runtime.device.startsWith('cuda') ? 'GPU' : 'CPU') + 'キャッシュ）' : '（CPU抽出）') : '';
    if (state.preview) {
      if ((!initialJobAdopted || state.status === 'running') && !busy && !matches(state.preview)) {
        // Reloading the page must not leave another AI selected with its controls locked.
        $('map-list').value = state.preview.map.name;
        $('mode').value = state.mode;
        $('max-steps').value = state.preview.settings.max_steps;
      }
      initialJobAdopted = true;
      const key = state.job_id + ':' + state.preview_revision;
      if (state.preview_revision === undefined || key !== lastPreviewKey) {
        render(state.preview); lastPreviewKey = key;
      }
    }
    else if (state.status === 'complete' && state.job_id !== lastCompletedJob) {
      lastCompletedJob = state.job_id;
      const data = await api('/api/result'); render(data);
      if (data.benchmark) message(data.benchmark.status === 'reached'
        ? '計測完了：100%到達まで ' + data.success_100_milestone.seconds.toFixed(2) + ' 秒（' + data.success_100_milestone.episode.toLocaleString() + '試行）。'
        : data.benchmark.status === 'limit' ? '時間上限に達したため計測を終了しました。成功率100%には未達成です。' : '計測を停止しました。');
    }
    if (state.status === 'error') message(state.error || '学習中にエラーが発生しました。', true);
    else if (state.control_error) message(state.control_error, true);
    syncControls();
  } catch (error) { if (initialized && !programStopped) message('接続を確認しています… ' + error.message, true); }
  finally { if (!programStopped) setTimeout(poll, job?.status === 'paused' ? 250 : 150); }
}
const batchEpisodesInput = $('batch-episodes');
const batchEpisodesStorageKey = 'gridworld.training.batchEpisodes';
try {
  const savedCount = Number(localStorage.getItem(batchEpisodesStorageKey));
  if (integer(savedCount, 1, 1000000)) batchEpisodesInput.value = String(savedCount);
} catch (_) { /* Storage can be unavailable in some browser sessions. */ }

batchEpisodesInput.oninput = () => {
  const count = Number(batchEpisodesInput.value);
  if (integer(count, 1, 1000000)) {
    try { localStorage.setItem(batchEpisodesStorageKey, String(count)); }
    catch (_) { /* The number is still usable for the current page. */ }
  }
  syncControls();
};
syncControls();
$('restart-program').onclick = async () => {
  if (!confirm('再起動すると実行中の学習は終了します。再起動しますか？')) return;
  const button = $('restart-program');
  button.disabled = true;
  try {
    const previous = await api('/api/runtime');
    await api('/api/restart', {});
    programStopped = true;
    document.querySelectorAll('button, input, select').forEach(element => element.disabled = true);
    $('phase-badge').textContent = '再起動中';
    message('再起動しています…');
    const deadline = Date.now() + 60000;
    while (Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 500));
      try {
        const response = await fetch('/api/runtime', {cache: 'no-store', signal: AbortSignal.timeout(2000)});
        if (response.ok && (await response.json()).runtime_id !== previous.runtime_id) {
          location.reload();
          return;
        }
      } catch (_) { /* The server may be restarting. */ }
    }
    message('再起動を確認できませんでした。ページを再読み込みしてください。', true);
  } catch (error) { button.disabled = false; message(error.message, true); }
};
$('shutdown-program').onclick = async () => {
  const button = $('shutdown-program');
  button.disabled = true;
  try {
    await api('/api/shutdown', {});
    programStopped = true;
    document.querySelectorAll('button, input, select').forEach(element => element.disabled = true);
    $('phase-badge').textContent = 'プログラム停止済み';
    message('プログラムを停止しました。');
  } catch (error) { button.disabled = false; message(error.message, true); }
};
$('new-training').onclick = () => action('new');
$('start-measurement').onclick = () => action('measure');
$('observe-training').onclick = () => action('observe');
$('pause-training').onclick = () => action('pause');
$('skip-training').onclick = () => action('skip');
$('stop-fast').onclick = () => action('stop-fast');
$('map-list').onchange = () => chooseMap().catch(error => message(error.message, true));
$('mode').onchange = () => { syncControls(); chooseMap().catch(error => message(error.message, true)); };
$('observation-speed').onchange = () => { if (interactive()) control('speed', { step_delay_ms: Number($('observation-speed').value) }).catch(error => message(error.message, true)); };
$('max-steps').onchange = () => {
  const value = Number($('max-steps').value);
  if (!integer(value, 1, 1000000)) { message('最大行動回数は1〜1,000,000の整数にしてください。', true); return; }
  if (interactive()) control('limit', { max_steps: value }).then(() => message('最大行動回数を変更しました。次の試行から適用します。')).catch(error => message(error.message, true));
};
$('grag-contrastive-epochs').onchange = async () => {
  const value = Number($('grag-contrastive-epochs').value);
  if (!integer(value, 1, 1000)) { message('対照学習のエポック数は1〜1000の整数にしてください。', true); return; }
  if (!interactive()) return;
  try {
    await control('contrastive_epochs', { grag_contrastive_epochs: value });
    message('対照学習のエポック数を変更しました。次の更新から適用します。');
  } catch (error) { message(error.message, true); }
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
$('heat-fit').onclick = () => {
  $('heat-zoom').value = '100';
  redrawHeat();
  $('heatmap').parentElement.scrollTo(0, 0);
  positionPolicyPopup();
};
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
      epsilon: 'epsilon', epsilon_decay: 'epsilon-decay', min_epsilon: 'min-epsilon', ppo_learning_rate: 'ppo-learning-rate', ppo_hidden_size: 'ppo-hidden-size',
      crl_hidden_size: 'crl-hidden-size', crl_min_replay_size: 'crl-min-replay',
      grag_update_interval: 'grag-update-interval', grag_contrastive_epochs: 'grag-contrastive-epochs', grag_device: 'grag-device',
      crl_device: 'sgcrl-device', crl_batch_size: 'crl-batch-size', crl_replay_cache_mb: 'crl-cache-mb' };
    for (const [key, id] of Object.entries(cppoFields)) ids['cppo_' + key] = id;
    for (const [key, id] of Object.entries(ids)) if (config[key] !== undefined) $(id).value = config[key] ?? '';
    $('hindsight-gamma').value = config.hindsight_gamma ?? 0.99;
    $('crl-gamma').value = config.crl_gamma ?? 0.99;
    $('crl-learning-rate').value = ($('mode').value === 'grag' ? config.contrastive_learning_rate ?? 0.00025 : config.crl_learning_rate ?? 0.0003);
    await chooseMap();
    initialized = true; syncControls(); poll();
    if (!maps.length) message('マップ作成アプリでマップを保存してください。');
    try {
      const saved = await api('/api/result');
      if (!job?.job_id && !result?.session_id && !busy && maps.some(map => map.name === saved.map.name) && [...$('mode').options].some(option => option.value === saved.mode)) {
        $('map-list').value = saved.map.name; $('mode').value = saved.mode;
        for (const [key, id] of Object.entries(ids)) if (saved.settings[key] !== undefined) $(id).value = saved.settings[key] ?? '';
        $('hindsight-gamma').value = saved.settings.hindsight_gamma ?? 0.99;
        $('crl-gamma').value = saved.settings.crl_gamma ?? 0.99;
        $('crl-learning-rate').value = (saved.mode === 'grag' ? saved.settings.contrastive_learning_rate ?? 0.00025 : saved.settings.crl_learning_rate ?? 0.0003);
        render(saved);
        message('前回の学習状態を読み込みました。観察または高速学習を続けられます。');
      }
    } catch (_) { /* 保存結果がなければ学習0回から始める。 */ }
    await chooseMap();
  } catch (error) { message(error.message, true); }
})();

let valueGrid = null, valueMap = null, valueTimer = null, valueToken = 0, valueKey = null;
let valueGoal = null, valueSnapshotGoal = null;
let positiveCountKey = null, positiveCountMap = null, positiveCountGrid = null;
function drawPositiveCountMap() {
  if (!positiveCountGrid || $('value-section').hidden) return;
  const zoom = Number($('value-zoom').value || 100) / 100;
  const {max, total} = drawCountHeatmap($('positive-count-map'), positiveCountMap, positiveCountGrid, zoom);
  $('positive-count-max').textContent = max.toLocaleString() + '件';
  $('positive-count-total').textContent = '累計 ' + total.toLocaleString() + '件';
}
function renderPositiveCounts(data) {
  const counts = data.cumulative_positive_goal_counts;
  const key = data.session_id + ':' + data.map.name + ':' + JSON.stringify(counts);
  if (key === positiveCountKey) return;
  positiveCountKey = key;
  $('positive-count-stage').hidden = !counts?.length;
  $('positive-count-legend').hidden = !counts?.length;
  if (!counts?.length) {
    positiveCountMap = null; positiveCountGrid = null;
    $('positive-count-total').textContent = '';
    $('positive-count-status').textContent = '正例数の記録はまだありません。';
    return;
  }

  positiveCountMap = data.map;
  positiveCountGrid = Array.from({length: data.map.rows}, () => Array(data.map.cols).fill(0));
  for (const {goal: [row, col], count} of counts) {
    positiveCountGrid[row][col] = count;
  }
  $('positive-count-status').textContent = 'この学習セッションで各マスをHER目標として使った累計です。';
  drawPositiveCountMap();
}
function scheduleValueMap(data) {
  const visible = data?.mode === 'grag' && matches(data);
  $('value-section').hidden = !visible;
  if (!visible) { valueToken++; valueGrid = null; valueKey = null; closeValuePopup(); $('value-warning').hidden = true; return; }
  renderPositiveCounts(data);
  if (!valueGoal) {
    valueGoal = [...data.map.goal];
    $('value-goal-row').value = valueGoal[0]; $('value-goal-col').value = valueGoal[1];
    $('value-goal-row').max = data.map.rows - 1; $('value-goal-col').max = data.map.cols - 1;
  }
  const key = data.session_id + ':' + data.total_steps + ':' + valueGoal.join(',');
  if (key === valueKey) return;
  valueKey = key;
  if (!valueTimer) valueTimer = setTimeout(() => { valueTimer = null; refreshValueMap(result); }, 1000);
}
async function refreshValueMap(data = result) {
  if (!data || data.mode !== 'grag' || !matches(data)) return;
  const token = ++valueToken;
  $('value-status').textContent = '価値を取得しています。';
  try {
    const requestedGoal = [...valueGoal];
    const snapshot = await api('/api/train/grag-values', {session_id: data.session_id, mode: data.mode, map_name: data.map.name, goal: requestedGoal});
    if (token !== valueToken || !matches(result) || result.session_id !== data.session_id) return;
    if (!snapshot.values) {
      $('value-warning').textContent = snapshot.warning || '価値ヒートマップを取得できませんでした。';
      $('value-warning').hidden = false;
      $('value-status').textContent = valueGrid ? '更新できませんでした。前回取得した値を表示しています。' : '価値を取得できませんでした。';
      return;
    }
    valueGrid = snapshot.values; valueMap = data.map; valueSnapshotGoal = snapshot.goal || requestedGoal;
    $('value-warning').hidden = true;
    $('value-status').textContent = `目標 [${valueSnapshotGoal.join(', ')}] · 学習 ${snapshot.completed_episodes}回 / ${snapshot.total_steps}ステップ時点の値`;
    drawValueMap();
    if (selectedValueCell && valueAnchor && !$('value-popup').hidden) {
      openValueCell(selectedValueCell.row, selectedValueCell.col, valueAnchor.x, valueAnchor.y);
    }
  } catch (error) {
    if (token === valueToken && matches(result) && result.session_id === data.session_id) {
      $('value-warning').textContent = '価値ヒートマップを取得できませんでした。';
      $('value-warning').hidden = false;
      $('value-status').textContent = valueGrid ? '更新できませんでした。前回取得した値を表示しています。' : '価値を取得できませんでした。';
    }
  }
}
let selectedValueCell = null, valueAnchor = null;
function drawValueMap() {
  const canvas = $('value-map'), context = canvas.getContext('2d');
  if (!valueGrid || !valueMap) { context.clearRect(0, 0, canvas.width, canvas.height); return; }
  const zoom = Number($('value-zoom').value || 100) / 100;
  const stage = canvas.parentElement, style = getComputedStyle(stage);
  const availableWidth = Math.max(1, stage.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight));
  const heightLimit = parseFloat(style.maxHeight);
  const availableHeight = Number.isFinite(heightLimit) ? Math.max(1, heightLimit - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom) - (stage.offsetHeight - stage.clientHeight)) : availableWidth;
  const unit = Math.min(availableWidth / valueMap.cols, availableHeight / valueMap.rows) * zoom;
  const width = valueMap.cols * unit, height = valueMap.rows * unit;
  const density = Math.min(window.devicePixelRatio || 1, 2, 4096 / Math.max(width, height));
  const pw = Math.round(width * density), ph = Math.round(height * density);
  if (canvas.width !== pw) canvas.width = pw;
  if (canvas.height !== ph) canvas.height = ph;
  canvas.style.width = width + 'px'; canvas.style.height = height + 'px';
  context.setTransform(density, 0, 0, density, 0, 0);
  // 正規化された表現の内積と、その方策期待値は -1〜1 に収まる。
  $('value-min').textContent = '-1';
  $('value-max').textContent = '+1';
  $('value-zoom-value').value = Math.round(zoom * 100) + '%';
  context.fillStyle = '#dce6eb'; context.fillRect(0, 0, width, height);
  for (let r = 0; r < valueMap.rows; r++) for (let c = 0; c < valueMap.cols; c++) {
    const cell = valueGrid[r][c], ratio = cell ? Math.max(0, Math.min(1, (cell.value + 1) / 2)) : 0;
    context.fillStyle = cell ? 'rgb(' + Math.round(255 - 35 * ratio) + ',' + Math.round(255 - 158 * ratio) + ',' + Math.round(255 - 180 * ratio) + ')' : '#354f60';
    context.fillRect(c * unit + 1, r * unit + 1, unit - 2, unit - 2);
    if (cell) {
      context.fillStyle = ratio > .7 ? '#fff' : '#344e5b';
      context.font = '600 ' + Math.max(1, Math.min(38 * zoom, unit * .32)) + 'px sans-serif';
      context.textAlign = 'center'; context.textBaseline = 'middle';
      context.fillText(cell.value.toPrecision(3), (c + .5) * unit, (r + .5) * unit, Math.max(1, unit - 8));
    }
  }
  if (valueSnapshotGoal) {
    const [r, c] = valueSnapshotGoal;
    context.fillStyle = '#078878'; context.beginPath();
    context.arc((c + 0.84) * unit, (r + 0.16) * unit, Math.max(2, unit * .08), 0, Math.PI * 2);
    context.fill();
  }
  if (selectedValueCell) { context.strokeStyle = '#078878'; context.lineWidth = Math.max(3, unit * .045); context.strokeRect(selectedValueCell.col * unit + 3, selectedValueCell.row * unit + 3, unit - 6, unit - 6); }
}
function closeValuePopup() { $('value-popup').hidden = true; selectedValueCell = null; valueAnchor = null; drawValueMap(); }
function openValueCell(row, col, x, y) {
  const cell = valueGrid?.[row]?.[col]; if (!cell) return;
  selectedValueCell = {row, col}; valueAnchor = {x, y}; drawValueMap();
  $('value-title').textContent = `マス [${row}, ${col}] の行動価値`;
  $('value-snapshot').textContent = $('value-status').textContent;
  const directions = [['up','↑ 上'],['down','↓ 下'],['left','← 左'],['right','→ 右']];
  const qValues = directions.map(([key]) => cell.q_values[key]);
  const best = Math.max(...qValues);
  const worst = Math.min(...qValues);
  const scaleRange = best - worst;
  $('value-scale').textContent = scaleRange === 0 ? '4方向の Q 値は同じです。' :
    `このマスの範囲：${worst.toPrecision(9)} 〜 ${best.toPrecision(9)}`;
  $('value-actions').replaceChildren(...directions.map(([key,label]) => {
    const row = document.createElement('div'), name = document.createElement('span');
    const track = document.createElement('div'), fill = document.createElement('div');
    const qValue = document.createElement('strong');
    const advantage = document.createElement('small');
    const q = cell.q_values[key];
    const width = scaleRange === 0 ? 100 : (q - worst) / scaleRange * 100;
    row.className = 'value-action' + (q === best ? ' best' : '');
    name.className = 'action-label'; name.textContent = label;
    track.className = 'value-q-track'; track.setAttribute('aria-hidden', 'true');
    fill.className = 'value-q-fill';
    fill.style.width = width + '%';
    track.append(fill);
    qValue.className = 'value-q-number'; qValue.textContent = 'Q ' + q.toPrecision(9);
    advantage.className = 'value-advantage';
    advantage.textContent = 'A ' + (cell.advantages[key] > 0 ? '+' : '') + cell.advantages[key].toPrecision(3);
    row.append(name, track, qValue, advantage); return row;
  }));
  const popup = $('value-popup'); popup.hidden = false;
  const rect = popup.getBoundingClientRect(), margin = 12;
  popup.style.left = Math.max(margin, Math.min(x + 14, window.innerWidth - rect.width - margin)) + 'px';
  popup.style.top = Math.max(margin, Math.min(y + 14, window.innerHeight - rect.height - margin)) + 'px';
}
function setDisplayGoal(row, col) {
  if (!result || !matches(result) || result.mode !== 'grag') return;
  const map = result.map;
  if (!integer(row, 0, map.rows - 1) || !integer(col, 0, map.cols - 1) ||
      map.obstacles.some(([r, c]) => r === row && c === col)) return;
  $('value-goal-row').value = row; $('value-goal-col').value = col;
  $('value-goal-error').hidden = true;
  valueGoal = [row, col]; valueKey = null; valueToken++;
  redrawHeat();
  if (selectedHeatCell && !$('policy-popup').hidden)
    openPolicyCell(selectedHeatCell.row, selectedHeatCell.col, policyAnchor);
  refreshValueMap();
}
$('policy-set-goal').onclick = () => {
  if (selectedHeatCell) setDisplayGoal(selectedHeatCell.row, selectedHeatCell.col);
};
$('value-set-goal').onclick = () => {
  if (selectedValueCell) setDisplayGoal(selectedValueCell.row, selectedValueCell.col);
};
$('value-refresh').onclick = () => refreshValueMap();
$('value-goal-apply').onclick = () => {
  if (!result || !matches(result)) return;
  const rowInput = $('value-goal-row').value, colInput = $('value-goal-col').value;
  const row = Number(rowInput), col = Number(colInput), map = result.map;
  if (rowInput.trim() === '' || colInput.trim() === '' || !integer(row, 0, map.rows - 1) ||
      !integer(col, 0, map.cols - 1) || map.obstacles.some(([r, c]) => r === row && c === col)) {
    $('value-goal-error').textContent = '障害物でないマップ範囲内の座標を入力してください。';
    $('value-goal-error').hidden = false; return;
  }
  setDisplayGoal(row, col);
};
$('value-goal-reset').onclick = () => {
  if (!result || !matches(result)) return;
  $('value-goal-row').value = result.map.goal[0]; $('value-goal-col').value = result.map.goal[1];
  $('value-goal-apply').click();
};
$('value-zoom').value = '100';
$('value-zoom-value').value = '100%';
$('value-zoom').oninput = () => { drawValueMap(); drawPositiveCountMap(); };
$('value-fit').onclick = () => {
  $('value-zoom').value = '100';
  $('value-zoom').oninput();
  $('value-map').parentElement.scrollTo(0, 0);
  $('positive-count-stage').scrollTo(0, 0);
};
$('value-close').onclick = closeValuePopup;
$('value-map').onclick = event => {
  if (!valueGrid || !valueMap) return;
  const rect = event.currentTarget.getBoundingClientRect();
  openValueCell(Math.floor((event.clientY - rect.top) / rect.height * valueMap.rows), Math.floor((event.clientX - rect.left) / rect.width * valueMap.cols), event.clientX, event.clientY);
};
$('value-map').onkeydown = event => { if ((event.key === 'Enter' || event.key === ' ') && valueMap) { event.preventDefault(); const rect = event.currentTarget.getBoundingClientRect(); const pos = result?.live?.position || valueMap.start; openValueCell(pos[0], pos[1], rect.left + rect.width / 2, rect.top + rect.height / 2); } };
document.addEventListener('keydown', event => { if (event.key === 'Escape' && !$('value-popup').hidden) { closeValuePopup(); $('value-map').focus({preventScroll:true}); } });
document.addEventListener('pointerdown', event => { if (!$('value-popup').hidden && !$('value-popup').contains(event.target) && event.target !== $('value-map')) closeValuePopup(); });
window.addEventListener('resize', drawValueMap);
window.addEventListener('resize', drawPositiveCountMap);
if (typeof ResizeObserver !== 'undefined') {
  new ResizeObserver(drawValueMap).observe($('value-map').parentElement);
  new ResizeObserver(drawPositiveCountMap).observe($('positive-count-stage'));
}
