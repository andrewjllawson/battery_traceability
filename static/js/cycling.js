// Cycling page (templates/cycling.html): capacity fade, ΔQ(V) and the fade forecast.
//
// Same style as the mixing page (static/js/mixing.js): dark canvas, neon colours.
// Reads only GET endpoints: /api/cycling/{id}/features, /api/cycling/{id}/delta-q,
// /api/ml/cycle-life/{id} and /api/ml/models. Chart.js is loaded by base.html.
(function () {
    const COLOUR = {
        material: '#00e5ff', coating: '#7c4dff', slp: '#00ff9d',
        coincell: '#ff6ec7', mlp: '#ffb800', failed: '#ff5252',
    };
    // One colour per fade model, the same in every chart
    const MODEL_COLOUR = {
        'Linear fade': '#00e5ff', 'Square-root fade': '#ffb800',
        'Exponential fade': '#00ff9d', 'No further fade': '#9aa6c7',
    };
    const MODEL_ORDER = ['Linear fade', 'Square-root fade', 'Exponential fade', 'No further fade'];
    const GRID = 'rgba(255,255,255,0.08)';
    const TICK = '#9aa6c7';

    const app = document.getElementById('cyclingApp');
    const charts = {};
    let cellId = null;
    let current = null;     // the /api/cycling/{id}/features reply
    let deltaQ = null;      // the /api/cycling/{id}/delta-q reply, or {error}
    let forecast = null;    // the /api/ml/cycle-life/{id} reply, or {error}
    let registry = null;    // latest cycling runs from /api/ml/models
    let view = 'fade';

    Chart.defaults.color = TICK;
    Chart.defaults.font.family = "-apple-system, 'Segoe UI', Arial, sans-serif";

    // ---- small helpers -----------------------------------------------------------

    function esc(value) {
        return String(value)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    }
    function num(value, digits, unit) {
        if (value === null || value === undefined) return '—';
        return Number(value).toFixed(digits) + (unit || '');
    }
    function draw(id, config) {
        if (charts[id]) charts[id].destroy();
        charts[id] = new Chart(document.getElementById(id), config);
        return charts[id];
    }
    function clear(id) {
        if (charts[id]) { charts[id].destroy(); delete charts[id]; }
    }
    const axis = (title, extra) => ({
        title: { display: !!title, text: title, color: TICK },
        grid: { color: GRID }, ticks: { color: TICK }, border: { color: GRID }, ...(extra || {}),
    });
    const legend = { labels: { color: '#e8e8f0', usePointStyle: true, boxWidth: 8 } };
    async function getJson(url) {
        try {
            const reply = await fetch(url);
            const data = await reply.json();
            return reply.ok ? data : { error: data.error || 'Not available.' };
        } catch (e) {
            return { error: 'Could not reach the server.' };
        }
    }
    function caption(text) { document.getElementById('cyCaption').textContent = text || ''; }

    // ---- views ------------------------------------------------------------------

    function drawFade() {
        const c = current.cycles;
        const passed = [], failed = [], efficiency = [];
        for (let i = 0; i < c.cycle_no.length; i++) {
            const point = { x: c.cycle_no[i], y: c.discharge_mah[i] };
            if (c.discharge_mah[i] !== null) (c.passed_gate[i] ? passed : failed).push(point);
            if (c.coulombic_eff[i] !== null) efficiency.push({ x: c.cycle_no[i], y: 100 * c.coulombic_eff[i] });
        }
        draw('cFade', {
            type: 'scatter',
            data: { datasets: [
                { label: 'Discharge capacity (mAh)', data: passed, backgroundColor: COLOUR.material, pointRadius: 2, yAxisID: 'y' },
                { label: 'Failed the quality gate', data: failed, backgroundColor: COLOUR.failed, pointRadius: 3, yAxisID: 'y' },
                { label: 'Coulombic efficiency (%)', data: efficiency, borderColor: COLOUR.coincell, backgroundColor: COLOUR.coincell,
                  showLine: true, borderWidth: 1, borderDash: [5, 4], pointRadius: 0, yAxisID: 'y2' },
            ] },
            options: {
                responsive: true, maintainAspectRatio: false, parsing: false, animation: false,
                scales: {
                    x: axis('Cycle number', { type: 'linear' }),
                    y: axis('Discharge capacity (mAh)'),
                    y2: axis('Coulombic efficiency (%)', { position: 'right', grid: { drawOnChartArea: false } }),
                },
                plugins: { legend },
            },
        });
        caption('Red points failed the quality gate and are left out of the model. Click a name in the legend to show or hide it.');
    }

    function drawDeltaQ() {
        if (!deltaQ || deltaQ.error) {
            clear('cDeltaQ');
            caption(deltaQ ? deltaQ.error : 'Loading…');
            return;
        }
        const points = deltaQ.voltage_v.map((v, i) => ({ x: v, y: deltaQ.delta_q_mah[i] }));
        draw('cDeltaQ', {
            type: 'line',
            data: { datasets: [
                { label: `ΔQ, cycle ${deltaQ.late_cycle} minus cycle ${deltaQ.early_cycle} (mAh)`, data: points,
                  borderColor: COLOUR.coating, backgroundColor: COLOUR.coating, borderWidth: 2, pointRadius: 0 },
            ] },
            options: {
                responsive: true, maintainAspectRatio: false, parsing: false, animation: false,
                scales: { x: axis('Voltage (V)', { type: 'linear', reverse: true }), y: axis('ΔQ (mAh)') },
                plugins: { legend },
            },
        });
        caption('How much less charge the cell gives at each voltage after 100 cycles than after 10. Its variance is the strongest early sign of cycle life in Severson et al. (2019).');
    }

    function bestRun() {
        if (!registry || !registry.length) return null;
        const forms = registry.filter(r => r.model !== 'No further fade' && r.metric !== null);
        if (!forms.length) return null;
        return forms.reduce((a, b) => (b.metric < a.metric ? b : a));
    }

    function drawModel() {
        const stats = document.getElementById('cyModelStats');
        const chips = [];
        if (forecast && !forecast.error) {
            const f = forecast;
            chips.push(`<span class="mx-chip">${esc(f.model)}</span>`);
            if (f.eol_cycle !== null) {
                chips.push(`<span class="mx-chip ${f.extrapolated ? '' : 'good'}">End of life ≈ cycle ${num(f.eol_cycle, 0)}</span>`);
                chips.push(`<span class="mx-chip">80% band: ${num(f.eol_low, 0)}–${num(f.eol_high, 0)}</span>`);
            }
            if (f.measured_eol_cycle !== null) chips.push(`<span class="mx-chip">Measured: cycle ${num(f.measured_eol_cycle, 0)}</span>`);
            if (f.warning) chips.push(`<span class="mx-chip warn">${esc(f.warning)}</span>`);
        } else if (forecast) {
            chips.push(`<span class="mx-chip">${esc(forecast.error)}</span>`);
        }
        const best = bestRun();
        if (best) {
            const better = best.baseline_metric !== null && best.metric < best.baseline_metric;
            chips.push(`<span class="mx-chip ${better ? 'good' : ''}">${esc(best.model)}: ${num(best.metric, 1)}% error on later cycles</span>`);
            chips.push(`<span class="mx-chip">No further fade: ${num(best.baseline_metric, 1)}%</span>`);
            chips.push(`<span class="mx-chip">${best.n_groups} cell${best.n_groups === 1 ? '' : 's'}</span>`);
        } else {
            chips.push('<span class="mx-chip">No scored runs yet (python3 -m scripts.train_cycle_life)</span>');
        }
        stats.innerHTML = chips.join('');

        drawForecast();
        drawRegistry(best);

        const notes = [];
        if (forecast && forecast.method) notes.push(forecast.method);
        if (best) notes.push(`Scores from run ${best.run_id}, ${best.trained_at.slice(0, 10)}, fitted on early cycles and scored on the cycles after them.`);
        document.getElementById('cyModelNote').textContent = notes.join(' ');
    }

    function drawForecast() {
        if (!forecast || forecast.error || !current) { clear('cForecast'); return; }
        const f = forecast, c = current.cycles, curve = f.curve;
        const measured = [];
        for (let i = 0; i < c.cycle_no.length; i++) {
            if (c.passed_gate[i] && c.discharge_mah[i] !== null) measured.push({ x: c.cycle_no[i], y: c.discharge_mah[i] });
        }
        const line = key => curve.cycle.map((x, i) => ({ x, y: curve[key][i] }));
        const colour = MODEL_COLOUR[f.model];
        const last = curve.cycle[curve.cycle.length - 1];
        draw('cForecast', {
            type: 'scatter',
            data: { datasets: [
                { label: 'Measured (mAh)', data: measured, backgroundColor: '#e8e8f0', pointRadius: 1.5 },
                { label: f.model, data: line('fit'), borderColor: colour, showLine: true, pointRadius: 0, borderWidth: 2 },
                { label: '80% band', data: line('low'), borderColor: 'transparent', backgroundColor: colour + '33',
                  showLine: true, pointRadius: 0, fill: '+1' },
                { label: '80% band (top)', data: line('high'), borderColor: 'transparent', showLine: true, pointRadius: 0 },
                { label: 'End of life (80%)', data: [{ x: curve.cycle[0], y: f.threshold_mah }, { x: last, y: f.threshold_mah }],
                  borderColor: COLOUR.failed, borderDash: [5, 4], showLine: true, pointRadius: 0, borderWidth: 1 },
            ] },
            options: {
                responsive: true, maintainAspectRatio: false, parsing: false, animation: false,
                scales: { x: axis('Cycle number', { type: 'linear' }), y: axis('Discharge capacity (mAh)') },
                plugins: { legend: { ...legend, labels: { ...legend.labels, filter: item => item.text !== '80% band (top)' } } },
            },
        });
    }

    function drawRegistry(best) {
        if (!best) { clear('cModelScatter'); clear('cModelErrors'); return; }
        const s = best.scatter || { actual: [], predicted: [] };
        const values = [...s.actual, ...s.predicted];
        if (!values.length) {
            clear('cModelScatter');
        } else {
            drawScatter(best, s, Math.min(...values), Math.max(...values));
        }
        drawErrors();
    }

    function drawScatter(best, s, low, high) {
        draw('cModelScatter', {
            type: 'scatter',
            data: { datasets: [
                { label: 'Later cycles', data: s.actual.map((a, i) => ({ x: a, y: s.predicted[i] })),
                  backgroundColor: MODEL_COLOUR[best.model] + '99', pointRadius: 2 },
                { label: 'Perfect', type: 'line', data: [{ x: low, y: low }, { x: high, y: high }],
                  borderColor: '#e8e8f0', borderDash: [4, 4], pointRadius: 0, borderWidth: 1 },
            ] },
            options: { responsive: true, maintainAspectRatio: false, parsing: false, animation: false,
                scales: { x: axis('Measured (mAh)', { type: 'linear' }), y: axis(`Predicted by ${best.model} (mAh)`) },
                plugins: { legend: { display: false } } },
        });
    }

    function drawErrors() {
        const runs = MODEL_ORDER.map(m => registry.find(r => r.model === m)).filter(r => r);
        draw('cModelErrors', {
            type: 'bar',
            data: { labels: runs.map(r => r.model), datasets: [{ label: 'Error on later cycles (%)',
                data: runs.map(r => r.metric), backgroundColor: runs.map(r => MODEL_COLOUR[r.model]), borderRadius: 4 }] },
            options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                scales: { x: axis('Error on later cycles, MAPE (%)', { beginAtZero: true }), y: axis() },
                plugins: { legend: { display: false } } },
        });
    }

    // ---- summary under the list -------------------------------------------------

    function renderPanel() {
        const row = (label, value) => `<div class="row"><span class="label">${esc(label)}</span><span class="value">${value}</span></div>`;
        let html = `<div class="mx-head">${esc(cellId)}</div>`;
        if (!current || current.error) {
            html += `<p class="mx-note">${esc(current ? current.error : '')}</p>`;
        } else {
            const passed = current.cycles.passed_gate.filter(p => p).length;
            html += row('File', esc(current.filename || '—')) +
                row('Uploaded', esc(current.uploaded_at.slice(0, 10))) +
                row('Cycles', current.n_cycles) +
                row('Passed the gate', passed) +
                row('Early-life features', current.features ? 'yes' : 'no');
            if (current.feature_error) html += `<p class="mx-note">${esc(current.feature_error)}</p>`;
            html += '<div class="mx-head" style="margin-top: 12px;">QUALITY GATE</div>';
            current.gate.forEach(step => {
                const mark = step.passed ? '<span class="gate-pass">✓</span>' : '<span class="gate-fail">✗</span>';
                const rows = step.rows_in === null || step.rows_in === undefined ? '' : `${step.rows_in}→${step.rows_out}`;
                html += row(`${step.check}`, `${rows} ${mark}`);
            });
        }
        html += `<p class="mx-note" style="margin-top: 10px;"><a class="mx-link" href="/card/${encodeURIComponent(cellId)}">Open the cell's card</a></p>`;
        document.getElementById('cyPanelBody').innerHTML = html;
        document.getElementById('cyPanel').hidden = false;
    }

    // ---- wiring -----------------------------------------------------------------

    function show(name) {
        view = name;
        document.querySelectorAll('.mx-tab').forEach(t => t.classList.toggle('active', t.dataset.view === name));
        document.querySelectorAll('.mx-view').forEach(v => { v.hidden = v.id !== 'view-' + name; });
        const ready = current && !current.error;
        document.getElementById('cyEmpty').hidden = !!ready || (name === 'model' && !!cellId);
        if (!cellId) { caption(''); return; }
        if (!ready && name !== 'model') {
            caption(current && current.error ? current.error : '');
            return;
        }
        if (name === 'fade') drawFade();
        if (name === 'dq') {
            drawDeltaQ();
            if (!deltaQ) loadDeltaQ();
        }
        if (name === 'model') { caption(''); drawModel(); }
    }

    async function loadDeltaQ() {
        const id = cellId;
        const data = await getJson('/api/cycling/' + encodeURIComponent(id) + '/delta-q');
        if (id !== cellId) return;  // another cell was chosen meanwhile
        deltaQ = data;
        if (view === 'dq') drawDeltaQ();
    }

    async function loadCell(id) {
        cellId = id;
        deltaQ = null;
        document.querySelectorAll('.mx-run').forEach(a => a.classList.toggle('active', a.dataset.id === id));
        history.replaceState(null, '', '/cycling?cell=' + encodeURIComponent(id));
        const [features, cellForecast] = await Promise.all([
            getJson('/api/cycling/' + encodeURIComponent(id) + '/features'),
            getJson('/api/ml/cycle-life/' + encodeURIComponent(id)),
        ]);
        if (id !== cellId) return;
        current = features;
        forecast = cellForecast;
        renderPanel();
        show(view);
    }

    async function loadRegistry() {
        const data = await getJson('/api/ml/models?data_type=cycling&latest=1');
        registry = data.error ? [] : data.runs;
        if (view === 'model') drawModel();
    }

    document.querySelectorAll('.mx-tab').forEach(t => t.addEventListener('click', () => show(t.dataset.view)));
    document.querySelectorAll('.mx-run').forEach(a => a.addEventListener('click', e => {
        e.preventDefault();
        loadCell(a.dataset.id);
    }));

    loadRegistry();
    const wanted = app.dataset.selected || (document.querySelector('.mx-run') || {}).dataset?.id;
    if (wanted) loadCell(wanted); else show('fade');
})();
