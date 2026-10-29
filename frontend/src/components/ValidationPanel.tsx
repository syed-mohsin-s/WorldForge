import React, { useState } from 'react';
import { Wrench } from 'lucide-react';
import {
  FaultInjectionMode,
  PricingTier,
  SceneGraph,
  SelfHealingResult,
  runSelfHealingLoop,
  runValidationSuite,
} from '../engine/worldforge';

interface ValidationPanelProps {
  scene: SceneGraph;
  tier: PricingTier;
  onRepairComplete: (result: SelfHealingResult) => void;
}

export const ValidationPanel: React.FC<ValidationPanelProps> = ({
  scene,
  tier,
  onRepairComplete,
}) => {
  const [faultMode, setFaultMode] = useState<FaultInjectionMode>('none');
  const [healingResult, setHealingResult] = useState<SelfHealingResult | null>(null);
  const [hoveredPointIdx, setHoveredPointIdx] = useState<number | null>(null);

  const currentReport = healingResult
    ? healingResult.finalReport
    : runValidationSuite(scene, faultMode, true);

  const handleRunSelfHealing = () => {
    const activeFault = faultMode === 'none' ? 'gate3_zero_reward' : faultMode;
    const res = runSelfHealingLoop(scene, activeFault, tier, 4);
    setHealingResult(res);
    onRepairComplete(res);
  };

  const handleSelectFault = (mode: FaultInjectionMode) => {
    setFaultMode(mode);
    setHealingResult(null);
  };

  // Dual-axis SVG chart geometry for Gate 5 PPO Convergence
  const chartW = 520;
  const chartH = 190;
  const padL = 44;
  const padR = 20;
  const padT = 18;
  const padB = 28;
  const plotW = chartW - padL - padR;
  const plotH = chartH - padT - padB;
  const maxReturn = 60;

  const curvePts = currentReport.learningCurve.map((pt, idx, arr) => {
    const x = padL + (idx / Math.max(arr.length - 1, 1)) * plotW;
    const y = padT + plotH - (Math.min(pt.meanReturn, maxReturn) / maxReturn) * plotH;
    const yBase =
      padT + plotH - (Math.min(pt.randomBaseline, maxReturn) / maxReturn) * plotH;
    return { ...pt, x, y, yBase };
  });

  const ppoPolyline = curvePts.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ');
  const basePolyline = curvePts
    .map((p) => `${p.x.toFixed(1)},${p.yBase.toFixed(1)}`)
    .join(' ');
  const ppoArea =
    curvePts.length > 1
      ? `${curvePts[0].x.toFixed(1)},${(padT + plotH).toFixed(1)} ${ppoPolyline} ${curvePts[
          curvePts.length - 1
        ].x.toFixed(1)},${(padT + plotH).toFixed(1)}`
      : '';

  return (
    <div className="divide-y divide-slate-800/80 bg-[#07090E]">
      {/* Section Header & Fault Injection Control Bar */}
      <div className="px-6 py-4 bg-[#0B0E17] flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-100">
            02. Sequential 5-Gate Validation &amp; Automated Self-Healing Loop
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Evaluates numerical stability, resting equilibrium, reward variance, closed-loop oracle solvability, and PPO convergence.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <label htmlFor="fault-injection-select" className="text-xs text-slate-300">
              Inject Test Fault:
            </label>
            <select
              id="fault-injection-select"
              value={faultMode}
              onChange={(e) => handleSelectFault(e.target.value as FaultInjectionMode)}
              className="bg-[#07090E] border border-slate-800 rounded px-3 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500"
            >
              <option value="none">None (Nominal SceneGraph)</option>
              <option value="gate1_instability">Gate 1: Sub-5mm Geom Instability</option>
              <option value="gate2_drift">Gate 2: Low Free-Joint Damping Drift</option>
              <option value="gate3_zero_reward">Gate 3: Constant-Zero Degenerate Reward</option>
            </select>
          </div>

          <button
            type="button"
            onClick={handleRunSelfHealing}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-colors cursor-pointer whitespace-nowrap shrink-0"
          >
            <Wrench className="w-3.5 h-3.5" />
            Run Self-Healing Repair Loop
          </button>
        </div>
      </div>

      {/* 5-Gate Hairline Matrix (Zero nested cards) */}
      <div className="grid grid-cols-1 md:grid-cols-5 divide-y md:divide-y-0 md:divide-x divide-slate-800/80">
        {[1, 2, 3, 4, 5].map((gateNum) => {
          const gate = currentReport.gates.find((g) => g.gateNumber === gateNum);
          const isSkipped = !gate;
          const passed = gate?.passed ?? false;

          return (
            <div
              key={gateNum}
              data-testid={`validation-gate-${gateNum}`}
              className="p-4 flex flex-col justify-between bg-[#07090E]"
            >
              <div>
                <div className="flex items-center justify-between text-xs font-mono mb-1.5">
                  <span className="text-slate-400">GATE 0{gateNum}</span>
                  <span
                    className={
                      isSkipped
                        ? 'text-slate-500'
                        : passed
                          ? 'text-emerald-400 font-semibold'
                          : 'text-rose-400 font-semibold'
                    }
                  >
                    {isSkipped ? '○ BLOCKED' : passed ? '● PASSED' : '✖ FAILED'}
                  </span>
                </div>

                <div className="text-xs font-semibold text-slate-100 mb-3">
                  {gate?.name ??
                    (gateNum === 2
                      ? 'Static Equilibrium (250 Steps)'
                      : gateNum === 3
                        ? 'Non-Degenerate Reward Variance'
                        : gateNum === 4
                          ? 'Waypoint Oracle Solvability'
                          : 'Fast PPO Convergence')}
                </div>

                {gate && (
                  <dl className="space-y-1 text-[11px] font-mono tabular-nums">
                    {Object.entries(gate.metrics).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-2 text-slate-300">
                        <dt className="text-slate-500">{k}</dt>
                        <dd className="text-slate-200">{String(v)}</dd>
                      </div>
                    ))}
                  </dl>
                )}

                {gate?.reason && (
                  <p className="mt-2.5 text-[11px] text-rose-300 font-mono leading-snug">
                    {gate.reason}
                  </p>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Bottom Split: Self-Healing Repair Trace & Dual-Curve SVG PPO Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-slate-800/80">
        {/* Left: Self-Healing Repair Log */}
        <div className="lg:col-span-6 p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Self-Healing Diagnostic &amp; Patch Trace
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Module: <code className="text-slate-200">src/validation/repair_loop.py</code> (cap N=4 tries)
              </p>
            </div>
            {healingResult && (
              <span className="text-xs font-mono text-emerald-400 tabular-nums">
                ● Recovered in {healingResult.attempts} attempts
              </span>
            )}
          </div>

          {!healingResult ? (
            <div className="border border-slate-800/80 bg-[#0B0E17] rounded p-4 text-xs text-slate-300 space-y-2">
              <p>
                Select a fault injection mode above or click <strong>Run Self-Healing Repair Loop</strong> to inspect the automated diagnostic prompt and deterministic parameter/reward repair.
              </p>
              <p className="font-mono text-[11px] text-slate-400">
                Active Diagnostics: {currentReport.diagnostics}
              </p>
            </div>
          ) : (
            <div className="divide-y divide-slate-800/80 border border-slate-800/80 rounded bg-[#0B0E17]" data-testid="self-healing-trace">
              {healingResult.history.map((item) => (
                <div key={item.attempt} className="p-4 space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="font-semibold text-slate-100">
                      Attempt #{item.attempt}
                    </span>
                    <span
                      className={
                        item.report.passed ? 'text-emerald-400' : 'text-amber-400'
                      }
                    >
                      {item.report.passed
                        ? '● ALL 5 GATES NOMINAL'
                        : `▲ Gate ${item.report.failedGate} Diagnostic Triggered`}
                    </span>
                  </div>

                  <div className="text-xs font-mono text-slate-300">
                    {item.report.diagnostics}
                  </div>

                  {item.repairApplied && (
                    <div className="text-xs font-mono text-cyan-300 pt-1">
                      Patch Applied: {item.repairApplied}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Scientific SVG Dual-Curve PPO Convergence Plot */}
        <div className="lg:col-span-6 p-6 flex flex-col justify-between space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Gate 5: DummyVecEnv PPO Convergence Curve (8,192 Steps)
              </h3>
              <p className="text-xs text-slate-400 mt-0.5">
                Empirical rollout return (4 parallel envs) vs random policy baseline.
              </p>
            </div>
            <div className="flex items-center gap-3 text-[11px] font-mono">
              <span className="text-cyan-400">— PPO Policy</span>
              <span className="text-slate-400">--- Random Baseline</span>
            </div>
          </div>

          {curvePts.length === 0 ? (
            <div className="h-44 border border-slate-800/80 bg-[#0B0E17] rounded flex items-center justify-center text-xs text-slate-400 font-mono">
              Gate 5 execution blocked until Gates 1–4 pass.
            </div>
          ) : (
            <div className="relative">
              <svg
                viewBox={`0 0 ${chartW} ${chartH}`}
                className="w-full h-48 bg-[#0B0E17] border border-slate-800/80 rounded overflow-visible"
              >
                <defs>
                  <linearGradient id="ppoAreaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.24" />
                    <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Horizontal Y-axis gridlines & labels */}
                {[0, 20, 40, 60].map((val) => {
                  const y = padT + plotH - (val / maxReturn) * plotH;
                  return (
                    <g key={val}>
                      <line
                        x1={padL}
                        y1={y}
                        x2={chartW - padR}
                        y2={y}
                        stroke="#1e293b"
                        strokeWidth="1"
                      />
                      <text
                        x={padL - 8}
                        y={y + 3}
                        textAnchor="end"
                        className="fill-slate-500 text-[10px] font-mono"
                      >
                        {val.toFixed(0)}
                      </text>
                    </g>
                  );
                })}

                {/* Area under PPO curve */}
                {ppoArea && <polygon points={ppoArea} fill="url(#ppoAreaGrad)" />}

                {/* Random baseline dashed curve */}
                <polyline
                  fill="none"
                  stroke="#64748b"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                  points={basePolyline}
                />

                {/* PPO learning curve */}
                <polyline
                  fill="none"
                  stroke="#06b6d4"
                  strokeWidth="2.2"
                  points={ppoPolyline}
                />

                {/* Interactive vertices */}
                {curvePts.map((pt, idx) => (
                  <g
                    key={pt.step}
                    onMouseEnter={() => setHoveredPointIdx(idx)}
                    onMouseLeave={() => setHoveredPointIdx(null)}
                    className="cursor-pointer"
                  >
                    <circle
                      cx={pt.x}
                      cy={pt.y}
                      r={hoveredPointIdx === idx ? 5 : 3.5}
                      fill="#06b6d4"
                      stroke="#07090e"
                      strokeWidth="1.5"
                    />
                    <text
                      x={pt.x}
                      y={chartH - 8}
                      textAnchor="middle"
                      className="fill-slate-400 text-[10px] font-mono"
                    >
                      {pt.step}
                    </text>
                  </g>
                ))}
              </svg>

              {/* Precision Readout Footer */}
              <div className="mt-2 flex items-center justify-between text-xs font-mono text-slate-400 tabular-nums">
                <span>
                  {hoveredPointIdx !== null
                    ? `Step ${curvePts[hoveredPointIdx].step}: Return = +${curvePts[
                        hoveredPointIdx
                      ].meanReturn.toFixed(2)}`
                    : `Final Eval Return (Step 8,192): +${curvePts[
                        curvePts.length - 1
                      ].meanReturn.toFixed(2)}`}
                </span>
                <span className="text-emerald-400">
                  Δ vs Baseline: +
                  {(
                    curvePts[curvePts.length - 1].meanReturn -
                    curvePts[curvePts.length - 1].randomBaseline
                  ).toFixed(2)}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
