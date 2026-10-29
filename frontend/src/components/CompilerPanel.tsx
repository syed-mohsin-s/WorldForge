import React, { useState } from 'react';
import { Copy, Check, Plus, Shuffle } from 'lucide-react';
import {
  MATERIAL_DB,
  ObjectNode,
  PrimitiveType,
  SceneGraph,
  compileSceneToMjcf,
  snapSceneToTable,
} from '../engine/worldforge';

interface CompilerPanelProps {
  scene: SceneGraph;
  onAddObject: (obj: ObjectNode) => void;
}

export const CompilerPanel: React.FC<CompilerPanelProps> = ({ scene, onAddObject }) => {
  const [randomize, setRandomize] = useState(false);
  const [seed, setSeed] = useState(42);
  const [copied, setCopied] = useState(false);

  // Add Object form state
  const [newId, setNewId] = useState('');
  const [newLabel, setNewLabel] = useState('');
  const [newPrimitive, setNewPrimitive] = useState<PrimitiveType>('box');
  const [newMaterial, setNewMaterial] = useState<string>('plastic');
  const [newPosX, setNewPosX] = useState('0.08');
  const [newPosY, setNewPosY] = useState('-0.08');
  const [formError, setFormError] = useState<string | null>(null);

  const { entries } = snapSceneToTable(scene);
  const mjcfXml = compileSceneToMjcf(scene, randomize, seed);

  const handleCopyXml = () => {
    navigator.clipboard?.writeText(mjcfXml);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleCreateObject = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedId = newId.trim().replace(/\s+/g, '_');
    const trimmedLabel = newLabel.trim();

    if (!trimmedId || !trimmedLabel) {
      setFormError('Object ID and descriptive label are both required.');
      return;
    }
    if (scene.objects.some((o) => o.id === trimmedId)) {
      setFormError(`Object ID "${trimmedId}" already exists in this scene.`);
      return;
    }

    const px = Number(newPosX);
    const py = Number(newPosY);
    if (Number.isNaN(px) || Number.isNaN(py)) {
      setFormError('X and Y coordinates must be valid numbers.');
      return;
    }

    setFormError(null);
    const defaultSize =
      newPrimitive === 'box'
        ? [0.025, 0.025, 0.025]
        : newPrimitive === 'cylinder'
          ? [0.022, 0.03]
          : [0.025];

    onAddObject({
      id: trimmedId,
      label: trimmedLabel,
      primitive: newPrimitive,
      size: defaultSize,
      position: [px, py, 0.0],
      quaternion: [1, 0, 0, 0],
      material: newMaterial,
      is_static: false,
      joint: { type: 'free', axis: [0, 0, 1], damping: 0.5, is_prior_guess: false },
      parent_id: null,
    });

    setNewId('');
    setNewLabel('');
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 divide-y xl:divide-y-0 xl:divide-x divide-slate-800/80 bg-[#07090E]">
      {/* Left column: SceneGraph Auto-Snap Table + Add Primitive Form */}
      <div className="xl:col-span-6 divide-y divide-slate-800/80">
        <div className="p-6">
          <div className="mb-4">
            <h2 className="text-sm font-semibold text-slate-100">
              01. Stage 4: Pydantic V2 SceneGraph &amp; Auto-Snap Grounding
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Snaps resting root bodies to <code className="text-slate-200">table_z + half_h + 0.001m</code> to prevent Frame-0 contact explosions.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400">
                  <th className="py-2 pr-3 font-medium">Object ID</th>
                  <th className="py-2 px-3 font-medium">Primitive</th>
                  <th className="py-2 px-3 font-medium">Half-Height</th>
                  <th className="py-2 px-3 font-medium">Raw Z</th>
                  <th className="py-2 pl-3 font-medium">Snapped Z</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono tabular-nums">
                {entries.map((entry) => (
                  <tr key={entry.id} className="text-slate-200">
                    <td className="py-2.5 pr-3 font-semibold text-cyan-300">{entry.id}</td>
                    <td className="py-2.5 px-3 text-slate-300">{entry.primitive}</td>
                    <td className="py-2.5 px-3 text-slate-300">{entry.halfHeight.toFixed(3)} m</td>
                    <td className="py-2.5 px-3 text-slate-400">{entry.rawZ.toFixed(3)} m</td>
                    <td className="py-2.5 pl-3 text-emerald-400">
                      {entry.snappedZ.toFixed(3)} m
                      {entry.isChild ? ' (child rel)' : entry.wasSnapped ? ' (+1mm snap)' : ''}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Add Rigid-Body Primitive Form */}
        <form onSubmit={handleCreateObject} className="p-6 space-y-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-100">
              Add Tabletop Primitive to SceneGraph
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              New objects initialize at <code className="text-slate-300">z = 0.0</code> and ground automatically via <code className="text-slate-300">_snap_objects_to_table</code> into the 3D WebGL stage.
            </p>
          </div>

          {formError && (
            <div
              role="alert"
              className="text-xs text-rose-300 bg-rose-950/60 border border-rose-800/70 rounded px-3 py-2"
            >
              {formError}
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
            <div>
              <label htmlFor="new-obj-id" className="block text-xs text-slate-300 mb-1">
                Object ID
              </label>
              <input
                id="new-obj-id"
                type="text"
                value={newId}
                onChange={(e) => setNewId(e.target.value)}
                placeholder="e.g. ceramic_mug"
                className="w-full bg-[#0B0E17] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 font-mono focus:outline-none focus:border-cyan-500"
              />
            </div>
            <div>
              <label htmlFor="new-obj-label" className="block text-xs text-slate-300 mb-1">
                Semantic Label
              </label>
              <input
                id="new-obj-label"
                type="text"
                value={newLabel}
                onChange={(e) => setNewLabel(e.target.value)}
                placeholder="e.g. white ceramic mug"
                className="w-full bg-[#0B0E17] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
              />
            </div>
            <div>
              <label htmlFor="new-obj-primitive" className="block text-xs text-slate-300 mb-1">
                MuJoCo Primitive
              </label>
              <select
                id="new-obj-primitive"
                value={newPrimitive}
                onChange={(e) => setNewPrimitive(e.target.value as PrimitiveType)}
                className="w-full bg-[#0B0E17] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                <option value="box">box [hx, hy, hz]</option>
                <option value="cylinder">cylinder [radius, half_len]</option>
                <option value="sphere">sphere [radius]</option>
              </select>
            </div>
            <div>
              <label htmlFor="new-obj-material" className="block text-xs text-slate-300 mb-1">
                Material Prior (configs/materials.yaml)
              </label>
              <select
                id="new-obj-material"
                value={newMaterial}
                onChange={(e) => setNewMaterial(e.target.value)}
                className="w-full bg-[#0B0E17] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-cyan-500"
              >
                {Object.keys(MATERIAL_DB).map((mat) => (
                  <option key={mat} value={mat}>
                    {mat}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="new-obj-x" className="block text-xs text-slate-300 mb-1">
                Table X (m)
              </label>
              <input
                id="new-obj-x"
                type="number"
                step="0.01"
                value={newPosX}
                onChange={(e) => setNewPosX(e.target.value)}
                className="w-full bg-[#0B0E17] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 font-mono focus:outline-none focus:border-cyan-500"
              />
            </div>
            <div>
              <label htmlFor="new-obj-y" className="block text-xs text-slate-300 mb-1">
                Table Y (m)
              </label>
              <input
                id="new-obj-y"
                type="number"
                step="0.01"
                value={newPosY}
                onChange={(e) => setNewPosY(e.target.value)}
                className="w-full bg-[#0B0E17] border border-slate-800 rounded px-3 py-2 text-xs text-slate-100 font-mono focus:outline-none focus:border-cyan-500"
              />
            </div>
          </div>

          <button
            type="submit"
            className="inline-flex items-center gap-1.5 px-4 py-2 rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs transition-colors cursor-pointer whitespace-nowrap shrink-0"
          >
            <Plus className="w-3.5 h-3.5" />
            Snap &amp; Compile Object
          </button>
        </form>
      </div>

      {/* Right column: Deterministic MJCF 3.x XML Output */}
      <div className="xl:col-span-6 p-6 flex flex-col">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="text-sm font-semibold text-slate-100">
              Stage 5: Compiled MuJoCo 3.x MJCF XML
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Implicit-fast integrator (<code className="text-slate-300">timestep=&quot;0.002&quot;</code>) · Weld mocap gripper · <code className="text-slate-300">condim=&quot;4&quot;</code>
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setRandomize((r) => !r)}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors cursor-pointer whitespace-nowrap shrink-0 ${
                randomize
                  ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                  : 'bg-[#0B0E17] border border-slate-800 text-slate-300 hover:bg-slate-800'
              }`}
            >
              <Shuffle className="w-3.5 h-3.5" />
              {randomize ? 'Domain Rand: ON' : 'Domain Rand: Midpoint'}
            </button>

            {randomize && (
              <div className="flex items-center gap-1 text-xs font-mono">
                <label htmlFor="rand-seed" className="text-slate-400">
                  Seed:
                </label>
                <input
                  id="rand-seed"
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value) || 0)}
                  className="w-16 bg-[#0B0E17] border border-slate-800 rounded px-2 py-1 text-xs text-slate-100"
                />
              </div>
            )}

            <button
              type="button"
              onClick={handleCopyXml}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#0B0E17] border border-slate-800 hover:bg-slate-800 text-slate-200 text-xs font-medium transition-colors cursor-pointer whitespace-nowrap shrink-0"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? 'Copied XML' : 'Copy MJCF'}
            </button>
          </div>
        </div>

        <pre
          data-testid="mjcf-xml-output"
          className="flex-1 bg-[#0B0E17] border border-slate-800/80 rounded p-4 text-[11px] leading-relaxed font-mono text-slate-300 overflow-x-auto max-h-[460px] overflow-y-auto"
        >
          {mjcfXml}
        </pre>
      </div>
    </div>
  );
};
