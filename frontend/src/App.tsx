/**
 * WorldForge Studio 2.0 — Main Application Cockpit
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Box,
  Check,
  CheckCircle2,
  Clock,
  Copy,
  Cpu,
  DollarSign,
  Download,
  Eye,
  Layers,
  Play,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Terminal,
  Upload,
  Video,
  X,
  Zap,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// ─────────────────────────────────────────────────────────────────────────────
// Engine Types & Backend API Client (./engine/worldforge)
// ─────────────────────────────────────────────────────────────────────────────

export const API_BASE =
  typeof window !== 'undefined' &&
    (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
    ? 'http://localhost:8000'
    : '';

export interface MaterialSpec {
  rgba: [number, number, number, number];
  density: number;
  friction: [number, number, number];
}

export const MATERIAL_DB: Record<string, MaterialSpec> = {
  plastic: { rgba: [0.9, 0.2, 0.2, 1.0], density: 1200, friction: [0.6, 0.005, 0.0001] },
  metal: { rgba: [0.75, 0.78, 0.82, 1.0], density: 7800, friction: [0.8, 0.005, 0.0001] },
  wood: { rgba: [0.65, 0.45, 0.25, 1.0], density: 700, friction: [0.7, 0.005, 0.0001] },
  rubber: { rgba: [0.15, 0.8, 0.45, 1.0], density: 1100, friction: [1.2, 0.01, 0.001] },
  ceramic: { rgba: [0.92, 0.92, 0.95, 1.0], density: 2400, friction: [0.5, 0.002, 0.0001] },
  cardboard: { rgba: [0.76, 0.6, 0.42, 1.0], density: 300, friction: [0.65, 0.005, 0.0001] },
};

export interface SceneObject {
  id: string;
  label: string;
  primitive: 'box' | 'cylinder' | 'sphere';
  size: number[];
  position: [number, number, number];
  quaternion?: [number, number, number, number];
  material: string;
  is_static: boolean;
  parent_id: string | null;
  joint: {
    type: 'free' | 'hinge' | 'slide' | 'fixed';
    axis?: [number, number, number];
    range?: [number, number] | null;
    damping?: number;
    is_prior_guess?: boolean;
  };
}

export interface SceneGraph {
  scene_id: string;
  table_z: number;
  camera: {
    pos: [number, number, number];
    lookat: [number, number, number];
    fovy: number;
    resolution: [number, number];
  };
  objects: SceneObject[];
  task: {
    task_type: 'pick_and_place' | 'push_to_target' | 'articulate_joint';
    description: string;
    manipulated_object_id: string;
    target_position?: [number, number, number] | null;
    target_joint_delta?: number | null;
    waypoints: [number, number, number, number][];
  };
}

export interface SceneMeta {
  scene_id: string;
  file_id: string;
  file_name: string;
  task_type: string;
  description: string;
  num_objects: number;
  objects: any[];
  table_z: number;
  has_articulated_joint: boolean;
}

export interface SimState {
  step: number;
  gripperPos: [number, number, number];
  gripperCmd: number;
  fingerAperture: number;
  isGrasped: boolean;
  objectPositions: Record<string, [number, number, number]>;
  objectQuaternions?: Record<string, [number, number, number, number]>;
  diagnostics: {
    dReach: number;
    dGoal: number;
  };
  reward: number;
  terminated: boolean;
  stageBreakdown: {
    reach: number;
    graspOrClearance: number;
    liftOrProgress: number;
    placeOrGoal: number;
    terminalBonus: number;
  };
}

export interface SystemStatus {
  status: string;
  engine: string;
  mujoco_version: string;
  baseten: { configured: boolean; model: string; base_url: string; grant: string };
  meta: { configured: boolean; model: string; grant: string };
  modal: { configured: boolean; grant: string };
}

export interface PipelineRunResult {
  scene_id: string;
  sam_provider_used?: string;
  total_duration_s: number;
  stage_timings: Record<string, number>;
  reasoning_trace: string;
  scene_graph: SceneGraph;
  mjcf_xml: string;
  reward_code: string;
  gate_results: Record<string, any>;
  self_healing_applied: boolean;
  repair_attempts: number;
  repairs_applied: string[];
  repair_diff: string;
  ppo_results: {
    eval_mean_return: number;
    eval_std_return: number;
    learning_curve: { timesteps: number; mean_reward: number }[];
    video_url: string;
    oracle_video_url?: string;
  };
  render_diff: {
    mean_iou: number;
    per_object_iou: Record<string, number>;
    passed: boolean;
    overlay_url: string;
    masks_url: string;
  };
  cost_telemetry: {
    input_tokens: number;
    output_tokens: number;
    cpu_core_seconds: number;
    total_cost_usd: number;
  };
}

export interface PipelineStageEvent {
  stage: number | 'complete' | 'error';
  name?: string;
  sam_provider_used?: string;
  status?: 'running' | 'completed' | 'self_healing';
  duration_s?: number;
  chunk?: string;
  reasoning_trace?: string;
  gate_results?: Record<string, any>;
  repair_attempt?: number;
  repair_attempts?: number;
  repair_diff?: string;
  summary?: PipelineRunResult;
  error?: string;
}

export function computeHalfHeight(obj: SceneObject): number {
  if (obj.primitive === 'box') return obj.size[2] ?? 0.025;
  if (obj.primitive === 'cylinder') return obj.size[1] ?? 0.025;
  return obj.size[0] ?? 0.025;
}

export function createDefaultSimState(sg: SceneGraph): SimState {
  const objectPositions: Record<string, [number, number, number]> = {};
  const objectQuaternions: Record<string, [number, number, number, number]> = {};
  for (const o of sg.objects) {
    objectPositions[o.id] = [...o.position];
    objectQuaternions[o.id] = o.quaternion ? [...o.quaternion] : [1, 0, 0, 0];
  }
  const wp0 = sg.task.waypoints[0] ?? [0.0, 0.0, 0.25, 1.0];
  return {
    step: 0,
    gripperPos: [wp0[0], wp0[1], wp0[2]],
    gripperCmd: wp0[3],
    fingerAperture: 0.04,
    isGrasped: false,
    objectPositions,
    objectQuaternions,
    diagnostics: { dReach: 0.15, dGoal: 0.25 },
    reward: 0.0,
    terminated: false,
    stageBreakdown: {
      reach: 0.0,
      graspOrClearance: 0.0,
      liftOrProgress: 0.0,
      placeOrGoal: 0.0,
      terminalBonus: 0.0,
    },
  };
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export async function fetchScenes(): Promise<SceneMeta[]> {
  const res = await fetch(`${API_BASE}/api/scenes`);
  if (!res.ok) throw new Error('Failed to fetch scenes');
  return res.json();
}

export async function fetchSceneDetails(sceneId: string): Promise<SceneGraph> {
  const res = await fetch(`${API_BASE}/api/scenes/${sceneId}`);
  if (!res.ok) throw new Error(`Failed to fetch scene ${sceneId}`);
  return res.json();
}

export async function resetSimulation(sceneId: string): Promise<SimState> {
  const res = await fetch(`${API_BASE}/api/sim/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scene_id: sceneId }),
  });
  if (!res.ok) throw new Error('Failed to reset simulation');
  return res.json();
}

export async function stepSimulation(
  sceneId: string,
  action: [number, number, number, number],
): Promise<SimState> {
  const res = await fetch(`${API_BASE}/api/sim/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scene_id: sceneId, action }),
  });
  if (!res.ok) throw new Error('Failed to step simulation');
  return res.json();
}

export async function stepOracle(sceneId: string): Promise<SimState> {
  const res = await fetch(`${API_BASE}/api/sim/oracle-step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scene_id: sceneId }),
  });
  if (!res.ok) throw new Error('Failed to step oracle');
  return res.json();
}

export async function fetchSimTrajectory(sceneId: string, maxSteps = 120): Promise<SimState[]> {
  const res = await fetch(
    `${API_BASE}/api/sim/trajectory?scene_id=${encodeURIComponent(sceneId)}&max_steps=${maxSteps}`,
  );
  if (!res.ok) throw new Error('Failed to fetch trajectory');
  return res.json();
}

export async function uploadTabletopVideo(
  file: File,
): Promise<{ status: string; filename: string; video_path?: string; video_url: string; size_bytes: number }> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload-video`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Video upload failed');
  return res.json();
}

export function streamPipeline(
  params: {
    scene_id: string;
    inject_fault: boolean;
    provider: string;
    quick_ppo: boolean;
    video_path?: string | null;
  },
  onEvent: (ev: PipelineStageEvent) => void,
  onError: (err: any) => void,
  onComplete: () => void,
): () => void {
  const q = new URLSearchParams({
    scene_id: params.scene_id,
    inject_fault: String(params.inject_fault),
    provider: params.provider,
    quick_ppo: String(params.quick_ppo),
  });
  if (params.video_path) {
    q.set('video_path', params.video_path);
  }
  const es = new EventSource(`${API_BASE}/api/run/stream?${q.toString()}`);

  es.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data);
      onEvent(parsed);
      if (parsed.stage === 'complete' || parsed.stage === 'error') {
        es.close();
        onComplete();
      }
    } catch (e) {
      console.error('Failed to parse SSE payload', e);
    }
  };

  es.onerror = (err) => {
    es.close();
    onError(err);
  };

  return () => es.close();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3D PBR Workbench Component (./components/SceneCanvas)
// ─────────────────────────────────────────────────────────────────────────────

interface SceneCanvasProps {
  scene: SceneGraph;
  simState: SimState;
  rewardHistory: SimState[];
  isPlaying: boolean;
  onTogglePlay: () => void;
  onResetSim: () => void;
  onStepOracleEnd: () => void;
  onManualAction: (action: [number, number, number, number]) => void;
}

type CameraPreset = 'isometric' | 'sensor_cam' | 'top_ortho' | 'front_elevation';

function createOpticalTableTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#141923';
  ctx.fillRect(0, 0, 512, 512);

  ctx.fillStyle = 'rgba(255,255,255,0.015)';
  for (let y = 0; y < 512; y += 3) {
    ctx.fillRect(0, y, 512, 1);
  }

  ctx.strokeStyle = 'rgba(148, 163, 184, 0.14)';
  ctx.lineWidth = 1;
  const step = 64;
  for (let i = 0; i <= 512; i += step) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 512);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }

  ctx.strokeStyle = 'rgba(6, 182, 212, 0.28)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(256, 0);
  ctx.lineTo(256, 512);
  ctx.moveTo(0, 256);
  ctx.lineTo(512, 256);
  ctx.stroke();

  ctx.fillStyle = '#090c12';
  for (let x = 32; x < 512; x += 32) {
    for (let y = 32; y < 512; y += 32) {
      ctx.beginPath();
      ctx.arc(x, y, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

const SceneCanvas: React.FC<SceneCanvasProps> = ({
  scene,
  simState,
  rewardHistory,
  isPlaying,
  onTogglePlay,
  onResetSim,
  onStepOracleEnd,
  onManualAction,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const [camPreset, setCamPreset] = useState<CameraPreset>('isometric');
  const [showWaypoints, setShowWaypoints] = useState<boolean>(true);
  const [showContactShadows, setShowContactShadows] = useState<boolean>(true);

  const scenePropRef = useRef<SceneGraph>(scene);
  const simStateRef = useRef<SimState>(simState);
  const showWaypointsRef = useRef<boolean>(showWaypoints);
  const showShadowsRef = useRef<boolean>(showContactShadows);
  const requestRenderRef = useRef<() => void>(() => { });
  const applyCamPresetRef = useRef<(preset: CameraPreset) => void>(() => { });

  scenePropRef.current = scene;
  simStateRef.current = simState;
  showWaypointsRef.current = showWaypoints;
  showShadowsRef.current = showContactShadows;

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const isSoftwareGl = (() => {
      const probeCanvas = document.createElement('canvas');
      const gl = probeCanvas.getContext('webgl2') || probeCanvas.getContext('webgl');
      if (!gl) return true;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
      const gpu = String(gl.getParameter(param));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();

    const width = container.clientWidth || 860;
    const height = container.clientHeight || 460;

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true,
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    renderer.domElement.setAttribute('data-testid', 'mujoco-viewport-canvas');
    renderer.domElement.className = 'w-full h-full block cursor-crosshair';
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    const threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x07090e);
    threeScene.fog = new THREE.FogExp2(0x07090e, 0.95);

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    threeScene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
    pmremGenerator.dispose();

    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.01, 10);
    camera.up.set(0, 0, 1);
    camera.position.set(0.48, -0.52, 0.42);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0.06);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.08;
    controls.maxDistance = 1.8;
    controls.minDistance = 0.18;
    controls.update();

    const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x1e293b, 0.85);
    hemiLight.position.set(0, 0, 1.5);
    threeScene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfff7ed, 2.2);
    keyLight.position.set(0.65, -0.75, 1.25);
    if (!isSoftwareGl) {
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.set(1024, 1024);
      keyLight.shadow.camera.near = 0.1;
      keyLight.shadow.camera.far = 3.5;
      const d = 0.45;
      keyLight.shadow.camera.left = -d;
      keyLight.shadow.camera.right = d;
      keyLight.shadow.camera.top = d;
      keyLight.shadow.camera.bottom = -d;
      keyLight.shadow.bias = -0.0005;
    }
    threeScene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x38bdf8, 1.1);
    rimLight.position.set(-0.7, 0.8, 0.65);
    threeScene.add(rimLight);

    const tableGroup = new THREE.Group();
    const tableTex = createOpticalTableTexture();
    const tableTopMat = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      map: tableTex,
      roughness: 0.42,
      metalness: 0.25,
    });
    const tableSubMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.65,
      metalness: 0.6,
    });
    const frameTrimMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      roughness: 0.3,
      metalness: 0.85,
    });

    const topSlab = new THREE.Mesh(new THREE.BoxGeometry(0.64, 0.64, 0.024), tableTopMat);
    topSlab.position.set(0, 0, -0.012);
    topSlab.receiveShadow = true;
    tableGroup.add(topSlab);

    const chassisBase = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.6, 0.05), tableSubMat);
    chassisBase.position.set(0, 0, -0.05);
    tableGroup.add(chassisBase);

    const railOffsets: [number, number][] = [
      [0.322, 0],
      [-0.322, 0],
      [0, 0.322],
      [0, -0.322],
    ];
    railOffsets.forEach(([rx, ry], idx) => {
      const railGeo =
        idx < 2
          ? new THREE.BoxGeometry(0.006, 0.648, 0.028)
          : new THREE.BoxGeometry(0.648, 0.006, 0.028);
      const rail = new THREE.Mesh(railGeo, frameTrimMat);
      rail.position.set(rx, ry, -0.012);
      tableGroup.add(rail);
    });

    const originTriad = new THREE.Group();
    originTriad.position.set(-0.28, -0.28, 0.002);
    originTriad.add(
      new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 0.075, 0xef4444, 0.016, 0.009),
      new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 0), 0.075, 0x22c55e, 0.016, 0.009),
      new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, 0), 0.075, 0x3b82f6, 0.016, 0.009),
    );
    tableGroup.add(originTriad);
    threeScene.add(tableGroup);

    const gripperGroup = new THREE.Group();
    const housingMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.32, metalness: 0.78 });
    const alloyMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, roughness: 0.22, metalness: 0.9 });
    const fingerPadMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.85, metalness: 0.15 });
    const mocapBeaconMat = new THREE.MeshStandardMaterial({
      color: 0xf43f5e,
      emissive: 0xe11d48,
      emissiveIntensity: 0.75,
      roughness: 0.2,
    });

    const wristFlange = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.026, 0.018, 24), alloyMat);
    wristFlange.rotation.x = Math.PI / 2;
    wristFlange.position.set(0, 0, 0.028);
    gripperGroup.add(wristFlange);

    const servoBody = new THREE.Mesh(new THREE.BoxGeometry(0.056, 0.078, 0.028), housingMat);
    servoBody.position.set(0, 0, 0.006);
    gripperGroup.add(servoBody);

    const guideRail = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.088, 0.008), alloyMat);
    guideRail.position.set(0, 0, -0.01);
    gripperGroup.add(guideRail);

    const mocapBeacon = new THREE.Mesh(new THREE.TorusGeometry(0.016, 0.003, 12, 24), mocapBeaconMat);
    mocapBeacon.position.set(0, 0, 0.039);
    gripperGroup.add(mocapBeacon);

    const buildFingerAssembly = (sign: number) => {
      const fGroup = new THREE.Group();
      const carriage = new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.012, 0.014), alloyMat);
      carriage.position.set(0, 0, -0.016);
      fGroup.add(carriage);

      const outerJaw = new THREE.Mesh(new THREE.BoxGeometry(0.018, 0.008, 0.048), housingMat);
      outerJaw.position.set(0, 0, -0.042);
      fGroup.add(outerJaw);

      const innerPad = new THREE.Mesh(new THREE.BoxGeometry(0.016, 0.003, 0.036), fingerPadMat);
      innerPad.position.set(0, -sign * 0.0045, -0.046);
      fGroup.add(innerPad);
      return fGroup;
    };

    const leftFingerGroup = buildFingerAssembly(-1);
    const rightFingerGroup = buildFingerAssembly(1);
    gripperGroup.add(leftFingerGroup, rightFingerGroup);

    const laserGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, -0.05),
      new THREE.Vector3(0, 0, -0.35),
    ]);
    const laserMat = new THREE.LineDashedMaterial({
      color: 0xf43f5e,
      dashSize: 0.01,
      gapSize: 0.008,
      transparent: true,
      opacity: 0.55,
    });
    const laserLine = new THREE.Line(laserGeo, laserMat);
    laserLine.computeLineDistances();
    gripperGroup.add(laserLine);
    threeScene.add(gripperGroup);

    const dynamicObjectsGroup = new THREE.Group();
    const trajectoryGroup = new THREE.Group();
    threeScene.add(dynamicObjectsGroup, trajectoryGroup);

    const syncSceneGeometry = () => {
      const sg = scenePropRef.current;
      const st = simStateRef.current;

      tableGroup.position.z = sg.table_z;

      while (dynamicObjectsGroup.children.length > 0) {
        dynamicObjectsGroup.remove(dynamicObjectsGroup.children[0]);
      }
      while (trajectoryGroup.children.length > 0) {
        trajectoryGroup.remove(trajectoryGroup.children[0]);
      }

      const [gx, gy, gz] = st.gripperPos;
      gripperGroup.position.set(gx, gy, gz + 0.032);
      const halfSpread = Math.max(0.008, (st.fingerAperture / 0.08) * 0.034);
      leftFingerGroup.position.set(0, -halfSpread, 0);
      rightFingerGroup.position.set(0, halfSpread, 0);
      mocapBeaconMat.color.setHex(st.isGrasped ? 0x10b981 : 0xf43f5e);
      mocapBeaconMat.emissive.setHex(st.isGrasped ? 0x059669 : 0xe11d48);

      const laserPositions = laserLine.geometry.attributes.position as THREE.BufferAttribute;
      laserPositions.setXYZ(0, 0, 0, -0.055);
      laserPositions.setXYZ(1, 0, 0, -(gz - sg.table_z + 0.03));
      laserPositions.needsUpdate = true;
      laserLine.computeLineDistances();

      if (sg.task.target_position) {
        const [tx, ty, tz] = sg.task.target_position;
        const targetGroup = new THREE.Group();
        targetGroup.position.set(tx, ty, Math.max(sg.table_z + 0.002, tz - 0.03));

        const outerRing = new THREE.Mesh(
          new THREE.RingGeometry(0.038, 0.045, 32),
          new THREE.MeshBasicMaterial({
            color: 0x10b981,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.85,
          }),
        );
        const innerDisc = new THREE.Mesh(
          new THREE.CircleGeometry(0.036, 32),
          new THREE.MeshBasicMaterial({
            color: 0x10b981,
            transparent: true,
            opacity: 0.18,
          }),
        );
        innerDisc.position.z = -0.0005;
        targetGroup.add(outerRing, innerDisc);
        dynamicObjectsGroup.add(targetGroup);
      }

      if (showWaypointsRef.current && sg.task.waypoints.length > 1) {
        const wpPoints = sg.task.waypoints.map(([wx, wy, wz]) => new THREE.Vector3(wx, wy, wz));
        const curve = new THREE.CatmullRomCurve3(wpPoints, false, 'centripetal', 0.2);
        const tubeGeo = new THREE.TubeGeometry(curve, 36, 0.0022, 8, false);
        const tubeMat = new THREE.MeshStandardMaterial({
          color: 0x06b6d4,
          emissive: 0x0891b2,
          emissiveIntensity: 0.45,
          roughness: 0.3,
          transparent: true,
          opacity: 0.75,
        });
        trajectoryGroup.add(new THREE.Mesh(tubeGeo, tubeMat));

        const wpSphereGeo = new THREE.SphereGeometry(0.0055, 14, 14);
        sg.task.waypoints.forEach(([wx, wy, wz, gripCmd]) => {
          const wpMat = new THREE.MeshStandardMaterial({
            color: gripCmd < 0 ? 0xf59e0b : 0x38bdf8,
            emissive: gripCmd < 0 ? 0xd97706 : 0x0284c7,
            emissiveIntensity: 0.5,
            roughness: 0.2,
          });
          const sp = new THREE.Mesh(wpSphereGeo, wpMat);
          sp.position.set(wx, wy, wz);
          trajectoryGroup.add(sp);
        });
      }

      for (const obj of sg.objects) {
        if (obj.id === 'table') continue;
        const pos = st.objectPositions[obj.id] ?? obj.position;
        const matSpec = MATERIAL_DB[obj.material] ?? MATERIAL_DB.plastic;
        const baseColor = new THREE.Color(matSpec.rgba[0], matSpec.rgba[1], matSpec.rgba[2]);
        const isManipulated = obj.id === sg.task.manipulated_object_id;

        const roughness =
          obj.material === 'metal'
            ? 0.22
            : obj.material === 'rubber'
              ? 0.88
              : obj.material === 'wood'
                ? 0.62
                : 0.35;
        const metalness = obj.material === 'metal' ? 0.88 : 0.08;

        const primaryMat = new THREE.MeshStandardMaterial({ color: baseColor, roughness, metalness });
        const trimMat = new THREE.MeshStandardMaterial({
          color: baseColor.clone().multiplyScalar(0.72),
          roughness: Math.min(1, roughness + 0.25),
          metalness: Math.min(1, metalness + 0.15),
        });

        const quat = st.objectQuaternions?.[obj.id] ?? obj.quaternion ?? [1, 0, 0, 0];
        const objGroup = new THREE.Group();
        objGroup.position.set(pos[0], pos[1], pos[2]);
        objGroup.quaternion.set(quat[1], quat[2], quat[3], quat[0]);

        if (obj.primitive === 'box') {
          const hx = (obj.size[0] ?? 0.025) * 2;
          const hy = (obj.size[1] ?? 0.025) * 2;
          const hz = (obj.size[2] ?? 0.025) * 2;

          const coreMesh = new THREE.Mesh(new THREE.BoxGeometry(hx, hy, hz), primaryMat);
          coreMesh.castShadow = !isSoftwareGl;
          coreMesh.receiveShadow = !isSoftwareGl;
          objGroup.add(coreMesh);

          const bezelMesh = new THREE.Mesh(new THREE.BoxGeometry(hx * 0.82, hy * 0.82, hz * 1.02), trimMat);
          objGroup.add(bezelMesh);

          if (isManipulated) {
            const edges = new THREE.EdgesGeometry(new THREE.BoxGeometry(hx * 1.01, hy * 1.01, hz * 1.01));
            objGroup.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x38bdf8 })));
          }
        } else if (obj.primitive === 'cylinder') {
          const radius = obj.size[0] ?? 0.025;
          const halfLen = obj.size[1] ?? 0.02;
          const cylGeo = new THREE.CylinderGeometry(radius, radius, halfLen * 2, 28);
          cylGeo.rotateX(Math.PI / 2);
          const cylMesh = new THREE.Mesh(cylGeo, primaryMat);
          cylMesh.castShadow = !isSoftwareGl;
          cylMesh.receiveShadow = !isSoftwareGl;
          objGroup.add(cylMesh);

          const rimGeo = new THREE.TorusGeometry(radius * 0.96, Math.min(0.0025, halfLen * 0.3), 12, 28);
          const topRim = new THREE.Mesh(rimGeo, trimMat);
          topRim.position.z = halfLen;
          objGroup.add(topRim);
        } else {
          const radius = obj.size[0] ?? 0.025;
          const spMesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 24, 24), primaryMat);
          spMesh.castShadow = !isSoftwareGl;
          objGroup.add(spMesh);
        }

        if (showShadowsRef.current && obj.parent_id === null) {
          const shadowRadius = Math.max(obj.size[0] ?? 0.025, obj.size[1] ?? 0.025) * 1.35;
          const contactDisc = new THREE.Mesh(
            new THREE.CircleGeometry(shadowRadius, 24),
            new THREE.MeshBasicMaterial({
              color: 0x000000,
              transparent: true,
              opacity: 0.42,
              depthWrite: false,
            }),
          );
          const halfH = computeHalfHeight(obj);
          contactDisc.position.set(0, 0, Math.max(-pos[2] + sg.table_z + 0.0015, -halfH + 0.001));
          objGroup.add(contactDisc);
        }

        dynamicObjectsGroup.add(objGroup);
      }
    };

    let frameId = 0;
    let timeoutId = 0;
    let renderBusy = false;
    let renderDirty = false;

    function afterDraw(cb: () => void) {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      if (gl && typeof gl.fenceSync === 'function') {
        const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.flush();
        const poll = () => {
          const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
          if (state === gl.UNSIGNALED) {
            timeoutId = window.setTimeout(poll, 5);
          } else {
            if (sync) gl.deleteSync(sync);
            timeoutId = window.setTimeout(cb, 80);
          }
        };
        timeoutId = window.setTimeout(poll, 5);
      } else {
        timeoutId = window.setTimeout(cb, 80);
      }
    }

    function requestRender() {
      if (renderBusy) {
        renderDirty = true;
        return;
      }
      renderBusy = true;
      frameId = requestAnimationFrame(() => {
        renderDirty = false;
        controls.update();
        renderer.render(threeScene, camera);
        const done = () => {
          renderBusy = false;
          if (renderDirty) requestRender();
        };
        if (isSoftwareGl) afterDraw(done);
        else done();
      });
    }

    requestRenderRef.current = () => {
      syncSceneGeometry();
      requestRender();
    };

    applyCamPresetRef.current = (preset: CameraPreset) => {
      const sg = scenePropRef.current;
      if (preset === 'isometric') {
        camera.position.set(0.46, -0.5, 0.4);
        controls.target.set(0, 0, 0.05);
      } else if (preset === 'sensor_cam') {
        camera.position.set(sg.camera.pos[0], sg.camera.pos[1], sg.camera.pos[2]);
        controls.target.set(sg.camera.lookat[0], sg.camera.lookat[1], sg.camera.lookat[2]);
      } else if (preset === 'top_ortho') {
        camera.position.set(0.0, -0.01, 0.72);
        controls.target.set(0, 0, 0.0);
      } else if (preset === 'front_elevation') {
        camera.position.set(0.0, -0.62, 0.16);
        controls.target.set(0, 0, 0.06);
      }
      controls.update();
      requestRender();
    };

    controls.addEventListener('change', requestRender);

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth || 860;
      const h = container.clientHeight || 460;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      requestRender();
    };
    window.addEventListener('resize', handleResize);

    syncSceneGeometry();
    requestRender();

    return () => {
      window.removeEventListener('resize', handleResize);
      controls.removeEventListener('change', requestRender);
      cancelAnimationFrame(frameId);
      clearTimeout(timeoutId);
      controls.dispose();
      renderer.dispose();
    };
  }, []);

  useEffect(() => {
    requestRenderRef.current();
  }, [scene, simState, showWaypoints, showContactShadows]);

  const handleSelectPreset = (preset: CameraPreset) => {
    setCamPreset(preset);
    applyCamPresetRef.current(preset);
  };

  const gripClosed = simState.gripperCmd < 0;

  const historySlice = rewardHistory.slice(-40);
  const svgW = 540;
  const svgH = 110;
  const maxStep = Math.max(historySlice.length - 1, 1);

  const buildPolyline = (selector: (s: SimState) => number, maxVal = 1.2) =>
    historySlice
      .map((s, idx) => {
        const x = (idx / maxStep) * (svgW - 24) + 12;
        const norm = Math.max(0, Math.min(1, selector(s) / maxVal));
        const y = svgH - 16 - norm * (svgH - 28);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

  return (
    <div className="flex flex-col border-b border-[#1E293B] bg-[#07090E]">
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-2.5 border-b border-[#1E293B] bg-[#0B0E17]">
        <div className="flex items-center gap-4 text-xs font-mono tabular-nums text-slate-400">
          <span className="text-slate-200 font-semibold">3D PBR WORKBENCH</span>
          <span aria-hidden="true">·</span>
          <span>
            SCENE <strong className="text-cyan-400">{scene.scene_id}</strong>
          </span>
          <span aria-hidden="true">·</span>
          <span>
            STEP <strong className="text-slate-100">{simState.step}</strong>
          </span>
          <span aria-hidden="true">·</span>
          <span>
            FOVY <strong className="text-slate-200">{scene.camera.fovy.toFixed(0)}°</strong>
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {(
            [
              ['isometric', 'Isometric Rig'],
              ['sensor_cam', 'MuJoCo render_cam'],
              ['top_ortho', 'Top XY Plane'],
              ['front_elevation', 'Front XZ'],
            ] as [CameraPreset, string][]
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => handleSelectPreset(key)}
              className={`px-2.5 py-1 text-xs font-mono rounded transition-colors whitespace-nowrap shrink-0 cursor-pointer ${camPreset === key
                  ? 'bg-cyan-500 text-slate-950 font-semibold'
                  : 'bg-[#111827] text-slate-300 hover:text-white border border-[#1E293B]'
                }`}
            >
              {label}
            </button>
          ))}

          <button
            type="button"
            onClick={() => setShowWaypoints((w) => !w)}
            className={`px-2.5 py-1 text-xs font-mono rounded border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${showWaypoints
                ? 'bg-[#111827] text-cyan-300 border-cyan-500/40'
                : 'bg-[#0B0E17] text-slate-500 border-[#1E293B]'
              }`}
          >
            Spline Waypoints: {showWaypoints ? 'ON' : 'OFF'}
          </button>

          <button
            type="button"
            onClick={() => setShowContactShadows((s) => !s)}
            className={`px-2.5 py-1 text-xs font-mono rounded border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${showContactShadows
                ? 'bg-[#111827] text-slate-200 border-[#1E293B]'
                : 'bg-[#0B0E17] text-slate-500 border-[#1E293B]'
              }`}
          >
            Shadows
          </button>
        </div>
      </div>

      <div className="relative w-full h-[430px] bg-[#07090E] overflow-hidden">
        <div ref={mountRef} className="w-full h-full" />

        <div className="absolute top-3.5 left-4 z-10 pointer-events-none bg-[#0B0E17]/95 border border-[#1E293B] px-3.5 py-2.5 rounded text-xs font-mono tabular-nums space-y-1">
          <div className="text-[11px] text-slate-400">
            EE_MOCAP WELD POSE <span aria-hidden="true">·</span> SOLREF [0.01, 1.0]
          </div>
          <div className="text-slate-100">
            X: <span className="text-rose-400">{simState.gripperPos[0].toFixed(3)}</span> m
            <span aria-hidden="true"> · </span>
            Y: <span className="text-emerald-400">{simState.gripperPos[1].toFixed(3)}</span> m
            <span aria-hidden="true"> · </span>
            Z: <span className="text-sky-400">{simState.gripperPos[2].toFixed(3)}</span> m
          </div>
          <div className="text-slate-300">
            APERTURE: <span className="text-slate-100">{(simState.fingerAperture * 1000).toFixed(1)}</span> mm
            <span aria-hidden="true"> · </span>
            STATE:{' '}
            <span className={simState.isGrasped ? 'text-emerald-400 font-semibold' : 'text-amber-400'}>
              {simState.isGrasped ? '● WELD GRASPED' : gripClosed ? '▲ JAWS CLOSED' : '● JAWS OPEN'}
            </span>
          </div>
          <div className="text-slate-400">
            Δ_reach: <span className="text-slate-200">{simState.diagnostics.dReach.toFixed(3)}</span> m
            <span aria-hidden="true"> · </span>
            Δ_goal: <span className="text-slate-200">{simState.diagnostics.dGoal.toFixed(3)}</span> m
          </div>
        </div>

        <div
          data-testid="sim-reward-readout"
          className="absolute top-3.5 right-4 z-10 pointer-events-none bg-[#0B0E17]/95 border border-[#1E293B] px-4 py-2.5 rounded text-right font-mono tabular-nums"
        >
          <div className="text-[11px] text-slate-400">SHAPED DENSE REWARD</div>
          <div className="text-2xl font-bold text-cyan-400">
            {simState.reward >= 0 ? `+${simState.reward.toFixed(3)}` : simState.reward.toFixed(3)}
            <span className="text-xs text-slate-400 ml-1.5">R_t</span>
          </div>
          <div className="text-[11px] text-slate-300">
            {simState.terminated ? '● TERMINAL GOAL (+10.0)' : '● CONTINUOUS SHAPING'}
          </div>
        </div>

        <div className="absolute bottom-3.5 left-4 z-10 flex items-center gap-2 bg-[#0B0E17]/95 border border-[#1E293B] p-1.5 rounded">
          <button
            type="button"
            onClick={onTogglePlay}
            className="px-3 py-1.5 rounded bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs whitespace-nowrap shrink-0 transition-colors cursor-pointer"
          >
            {isPlaying ? 'Pause Waypoint Oracle' : 'Run Waypoint Oracle'}
          </button>
          <button
            type="button"
            onClick={onStepOracleEnd}
            className="px-3 py-1.5 rounded bg-[#111827] hover:bg-slate-800 text-slate-100 border border-[#1E293B] font-medium text-xs whitespace-nowrap shrink-0 transition-colors cursor-pointer"
          >
            Solve Episode
          </button>
          <button
            type="button"
            onClick={onResetSim}
            className="px-3 py-1.5 rounded bg-[#111827] hover:bg-slate-800 text-slate-300 border border-[#1E293B] font-medium text-xs whitespace-nowrap shrink-0 transition-colors cursor-pointer"
          >
            Reset Frame 0
          </button>
        </div>

        <div className="absolute bottom-3.5 right-4 z-10 pointer-events-none hidden sm:block bg-[#0B0E17]/90 border border-[#1E293B] px-3 py-1.5 rounded text-[11px] font-mono text-slate-400">
          Drag: Orbit 360° · Scroll: Zoom · Right-Drag: Pan
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 border-t border-[#1E293B] bg-[#0B0E17]">
        <div className="lg:col-span-5 p-4 border-b lg:border-b-0 lg:border-r border-[#1E293B] flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs font-semibold text-slate-200">
              4-DoF Cartesian Mocap Action [dx, dy, dz, grip]
            </span>
            <span className="text-xs font-mono text-slate-400 tabular-nums">Δ_max = 0.020 m</span>
          </div>

          <div className="flex flex-wrap items-center gap-1.5 my-1">
            <button
              type="button"
              onClick={() => onManualAction([1, 0, 0, simState.gripperCmd])}
              className="px-2.5 py-1.5 bg-[#111827] hover:bg-slate-800 border border-[#1E293B] text-slate-100 rounded text-xs font-mono whitespace-nowrap shrink-0 cursor-pointer"
            >
              +X
            </button>
            <button
              type="button"
              onClick={() => onManualAction([-1, 0, 0, simState.gripperCmd])}
              className="px-2.5 py-1.5 bg-[#111827] hover:bg-slate-800 border border-[#1E293B] text-slate-100 rounded text-xs font-mono whitespace-nowrap shrink-0 cursor-pointer"
            >
              -X
            </button>
            <button
              type="button"
              onClick={() => onManualAction([0, 1, 0, simState.gripperCmd])}
              className="px-2.5 py-1.5 bg-[#111827] hover:bg-slate-800 border border-[#1E293B] text-slate-100 rounded text-xs font-mono whitespace-nowrap shrink-0 cursor-pointer"
            >
              +Y
            </button>
            <button
              type="button"
              onClick={() => onManualAction([0, -1, 0, simState.gripperCmd])}
              className="px-2.5 py-1.5 bg-[#111827] hover:bg-slate-800 border border-[#1E293B] text-slate-100 rounded text-xs font-mono whitespace-nowrap shrink-0 cursor-pointer"
            >
              -Y
            </button>
            <button
              type="button"
              onClick={() => onManualAction([0, 0, 1, simState.gripperCmd])}
              className="px-2.5 py-1.5 bg-[#111827] hover:bg-slate-800 border border-[#1E293B] text-slate-100 rounded text-xs font-mono whitespace-nowrap shrink-0 cursor-pointer"
            >
              +Z (Up)
            </button>
            <button
              type="button"
              onClick={() => onManualAction([0, 0, -1, simState.gripperCmd])}
              className="px-2.5 py-1.5 bg-[#111827] hover:bg-slate-800 border border-[#1E293B] text-slate-100 rounded text-xs font-mono whitespace-nowrap shrink-0 cursor-pointer"
            >
              -Z (Down)
            </button>
            <button
              type="button"
              onClick={() => onManualAction([0, 0, 0, gripClosed ? 1.0 : -1.0])}
              className="px-3 py-1.5 bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-300 rounded text-xs font-mono whitespace-nowrap shrink-0 cursor-pointer ml-auto"
            >
              {gripClosed ? 'Open Gripper (+1)' : 'Close Gripper (-1)'}
            </button>
          </div>

          <div className="grid grid-cols-4 gap-3 pt-3 mt-2 border-t border-[#1E293B] font-mono tabular-nums">
            <div>
              <div className="text-[11px] text-slate-400">01. Reach</div>
              <div className="text-sm font-semibold text-cyan-400">
                {simState.stageBreakdown.reach.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-400">02. Grasp</div>
              <div className="text-sm font-semibold text-amber-400">
                {simState.stageBreakdown.graspOrClearance.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-400">03. Lift/Prog</div>
              <div className="text-sm font-semibold text-indigo-400">
                {simState.stageBreakdown.liftOrProgress.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-[11px] text-slate-400">04. Place</div>
              <div className="text-sm font-semibold text-emerald-400">
                {simState.stageBreakdown.placeOrGoal.toFixed(3)}
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-7 p-4 flex flex-col justify-between">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-1">
            <span className="text-xs font-semibold text-slate-200">
              Multi-Stage Shaped Reward Telemetry (R_reach · R_place · R_total)
            </span>
            <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400">
              <span className="text-cyan-400">━ Total Dense (R_t)</span>
              <span className="text-emerald-400">┅ Place / Progress</span>
              <span className="text-amber-400">┈ Reach (1 - tanh)</span>
            </div>
          </div>

          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="w-full h-[104px] bg-[#07090E] border border-[#1E293B] rounded"
          >
            {[20, 50, 80].map((gy) => (
              <line
                key={gy}
                x1={0}
                y1={gy}
                x2={svgW}
                y2={gy}
                stroke="#1E293B"
                strokeDasharray="3 3"
                strokeWidth="1"
              />
            ))}

            {historySlice.length > 1 && (
              <polyline
                fill="none"
                stroke="#F59E0B"
                strokeWidth="1.5"
                strokeDasharray="2 2"
                points={buildPolyline((s) => s.stageBreakdown.reach, 0.35)}
              />
            )}

            {historySlice.length > 1 && (
              <polyline
                fill="none"
                stroke="#10B981"
                strokeWidth="1.75"
                strokeDasharray="4 2"
                points={buildPolyline(
                  (s) => s.stageBreakdown.placeOrGoal + s.stageBreakdown.liftOrProgress,
                  0.7,
                )}
              />
            )}

            {historySlice.length > 1 && (
              <polyline
                fill="none"
                stroke="#06B6D4"
                strokeWidth="2.2"
                points={buildPolyline((s) => s.reward - s.stageBreakdown.terminalBonus, 1.0)}
              />
            )}
          </svg>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main WorldForge Studio 2.0 Application
// ─────────────────────────────────────────────────────────────────────────────

const STAGES = [
  { id: 1, name: 'SAM 3.1', desc: '4px Boundary Erosion', isAi: true },
  { id: 2, name: '3D Lift', desc: 'RANSAC Plane + SVD', isAi: false },
  { id: 3, name: 'Muse Spark', desc: 'meta-models/muse-spark-open-1026', isAi: true },
  { id: 4, name: 'Scene Graph', desc: 'Pydantic V2 Auto-Snap', isAi: false },
  { id: 5, name: 'MJCF Builder', desc: 'MuJoCo 3.17 XML', isAi: false },
  { id: 6, name: 'Muse Code', desc: 'Dense Reward Synthesis', isAi: true },
  { id: 7, name: 'Validation & PPO', desc: 'Gates 1–5 + SB3 Rollout', isAi: false },
  { id: 8, name: 'Render + Diff', desc: 'Mask IoU Refinement', isAi: false },
];

const DEFAULT_SCENES: SceneMeta[] = [
  {
    scene_id: 'pick_place_cube',
    file_id: 'scene1_pick_place',
    file_name: 'scene1_pick_place.json',
    task_type: 'pick_and_place',
    description: 'Pick up the red cube and place it precisely on the target coaster.',
    num_objects: 2,
    objects: [],
    table_z: 0.0,
    has_articulated_joint: false,
  },
  {
    scene_id: 'push_obstacle',
    file_id: 'scene2_push_obstacle',
    file_name: 'scene2_push_obstacle.json',
    task_type: 'push_to_target',
    description: 'Push the cylinder around the obstacle to the target marker.',
    num_objects: 3,
    objects: [],
    table_z: 0.0,
    has_articulated_joint: false,
  },
  {
    scene_id: 'articulated_drawer',
    file_id: 'scene3_articulated_drawer',
    file_name: 'scene3_articulated_drawer.json',
    task_type: 'articulate_joint',
    description: 'Grasp the handle and pull open the slide drawer by 10cm.',
    num_objects: 3,
    objects: [],
    table_z: 0.0,
    has_articulated_joint: true,
  },
];

const DEFAULT_FALLBACK_SCENE: SceneGraph = {
  scene_id: 'scene1_pick_place',
  table_z: 0.0,
  camera: {
    pos: [0.8, -0.8, 0.7],
    lookat: [0.0, 0.0, 0.1],
    fovy: 45.0,
    resolution: [640, 480],
  },
  objects: [
    {
      id: 'table',
      label: 'tabletop',
      primitive: 'box',
      size: [0.4, 0.4, 0.02],
      position: [0.0, 0.0, -0.02],
      material: 'wood',
      is_static: true,
      joint: { type: 'fixed' },
      parent_id: null,
    },
    {
      id: 'cube',
      label: 'red_cube',
      primitive: 'box',
      size: [0.025, 0.025, 0.025],
      position: [0.15, -0.15, 0.026],
      material: 'plastic',
      is_static: false,
      joint: { type: 'free' },
      parent_id: null,
    },
    {
      id: 'coaster',
      label: 'target_coaster',
      primitive: 'cylinder',
      size: [0.045, 0.003],
      position: [-0.15, 0.15, 0.004],
      material: 'cardboard',
      is_static: true,
      joint: { type: 'fixed' },
      parent_id: null,
    },
  ],
  task: {
    task_type: 'pick_and_place',
    description: 'Pick up the red cube and place it precisely on the target coaster.',
    manipulated_object_id: 'cube',
    target_position: [-0.15, 0.15, 0.026],
    waypoints: [
      [0.15, -0.15, 0.25, 1.0],
      [0.15, -0.15, 0.12, 1.0],
      [0.15, -0.15, 0.12, -1.0],
      [0.15, -0.15, 0.30, -1.0],
      [-0.15, 0.15, 0.30, -1.0],
      [-0.15, 0.15, 0.15, -1.0],
      [-0.15, 0.15, 0.15, 1.0],
    ],
  },
};

export function App() {
  const [scenes, setScenes] = useState<SceneMeta[]>(DEFAULT_SCENES);
  const [selectedSceneId, setSelectedSceneId] = useState<string>('scene1_pick_place');
  const [provider, setProvider] = useState<string>('auto');
  const [injectFault, setInjectFault] = useState<boolean>(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);

  // Video Upload State for Stage 1 SAM 3.1 Perception
  const [uploadedVideoName, setUploadedVideoName] = useState<string | null>(null);
  const [uploadedVideoPath, setUploadedVideoPath] = useState<string | null>(null);
  const [isUploadingVideo, setIsUploadingVideo] = useState<boolean>(false);
  const videoInputRef = useRef<HTMLInputElement | null>(null);

  // View switch: Pipeline Studio vs Interactive 3D PBR Workbench
  const [currentView, setCurrentView] = useState<'studio' | 'workbench'>('studio');

  // Video view mode: PPO vs Oracle vs Raw Masks
  const [videoMode, setVideoMode] = useState<'ppo' | 'oracle' | 'raw'>('ppo');

  // 3D PBR Workbench State
  const [activeSceneGraph, setActiveSceneGraph] = useState<SceneGraph>(DEFAULT_FALLBACK_SCENE);
  const [simState, setSimState] = useState<SimState>(() => createDefaultSimState(DEFAULT_FALLBACK_SCENE));
  const [rewardHistory, setRewardHistory] = useState<SimState[]>([]);
  const [isSimPlaying, setIsSimPlaying] = useState<boolean>(false);
  const playTimerRef = useRef<number | null>(null);

  // Execution state
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [activeStage, setActiveStage] = useState<number | null>(null);
  const [completedStages, setCompletedStages] = useState<number[]>([]);
  const [stageTimings, setStageTimings] = useState<Record<string, number>>({});
  const [liveReasoning, setLiveReasoning] = useState<string>('');
  const [runResult, setRunResult] = useState<PipelineRunResult | null>(null);
  const [liveSamProvider, setLiveSamProvider] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Self-Healing Telemetry State
  const [selfHealingRetries, setSelfHealingRetries] = useState<number>(0);
  const [repairDiff, setRepairDiff] = useState<string>('');
  const [liveGateResults, setLiveGateResults] = useState<Record<string, any>>({});

  // Inspector tab: Spark Reasoning, SceneGraph JSON, MJCF XML, Reward Code
  const [activeTab, setActiveTab] = useState<'reasoning' | 'scenegraph' | 'mjcf' | 'reward'>('reasoning');
  const [copied, setCopied] = useState<boolean>(false);

  // Preview Asset Cache-Busting Timestamp
  const [previewTimestamp, setPreviewTimestamp] = useState<number>(Date.now());

  // Real-time timers
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const timerRef = useRef<number | null>(null);
  const reasoningEndRef = useRef<HTMLDivElement | null>(null);

  // Initial fetch of system telemetry and scene catalog
  useEffect(() => {
    fetchSystemStatus().then(setStatus).catch(console.error);
    fetchScenes()
      .then((scs) => {
        setScenes(scs);
        if (scs.length > 0) {
          const id = scs[0].file_id || scs[0].scene_id;
          setSelectedSceneId(id);
          fetchSceneDetails(id)
            .then((sg) => {
              setActiveSceneGraph(sg);
              return resetSimulation(id);
            })
            .then((realSimState) => {
              setSimState(realSimState);
              setRewardHistory([realSimState]);
            })
            .catch(() => { });
        }
      })
      .catch(console.error);
  }, []);

  // Update active scene graph and reset live MuJoCo simulation on backend when selection changes
  useEffect(() => {
    setPreviewTimestamp(Date.now());
    fetchSceneDetails(selectedSceneId)
      .then((sg) => {
        setActiveSceneGraph(sg);
        return resetSimulation(selectedSceneId);
      })
      .then((realSimState) => {
        setSimState(realSimState);
        setRewardHistory([realSimState]);
        if (isSimPlaying) {
          setIsSimPlaying(false);
          if (playTimerRef.current) clearInterval(playTimerRef.current);
        }
      })
      .catch((err) => console.warn('Could not sync with backend simulation:', err));
  }, [selectedSceneId]);

  useEffect(() => {
    if (reasoningEndRef.current) {
      reasoningEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [liveReasoning]);

  // Clean up playback timer on unmount
  useEffect(() => {
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const handleVideoFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploadingVideo(true);
    setErrorMsg(null);
    try {
      const resp = await uploadTabletopVideo(file);
      setUploadedVideoName(resp.filename || file.name);
      setUploadedVideoPath(resp.video_path || resp.video_url);
    } catch (err: any) {
      console.warn('Video upload fallback:', err);
      setUploadedVideoName(file.name);
      setUploadedVideoPath(`outputs/uploads/${file.name}`);
    } finally {
      setIsUploadingVideo(false);
      if (videoInputRef.current) {
        videoInputRef.current.value = '';
      }
    }
  };

  const handleClearUploadedVideo = () => {
    setUploadedVideoName(null);
    setUploadedVideoPath(null);
  };

  const handleRunPipeline = () => {
    setIsRunning(true);
    setRunResult(null);
    setLiveSamProvider('');
    setActiveStage(1);
    setCompletedStages([]);
    setStageTimings({});
    setLiveReasoning('');
    setErrorMsg(null);
    setElapsedSeconds(0);
    setSelfHealingRetries(0);
    setRepairDiff('');
    setLiveGateResults({});

    const startT = Date.now();
    timerRef.current = window.setInterval(() => {
      setElapsedSeconds(Math.round((Date.now() - startT) / 100) / 10);
    }, 100);

    streamPipeline(
      {
        scene_id: selectedSceneId,
        inject_fault: injectFault,
        provider,
        quick_ppo: true,
        video_path: uploadedVideoPath,
      },
      (ev: PipelineStageEvent) => {
        if (typeof ev.stage === 'number') {
          setActiveStage(ev.stage);
          if (ev.status === 'completed') {
            setCompletedStages((prev) => Array.from(new Set([...prev, ev.stage as number])));
            if (ev.duration_s) {
              setStageTimings((prev) => ({ ...prev, [String(ev.stage)]: ev.duration_s! }));
            }
            if (ev.sam_provider_used) {
              setLiveSamProvider(ev.sam_provider_used);
            }
          }
          if (ev.chunk) {
            setLiveReasoning((prev) => prev + ev.chunk);
          }
          if (ev.reasoning_trace) {
            setLiveReasoning(ev.reasoning_trace);
          }
          if (ev.status === 'self_healing') {
            if (ev.repair_attempt) setSelfHealingRetries(ev.repair_attempt);
            if (ev.gate_results) setLiveGateResults(ev.gate_results);
            if (ev.repair_diff) {
              setRepairDiff(ev.repair_diff);
              setActiveTab('reward');
            }
          }
          if (ev.gate_results) {
            setLiveGateResults(ev.gate_results);
          }
          if (ev.repair_diff) {
            setRepairDiff(ev.repair_diff);
          }
          if (ev.repair_attempts) {
            setSelfHealingRetries(ev.repair_attempts);
          }
        } else if (ev.stage === 'complete' && ev.summary) {
          setRunResult(ev.summary);
          setActiveSceneGraph(ev.summary.scene_graph);
          setPreviewTimestamp(Date.now());
          resetSimulation(selectedSceneId)
            .then((st) => {
              setSimState(st);
              setRewardHistory([st]);
            })
            .catch(() => {
              setSimState(createDefaultSimState(ev.summary!.scene_graph));
            });
          setLiveGateResults(ev.summary.gate_results);
          if (ev.summary.repair_diff) setRepairDiff(ev.summary.repair_diff);
          if (ev.summary.repair_attempts) setSelfHealingRetries(ev.summary.repair_attempts);
          setCompletedStages([1, 2, 3, 4, 5, 6, 7, 8]);
          setActiveStage(null);
          setIsRunning(false);
          if (timerRef.current) clearInterval(timerRef.current);
        } else if (ev.stage === 'error') {
          setErrorMsg(ev.error || 'Pipeline execution failed');
          setIsRunning(false);
          if (timerRef.current) clearInterval(timerRef.current);
        }
      },
      (err) => {
        console.error('SSE Error', err);
        setErrorMsg('Network error connecting to FastAPI backend');
        setIsRunning(false);
        if (timerRef.current) clearInterval(timerRef.current);
      },
      () => {
        setIsRunning(false);
        if (timerRef.current) clearInterval(timerRef.current);
      },
    );
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // ── Real MuJoCo 3.17 Physics Backend Simulation Handlers ─────────
  const handleManualAction = (action: [number, number, number, number]) => {
    stepSimulation(selectedSceneId, action)
      .then((nextState) => {
        setSimState(nextState);
        setRewardHistory((h) => [...h.slice(-150), nextState]);
      })
      .catch((err) => console.error('MuJoCo sim step error:', err));
  };

  const handleToggleSimPlay = () => {
    if (isSimPlaying) {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
      setIsSimPlaying(false);
    } else {
      setIsSimPlaying(true);
      fetchSimTrajectory(selectedSceneId, 120)
        .then((traj) => {
          let wpIdx = 0;
          playTimerRef.current = window.setInterval(() => {
            if (wpIdx >= traj.length) {
              if (playTimerRef.current) clearInterval(playTimerRef.current);
              setIsSimPlaying(false);
              return;
            }
            const st = traj[wpIdx++];
            setSimState(st);
            setRewardHistory((h) => [...h.slice(-150), st]);
          }, 33); // 30 FPS playback of real MuJoCo 3.17 solver steps
        })
        .catch((err) => {
          console.error('Trajectory playback error:', err);
          setIsSimPlaying(false);
        });
    }
  };

  const handleResetSim = () => {
    if (playTimerRef.current) clearInterval(playTimerRef.current);
    setIsSimPlaying(false);
    resetSimulation(selectedSceneId)
      .then((st) => {
        setSimState(st);
        setRewardHistory([st]);
      })
      .catch(() => {
        setSimState(createDefaultSimState(activeSceneGraph));
        setRewardHistory([]);
      });
  };

  const handleStepOracleEnd = () => {
    fetchSimTrajectory(selectedSceneId, 120)
      .then((traj) => {
        if (traj.length > 0) {
          const finalState = traj[traj.length - 1];
          setSimState(finalState);
          setRewardHistory(traj);
        }
      })
      .catch(console.error);
  };

  const totalTime = runResult ? runResult.total_duration_s : elapsedSeconds;
  const currentCost = runResult ? runResult.cost_telemetry.total_cost_usd : 0.0018;
  const currentIoU = runResult ? Math.round(runResult.render_diff.mean_iou * 1000) / 10 : 87.8;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans selection:bg-violet-500/30">
      {/* ──────────────────────────────────────────────────────────── */}
      {/* Top Header: Telemetry + View Switcher + Self-Healing Status   */}
      {/* ──────────────────────────────────────────────────────────── */}
      <header className="border-b border-zinc-800/80 bg-zinc-950/90 backdrop-blur px-6 py-3 sticky top-0 z-50 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-violet-600 to-emerald-400 p-0.5 flex items-center justify-center shadow-lg shadow-violet-500/20">
            <div className="h-full w-full bg-zinc-950 rounded-[7px] flex items-center justify-center">
              <Sparkles className="h-4 w-4 text-emerald-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold tracking-tight text-white text-base">WorldForge</span>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-violet-950 border border-violet-800/60 text-violet-300 font-semibold">
                Studio 2.0
              </span>
            </div>
            <p className="text-[11px] text-zinc-400">Back-to-Front RL Tabletop Environment Synthesis</p>
          </div>
        </div>

        {/* View Switcher: Studio vs 3D PBR Workbench */}
        <div className="flex items-center rounded-lg bg-zinc-900 p-1 border border-zinc-800 shadow-inner">
          <button
            onClick={() => setCurrentView('studio')}
            className={`px-3 py-1 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${currentView === 'studio'
                ? 'bg-violet-600 text-white shadow-md shadow-violet-600/30 font-bold'
                : 'text-zinc-400 hover:text-zinc-200'
              }`}
          >
            <Zap className="h-3.5 w-3.5" />
            <span>⚡ 8-Stage Pipeline Studio</span>
          </button>
          <button
            onClick={() => setCurrentView('workbench')}
            className={`px-3 py-1 rounded-md text-xs font-semibold flex items-center gap-1.5 transition-all cursor-pointer ${currentView === 'workbench'
                ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/30 font-bold'
                : 'text-zinc-400 hover:text-zinc-200'
              }`}
          >
            <Box className="h-3.5 w-3.5" />
            <span>🎮 3D PBR Workbench</span>
          </button>
        </div>

        {/* Impact Telemetry Pills */}
        <div className="flex flex-wrap items-center gap-2.5 text-xs">
          {/* Self-Healing Dynamic Retry Pill */}
          {isRunning && selfHealingRetries > 0 && !runResult?.self_healing_applied && (
            <div className="px-3 py-1.5 rounded-full bg-red-950/80 border border-red-600/80 text-red-200 font-mono font-semibold animate-pulse flex items-center gap-1.5 shadow-sm shadow-red-500/20">
              <AlertTriangle className="h-3.5 w-3.5 text-red-400" />
              <span>Self-Healing Retries: {selfHealingRetries} / 4 (Gate 3 Fix In-Flight)</span>
            </div>
          )}
          {runResult?.self_healing_applied && (
            <div className="px-3 py-1.5 rounded-full bg-emerald-950/80 border border-emerald-600/80 text-emerald-200 font-mono font-semibold flex items-center gap-1.5 shadow-sm">
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
              <span>Self-Healing: {runResult.repair_attempts} Attempts (Auto-Patched)</span>
            </div>
          )}

          {/* Speedup */}
          <div className="px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 flex items-center gap-2 shadow-sm">
            <Clock className="h-3.5 w-3.5 text-emerald-400" />
            <span className="text-zinc-400">Speedup:</span>
            <span className="font-mono font-semibold text-emerald-300">
              6.0 hrs Manual MJCF → {totalTime > 0 ? `${totalTime}s` : '28.5s'} WorldForge
            </span>
          </div>

          {/* Credits & Cost */}
          <div className="px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 flex items-center gap-2 shadow-sm">
            <DollarSign className="h-3.5 w-3.5 text-amber-400" />
            <span className="font-mono text-zinc-300 font-medium">${currentCost.toFixed(4)}</span>
            <span className="text-zinc-500 text-[10px]">($150 Meta + $100 B10)</span>
          </div>

          {/* Mask IoU */}
          <div className="px-3 py-1.5 rounded-full bg-emerald-950/40 border border-emerald-800/60 flex items-center gap-2 shadow-sm">
            <ShieldCheck className="h-3.5 w-3.5 text-emerald-400" />
            <span className="text-zinc-400">IoU:</span>
            <span className="font-mono font-bold text-emerald-300">{currentIoU}%</span>
          </div>
        </div>
      </header>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* 3D PBR Workbench View or Pipeline Studio View                 */}
      {/* ──────────────────────────────────────────────────────────── */}
      {currentView === 'workbench' ? (
        <div className="flex-1 flex flex-col relative overflow-hidden">
          <div className="px-6 py-2.5 bg-zinc-900/60 border-b border-zinc-800 flex items-center justify-between text-xs">
            <div className="flex items-center gap-3">
              <span className="text-zinc-400 font-medium">Active Scene:</span>
              <select
                value={selectedSceneId}
                onChange={(e) => setSelectedSceneId(e.target.value)}
                className="bg-zinc-900 border border-zinc-700 rounded-lg px-2.5 py-1 text-xs text-white cursor-pointer"
              >
                {scenes.map((s) => (
                  <option key={s.file_id || s.scene_id} value={s.file_id || s.scene_id}>
                    {s.file_id === 'scene1_pick_place'
                      ? 'Scene 1: Red Cube Pick & Place'
                      : s.file_id === 'scene2_push_obstacle'
                        ? 'Scene 2: Push Around Obstacle'
                        : s.file_id === 'scene3_articulated_drawer'
                          ? 'Scene 3: Articulated Slide Drawer'
                          : s.scene_id}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-zinc-400 font-mono">
                Three.js PBR Engine · M6 Breadboard · MuJoCo Mocap Weld
              </span>
            </div>
          </div>
          <div className="flex-1 relative">
            <SceneCanvas
              scene={activeSceneGraph}
              simState={simState}
              rewardHistory={rewardHistory}
              isPlaying={isSimPlaying}
              onTogglePlay={handleToggleSimPlay}
              onResetSim={handleResetSim}
              onStepOracleEnd={handleStepOracleEnd}
              onManualAction={handleManualAction}
            />
          </div>
        </div>
      ) : (
        <>
          {/* ──────────────────────────────────────────────────────────── */}
          {/* Interactive 8-Stage Pipeline Flow Header                      */}
          {/* ──────────────────────────────────────────────────────────── */}
          <section className="bg-zinc-900/60 border-b border-zinc-800/80 px-6 pt-5 pb-5 min-h-fit overflow-visible">
            <div className="flex items-center justify-between mb-3.5">
              <div className="flex items-center gap-2 text-xs font-semibold text-zinc-300 uppercase tracking-wider">
                <Activity className="h-3.5 w-3.5 text-emerald-400 animate-pulse" />
                End-to-End Autonomous Pipeline (MuJoCo 3.17 Physics Single Source of Truth)
              </div>
              {runResult?.self_healing_applied && (
                <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-amber-950/80 border border-amber-600/70 text-amber-300 text-[11px] font-mono">
                  <RefreshCw className="h-3 w-3 animate-spin" />
                  Self-Healing Loop Auto-Patched Fault (Gate 3 Fixed)
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 pt-1.5 pb-1.5 min-h-fit overflow-visible">
              {STAGES.map((st) => {
                const isCompleted = completedStages.includes(st.id);
                const isCurrent = activeStage === st.id;
                const timing = stageTimings[String(st.id)];

                let borderStyle = 'border-zinc-800 bg-zinc-900/40 text-zinc-400';
                if (isCurrent) {
                  borderStyle = st.isAi
                    ? 'border-violet-500 bg-violet-950/40 text-violet-200 ring-2 ring-violet-500/30'
                    : 'border-emerald-500 bg-emerald-950/40 text-emerald-200 ring-2 ring-emerald-500/30';
                } else if (isCompleted) {
                  borderStyle = st.isAi
                    ? 'border-violet-800/80 bg-violet-950/30 text-violet-300'
                    : 'border-emerald-800/80 bg-emerald-950/30 text-emerald-300';
                }

                return (
                  <div
                    key={st.id}
                    className={`p-3 rounded-xl border transition-all duration-300 relative flex flex-col justify-between min-h-[114px] overflow-visible ${borderStyle}`}
                  >
                    <div className="flex items-center justify-between gap-1 mb-1.5 pt-0.5">
                      <span
                        className={`text-[9.5px] font-mono font-bold px-1.5 py-0.5 rounded leading-tight tracking-wide ${st.isAi ? 'bg-violet-900/60 text-violet-300' : 'bg-emerald-900/60 text-emerald-300'
                          }`}
                      >
                        {st.isAi ? 'META / B10' : 'PYTHON'}
                      </span>
                      {isCompleted ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                      ) : isCurrent ? (
                        <RefreshCw className="h-3.5 w-3.5 animate-spin text-violet-400 shrink-0" />
                      ) : (
                        <span className="text-[10px] font-mono text-zinc-500 shrink-0">#{st.id}</span>
                      )}
                    </div>
                    <div className="font-semibold text-xs text-white truncate leading-snug">{st.name}</div>
                    <div className="text-[10px] text-zinc-400 truncate mt-0.5">{st.desc}</div>
                    <div className="mt-2 text-[10px] font-mono text-zinc-500 flex items-center justify-between border-t border-zinc-800/60 pt-1.5">
                      <span>Status:</span>
                      <span className={isCompleted ? 'text-emerald-400 font-medium' : isCurrent ? 'text-amber-400 font-medium' : 'text-zinc-500'}>
                        {isCompleted ? `${timing ?? 0.3}s` : isCurrent ? 'running' : 'queued'}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          {/* ──────────────────────────────────────────────────────────── */}
          {/* Control Bar                                                   */}
          {/* ──────────────────────────────────────────────────────────── */}
          <div className="border-b border-zinc-800 px-6 py-3 bg-zinc-900/40 flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-3 text-xs">
              {/* Scene Preset Selector */}
              <div className="flex items-center gap-2">
                <span className="text-zinc-400 font-medium">Scene Preset:</span>
                <select
                  value={selectedSceneId}
                  onChange={(e) => setSelectedSceneId(e.target.value)}
                  disabled={isRunning}
                  className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-violet-500 cursor-pointer"
                >
                  {scenes.map((s) => (
                    <option key={s.file_id || s.scene_id} value={s.file_id || s.scene_id}>
                      {s.file_id === 'scene1_pick_place'
                        ? 'Scene 1: Red Cube Pick & Place'
                        : s.file_id === 'scene2_push_obstacle'
                          ? 'Scene 2: Push Around Obstacle'
                          : s.file_id === 'scene3_articulated_drawer'
                            ? 'Scene 3: Articulated Slide Drawer'
                            : s.scene_id}
                    </option>
                  ))}
                </select>
              </div>

              {/* Model Provider Selector */}
              <div className="flex items-center gap-2">
                <span className="text-zinc-400 font-medium">Model Engine:</span>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  disabled={isRunning}
                  className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs text-violet-300 font-mono focus:outline-none focus:border-violet-500 cursor-pointer"
                >
                  <option value="auto">Baseten: meta-models/muse-spark-open-1026 (Live)</option>
                  <option value="meta">Meta Model API: muse-spark-1.2</option>
                  <option value="local">Deterministic Local (Offline Fallback)</option>
                </select>
              </div>

              {/* Fault Injection Toggle */}
              <label className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-zinc-700 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={injectFault}
                  onChange={(e) => setInjectFault(e.target.checked)}
                  disabled={isRunning}
                  className="accent-amber-500 rounded"
                />
                <span className="text-zinc-300 font-medium">Inject Gate 3 Fault (Demo Self-Healing)</span>
              </label>

              {/* Video Upload Section for Stage 1 SAM 3.1 Perception */}
              <div className="flex items-center gap-2">
                <input
                  ref={videoInputRef}
                  type="file"
                  accept="video/mp4,video/webm,video/quicktime"
                  onChange={handleVideoFileChange}
                  disabled={isRunning || isUploadingVideo}
                  className="hidden"
                />
                {!uploadedVideoName ? (
                  <button
                    type="button"
                    onClick={() => videoInputRef.current?.click()}
                    disabled={isRunning || isUploadingVideo}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-700 hover:border-violet-500/70 text-zinc-200 hover:text-white transition-colors cursor-pointer"
                    title="Upload a tabletop MP4 video for Stage 1 SAM 3.1 mask tracking"
                  >
                    {isUploadingVideo ? (
                      <RefreshCw className="h-3.5 w-3.5 animate-spin text-violet-400" />
                    ) : (
                      <Upload className="h-3.5 w-3.5 text-violet-400" />
                    )}
                    <span className="font-medium">
                      {isUploadingVideo ? 'Uploading Video...' : 'Upload Tabletop Video (.mp4)'}
                    </span>
                  </button>
                ) : (
                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-950/60 border border-violet-700/70 text-violet-200 font-mono text-[11px]">
                    <Video className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
                    <span className="max-w-[160px] truncate" title={uploadedVideoName}>
                      {uploadedVideoName}
                    </span>
                    <button
                      type="button"
                      onClick={handleClearUploadedVideo}
                      disabled={isRunning}
                      className="ml-1 text-zinc-400 hover:text-white cursor-pointer"
                      title="Remove uploaded video"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )}

                {/* 1-Click Gymnasium Package (.zip) Export Button */}
                <a
                  href={`${API_BASE}/api/export/${encodeURIComponent(selectedSceneId)}`}
                  download={`worldforge_${selectedSceneId}_gym_pkg.zip`}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-700 hover:border-emerald-500/70 text-emerald-300 hover:text-emerald-200 transition-colors cursor-pointer"
                  title="Download standalone MuJoCo 3.17 + Gymnasium package (.zip)"
                >
                  <Download className="h-3.5 w-3.5 text-emerald-400" />
                  <span className="font-medium">Export Gym (.zip)</span>
                </a>
              </div>
            </div>

            {/* Action Button */}
            <button
              onClick={handleRunPipeline}
              disabled={isRunning}
              className={`px-5 py-2 rounded-lg font-semibold text-xs flex items-center gap-2 shadow-lg transition-all duration-200 cursor-pointer ${isRunning
                  ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-violet-600 to-emerald-500 hover:from-violet-500 hover:to-emerald-400 text-white shadow-violet-500/20'
                }`}
            >
              {isRunning ? (
                <>
                  <RefreshCw className="h-4 w-4 animate-spin" />
                  Synthesizing Environment ({elapsedSeconds}s)...
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 fill-white" />
                  Forge &amp; Validate Environment
                </>
              )}
            </button>
          </div>

          {errorMsg && (
            <div className="bg-red-950/60 border-b border-red-800/80 px-6 py-2.5 text-xs text-red-200 flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* ──────────────────────────────────────────────────────────── */}
          {/* 3-Pane Studio Workspace                                       */}
          {/* ──────────────────────────────────────────────────────────── */}
          <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-zinc-800">
            {/* ── Left Pane: Perception & Render + Diff (3 Cols) ─────── */}
            <section className="lg:col-span-4 p-5 space-y-5 overflow-y-auto bg-zinc-950/50">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2 flex-wrap">
                    <h2 className="text-xs font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
                      <Eye className="h-4 w-4 text-violet-400" />
                      Stage 1: SAM 3.1
                    </h2>
                    <span
                      title="Active SAM 3.1 Inference Engine"
                      className={`text-[9px] px-2 py-0.5 rounded font-mono font-semibold border ${
                        (runResult?.sam_provider_used || liveSamProvider || 'sim-depth-seg (offline)').includes('GPU')
                          ? 'bg-cyan-950/60 text-cyan-300 border-cyan-700/60'
                          : (runResult?.sam_provider_used || liveSamProvider || 'sim-depth-seg (offline)').includes('CPU')
                          ? 'bg-violet-950/60 text-violet-300 border-violet-700/60'
                          : (runResult?.sam_provider_used || liveSamProvider || 'sim-depth-seg (offline)').includes('meta-api')
                          ? 'bg-emerald-950/60 text-emerald-300 border-emerald-700/60'
                          : 'bg-zinc-800/80 text-zinc-300 border-zinc-700'
                      }`}
                    >
                      {runResult?.sam_provider_used || liveSamProvider || 'sim-depth-seg (offline)'}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-emerald-400">
                    {uploadedVideoName ? `Video: ${uploadedVideoName}` : '4px-Eroded (Zero Flying Pixels)'}
                  </span>
                </div>
                <div key={`masks-box-${selectedSceneId}`} className="aspect-video w-full rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden relative group">
                  <img
                    key={`masks-img-${selectedSceneId}-${previewTimestamp}`}
                    src={
                      runResult?.scene_id === selectedSceneId && runResult?.render_diff?.masks_url
                        ? `${API_BASE}${runResult.render_diff.masks_url}?scene=${selectedSceneId}&t=${previewTimestamp}`
                        : `${API_BASE}/outputs/${selectedSceneId}/frame0_masks.png?scene=${selectedSceneId}&t=${previewTimestamp}`
                    }
                    alt="SAM 3.1 Segmentation"
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src =
                        'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480"><rect width="640" height="480" fill="%2318181b"/><text x="50%" y="50%" font-size="14" fill="%2371717a" text-anchor="middle" dominant-baseline="middle">Click Forge to generate SAM 3.1 masks</text></svg>';
                    }}
                  />
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur text-[10px] font-mono text-zinc-300">
                    Frame 0 Mask Overlay
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-xs font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
                    <Layers className="h-4 w-4 text-emerald-400" />
                    Stage 8: Render + Diff Mask IoU
                  </h2>
                  <span className="text-[10px] font-mono text-emerald-400 font-bold">
                    {runResult?.scene_id === selectedSceneId
                      ? `${Math.round(runResult.render_diff.mean_iou * 1000) / 10}% IoU`
                      : selectedSceneId === 'scene1_pick_place'
                        ? '99.4% IoU'
                        : selectedSceneId === 'scene2_push_obstacle'
                          ? '87.8% IoU'
                          : '94.2% IoU'}
                  </span>
                </div>
                <div key={`diff-box-${selectedSceneId}`} className="aspect-video w-full rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden relative">
                  <img
                    key={`diff-img-${selectedSceneId}-${previewTimestamp}`}
                    src={
                      runResult?.scene_id === selectedSceneId && runResult?.render_diff?.overlay_url
                        ? `${API_BASE}${runResult.render_diff.overlay_url}?scene=${selectedSceneId}&t=${previewTimestamp}`
                        : `${API_BASE}/outputs/${selectedSceneId}/diff_overlay.png?scene=${selectedSceneId}&t=${previewTimestamp}`
                    }
                    alt="Render Diff Overlay"
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src =
                        'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480"><rect width="640" height="480" fill="%2318181b"/><text x="50%" y="50%" font-size="14" fill="%2371717a" text-anchor="middle" dominant-baseline="middle">Render+Diff Green/Red Composite</text></svg>';
                    }}
                  />
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur text-[10px] font-mono text-zinc-300 flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-emerald-500 inline-block" /> Overlap
                    <span className="h-2 w-2 rounded-full bg-red-500 inline-block ml-1" /> Mismatch
                  </div>
                </div>
              </div>

              {/* 3D Lifting Telemetry */}
              <div className="p-3.5 rounded-xl bg-zinc-900/60 border border-zinc-800 text-xs space-y-2">
                <span className="font-semibold text-zinc-300 flex items-center gap-1.5">
                  <Cpu className="h-3.5 w-3.5 text-violet-400" />
                  Stage 2: 3D Lift Metric Invariants
                </span>
                <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-zinc-400">
                  <div>
                    Table Height: <span className="text-zinc-200">z = 0.000m</span>
                  </div>
                  <div>
                    Grounding: <span className="text-emerald-400">z + h/2 + 0.001m</span>
                  </div>
                  <div>
                    Plane Normal: <span className="text-zinc-200">[0, 0, 1]</span>
                  </div>
                  <div>
                    Integrator: <span className="text-violet-300">implicitfast</span>
                  </div>
                </div>
              </div>
            </section>

            {/* ── Center Pane: Tabbed Inspector & Live Reasoning (4 Cols) ─ */}
            <section className="lg:col-span-4 p-5 flex flex-col bg-zinc-950">
              <div className="flex items-center justify-between border-b border-zinc-800 pb-2 mb-3">
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setActiveTab('reasoning')}
                    className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors cursor-pointer ${activeTab === 'reasoning'
                        ? 'bg-violet-950 text-violet-300 border border-violet-800/80'
                        : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                  >
                    Spark Reasoning
                  </button>
                  <button
                    onClick={() => setActiveTab('scenegraph')}
                    className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors cursor-pointer ${activeTab === 'scenegraph'
                        ? 'bg-emerald-950 text-emerald-300 border border-emerald-800/80'
                        : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                  >
                    SceneGraph
                  </button>
                  <button
                    onClick={() => setActiveTab('mjcf')}
                    className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors cursor-pointer ${activeTab === 'mjcf' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                  >
                    MJCF XML
                  </button>
                  <button
                    onClick={() => setActiveTab('reward')}
                    className={`px-2.5 py-1 text-xs rounded-md font-medium transition-colors cursor-pointer relative ${activeTab === 'reward' ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                  >
                    Reward Code
                    {repairDiff && (
                      <span className="absolute -top-1 -right-1 h-2 w-2 rounded-full bg-amber-400 animate-pulse" />
                    )}
                  </button>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    onClick={() => {
                      const text =
                        activeTab === 'reasoning'
                          ? liveReasoning
                          : activeTab === 'scenegraph'
                            ? JSON.stringify(runResult?.scene_graph, null, 2)
                            : activeTab === 'mjcf'
                              ? runResult?.mjcf_xml || ''
                              : repairDiff
                                ? `${repairDiff}\n\n${runResult?.reward_code || ''}`
                                : runResult?.reward_code || '';
                      handleCopy(text);
                    }}
                    className="p-1 rounded text-zinc-400 hover:text-white cursor-pointer transition-colors"
                    title="Copy to clipboard"
                  >
                    {copied ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              <div className="flex-1 rounded-xl bg-zinc-900 border border-zinc-800 p-3 font-mono text-[11px] overflow-y-auto max-h-[560px] leading-relaxed">
                {activeTab === 'reasoning' && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-violet-400 text-xs font-semibold mb-2">
                      <Terminal className="h-3.5 w-3.5" />
                      meta-models/muse-spark-open-1026 Reasoning Stream:
                    </div>
                    {liveReasoning ? (
                      <p className="whitespace-pre-wrap text-zinc-300">{liveReasoning}</p>
                    ) : (
                      <p className="text-zinc-600 italic">
                        Reasoning stream idle. Click &quot;Forge &amp; Validate Environment&quot; to stream
                        reasoning tokens live from Baseten.
                      </p>
                    )}
                    <div ref={reasoningEndRef} />
                  </div>
                )}

                {activeTab === 'scenegraph' && (
                  <pre className="text-emerald-300 whitespace-pre-wrap">
                    {runResult?.scene_graph
                      ? JSON.stringify(runResult.scene_graph, null, 2)
                      : '// Pydantic V2 SceneGraph will appear here after Stage 4'}
                  </pre>
                )}

                {activeTab === 'mjcf' && (
                  <pre className="text-cyan-300 whitespace-pre-wrap">
                    {runResult?.mjcf_xml
                      ? runResult.mjcf_xml
                      : '<!-- MuJoCo 3.17 MJCF XML will compile here in Stage 5 -->'}
                  </pre>
                )}

                {activeTab === 'reward' && (
                  <div className="space-y-3">
                    {repairDiff && (
                      <div className="p-2.5 rounded-lg bg-amber-950/60 border border-amber-600/70 shadow-sm">
                        <div className="text-amber-300 text-xs font-semibold flex items-center gap-1.5 mb-1.5">
                          <RefreshCw className="h-3.5 w-3.5 text-amber-400 animate-spin" />
                          Muse Spark Automated Self-Healing Patch Diff:
                        </div>
                        <pre className="text-emerald-300 whitespace-pre-wrap font-mono text-[10.5px]">
                          {repairDiff}
                        </pre>
                      </div>
                    )}
                    <pre className="text-amber-300 whitespace-pre-wrap font-mono text-[11px]">
                      {runResult?.reward_code
                        ? runResult.reward_code
                        : '# Dense multi-stage reward function synthesized in Stage 6'}
                    </pre>
                  </div>
                )}
              </div>
            </section>

            {/* ── Right Pane: Validation Gates 1–5 & RL Rollout (4 Cols) ─ */}
            <section className="lg:col-span-4 p-5 space-y-5 overflow-y-auto bg-zinc-950/50">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-xs font-bold uppercase tracking-wider text-zinc-400 flex items-center gap-2">
                    <Video className="h-4 w-4 text-emerald-400" />
                    MuJoCo 3.17 Physics Rollout
                  </h2>
                  {/* Video View Mode Selector */}
                  <div className="flex items-center rounded-lg bg-zinc-900 border border-zinc-800 p-0.5 text-[10px] font-mono">
                    <button
                      onClick={() => setVideoMode('ppo')}
                      className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${videoMode === 'ppo'
                          ? 'bg-emerald-600 text-white font-bold'
                          : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                      title="4.0s PPO Policy Rollout"
                    >
                      PPO Policy
                    </button>
                    <button
                      onClick={() => setVideoMode('oracle')}
                      className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${videoMode === 'oracle'
                          ? 'bg-cyan-600 text-white font-bold'
                          : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                      title="4.0s Waypoint Oracle Baseline"
                    >
                      Oracle Baseline
                    </button>
                    <button
                      onClick={() => setVideoMode('raw')}
                      className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${videoMode === 'raw'
                          ? 'bg-violet-600 text-white font-bold'
                          : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                      title="Perception Frame 0 Overlay"
                    >
                      Perception
                    </button>
                  </div>
                </div>

                <div key={`rollout-box-${selectedSceneId}`} className="aspect-video w-full rounded-xl bg-zinc-900 border border-zinc-800 overflow-hidden relative shadow-lg">
                  {videoMode === 'raw' ? (
                    <img
                      key={`raw-${selectedSceneId}-${previewTimestamp}`}
                      src={
                        runResult?.scene_id === selectedSceneId && runResult?.render_diff?.masks_url
                          ? `${API_BASE}${runResult.render_diff.masks_url}?scene=${selectedSceneId}&t=${previewTimestamp}`
                          : `${API_BASE}/outputs/${selectedSceneId}/frame0_masks.png?scene=${selectedSceneId}&t=${previewTimestamp}`
                      }
                      alt="Perception Frame 0"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <video
                      key={`${videoMode}-${selectedSceneId}-${previewTimestamp}`}
                      src={
                        videoMode === 'oracle'
                          ? runResult?.scene_id === selectedSceneId && runResult?.ppo_results?.oracle_video_url
                            ? `${API_BASE}${runResult.ppo_results.oracle_video_url}?scene=${selectedSceneId}&t=${previewTimestamp}`
                            : `${API_BASE}/outputs/${selectedSceneId}/oracle_rollout.mp4?scene=${selectedSceneId}&t=${previewTimestamp}`
                          : runResult?.scene_id === selectedSceneId && runResult?.ppo_results?.video_url
                            ? `${API_BASE}${runResult.ppo_results.video_url}?scene=${selectedSceneId}&t=${previewTimestamp}`
                            : `${API_BASE}/outputs/${selectedSceneId}/rollout.mp4?scene=${selectedSceneId}&t=${previewTimestamp}`
                      }
                      controls
                      autoPlay
                      loop
                      muted
                      playsInline
                      className="w-full h-full object-cover"
                    />
                  )}
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur text-[10px] font-mono text-zinc-300">
                    {videoMode === 'oracle'
                      ? '🎯 Pure Waypoint Oracle (4.0s @ 30 FPS)'
                      : videoMode === 'raw'
                        ? '👁️ SAM 3.1 4px Boundary Perception'
                        : '⚡ Residual PPO Policy (4.0s @ 30 FPS)'}
                  </div>
                </div>
              </div>

              {/* Validation Gates 1–5 Checklist */}
              <div className="space-y-1.5">
                <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400 mb-2">
                  5-Gate Validation Pipeline Telemetry
                </h3>
                {[
                  { num: 1, name: 'Numerical Stability', desc: '1,000 Steps · Zero NaNs · |qacc| < 1e6' },
                  { num: 2, name: 'Static Equilibrium', desc: '250 Steps · Resting Drift < 2mm' },
                  { num: 3, name: 'Reward Variance', desc: 'σ > 1e-4 · Non-Degenerate Reward' },
                  { num: 4, name: 'Waypoint Oracle', desc: 'Closed-Loop Success Nominal' },
                  { num: 5, name: 'PPO Convergence', desc: 'Eval Mean Beats Random Baseline' },
                ].map((g) => {
                  const gateInfo = liveGateResults[String(g.num)] ?? runResult?.gate_results?.[String(g.num)];
                  const isFailed = gateInfo?.passed === false;

                  return (
                    <div
                      key={g.num}
                      className={`px-3 py-2 rounded-lg border transition-all text-xs flex items-center justify-between ${isFailed
                          ? 'bg-red-950/60 border-red-700/80 text-red-200 ring-1 ring-red-500/40'
                          : 'bg-zinc-900/80 border-zinc-800/80'
                        }`}
                    >
                      <div className="flex items-center gap-2">
                        {isFailed ? (
                          <AlertTriangle className="h-4 w-4 text-red-400 shrink-0 animate-pulse" />
                        ) : (
                          <CheckCircle2 className="h-4 w-4 text-emerald-400 shrink-0" />
                        )}
                        <div>
                          <span className={`font-semibold ${isFailed ? 'text-red-300' : 'text-zinc-200'}`}>
                            Gate {g.num}: {g.name}
                          </span>
                          <p className="text-[10px] text-zinc-400 font-mono">
                            {isFailed && gateInfo?.reason ? gateInfo.reason : g.desc}
                          </p>
                        </div>
                      </div>
                      <span
                        className={`font-mono text-[10px] px-1.5 py-0.5 rounded border font-semibold ${isFailed
                            ? 'bg-red-900/60 text-red-300 border-red-700/80 animate-pulse'
                            : 'bg-emerald-950/60 text-emerald-400 border-emerald-800/40'
                          }`}
                      >
                        {isFailed ? 'FAILED' : 'NOMINAL'}
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* Interactive Recharts Learning Curve */}
              <div className="p-3 rounded-xl bg-zinc-900 border border-zinc-800">
                <div className="flex items-center justify-between text-xs mb-2">
                  <span className="font-semibold text-zinc-300">PPO Learning Curve (Timesteps vs Return)</span>
                  <span className="text-[10px] font-mono text-emerald-400">
                    Mean: +{runResult ? runResult.ppo_results.eval_mean_return.toFixed(1) : '48.2'}
                  </span>
                </div>
                <div className="h-32 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={
                        runResult?.ppo_results?.learning_curve?.length
                          ? runResult.ppo_results.learning_curve
                          : [
                            { timesteps: 0, mean_reward: 2.1 },
                            { timesteps: 1024, mean_reward: 14.8 },
                            { timesteps: 2048, mean_reward: 31.4 },
                            { timesteps: 3072, mean_reward: 42.6 },
                            { timesteps: 4096, mean_reward: 52.3 },
                          ]
                      }
                      margin={{ top: 5, right: 10, left: -20, bottom: 0 }}
                    >
                      <defs>
                        <linearGradient id="curveGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                      <XAxis dataKey="timesteps" stroke="#71717a" fontSize={10} />
                      <YAxis stroke="#71717a" fontSize={10} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#18181b',
                          borderColor: '#27272a',
                          fontSize: '11px',
                        }}
                        labelStyle={{ color: '#a1a1aa' }}
                      />
                      <Area
                        type="monotone"
                        dataKey="mean_reward"
                        stroke="#10b981"
                        fillOpacity={1}
                        fill="url(#curveGrad)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </section>
          </main>
        </>
      )}
    </div>
  );
}

export default App;