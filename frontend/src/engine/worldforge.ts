/**
 * WorldForge API Client & Domain Models
 * Connects the Studio-Grade UI directly to the Python FastAPI backend (MuJoCo 3.17, Baseten, SB3).
 */

export const API_BASE = "http://localhost:8000";

export interface CameraSpec {
  pos: [number, number, number];
  lookat: [number, number, number];
  fovy?: number;
  resolution?: [number, number];
}

export interface JointSpec {
  type: "free" | "hinge" | "slide" | "fixed";
  axis?: [number, number, number];
  range?: [number, number] | null;
  damping?: number;
  is_prior_guess?: boolean;
}

export interface ObjectNode {
  id: string;
  label: string;
  primitive: "box" | "cylinder" | "sphere";
  size: number[];
  position: [number, number, number];
  quaternion?: [number, number, number, number];
  material: string;
  is_static?: boolean;
  joint: JointSpec;
  parent_id?: string | null;
}

export interface TaskSpec {
  task_type: "pick_and_place" | "push_to_target" | "articulate_joint";
  description: string;
  manipulated_object_id: string;
  target_position?: [number, number, number] | null;
  target_joint_delta?: number | null;
  waypoints: [number, number, number, number][];
}

export interface SceneGraph {
  scene_id: string;
  table_z: number;
  camera: CameraSpec;
  objects: ObjectNode[];
  task: TaskSpec;
}

export interface SceneMeta {
  scene_id: string;
  file_id: string;
  file_name: string;
  task_type: string;
  description: string;
  num_objects: number;
  objects: Array<{
    id: string;
    label: string;
    primitive: string;
    material: string;
    joint_type: string;
  }>;
  table_z: number;
  has_articulated_joint: boolean;
}

export interface SystemStatus {
  status: string;
  engine: string;
  mujoco_version: string;
  baseten: {
    configured: boolean;
    model: string;
    base_url: string;
    grant: string;
  };
  meta: {
    configured: boolean;
    model: string;
    grant: string;
  };
  modal: {
    configured: boolean;
    grant: string;
  };
}

export interface CostTelemetry {
  input_tokens: number;
  output_tokens: number;
  cpu_core_seconds: number;
  gib_seconds: number;
  total_cost_usd: number;
  pricing_tier: string;
}

export interface PpoResults {
  eval_mean_return: number;
  eval_std_return: number;
  learning_curve: Array<{ timesteps: number; mean_reward: number }>;
  video_url: string;
}

export interface RenderDiffResults {
  initial_mean_iou: number;
  mean_iou: number;
  refined: boolean;
  iterations_used: number;
  overlay_url: string;
  masks_url?: string;
}

export interface PipelineRunResult {
  scene_id: string;
  total_duration_s: number;
  stage_timings: Record<string, number>;
  cost_telemetry: CostTelemetry;
  reasoning_trace: string;
  scene_graph: SceneGraph;
  mjcf_xml: string;
  reward_code: string;
  gate_results: Record<string, any>;
  repair_attempts: number;
  self_healing_applied: boolean;
  ppo_results: PpoResults;
  render_diff: RenderDiffResults;
}

export interface PipelineStageEvent {
  stage: number | string;
  name?: string;
  status?: "running" | "streaming" | "completed" | "self_healing" | "error" | "complete";
  duration_s?: number;
  chunk?: string;
  reasoning_trace?: string;
  message?: string;
  scene_graph?: SceneGraph;
  mjcf_xml?: string;
  reward_code?: string;
  gate_results?: Record<string, any>;
  repair_attempts?: number;
  self_healing_applied?: boolean;
  ppo_eval_mean_return?: number;
  learning_curve?: Array<{ timesteps: number; mean_reward: number }>;
  video_url?: string;
  initial_mean_iou?: number;
  mean_iou?: number;
  refined?: boolean;
  iterations_used?: number;
  overlay_url?: string;
  masks_image_url?: string;
  summary?: PipelineRunResult;
  error?: string;
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) throw new Error(`Status HTTP ${res.status}`);
  return res.json();
}

export async function fetchScenes(): Promise<SceneMeta[]> {
  const res = await fetch(`${API_BASE}/api/scenes`);
  if (!res.ok) throw new Error(`Scenes HTTP ${res.status}`);
  return res.json();
}

export function streamPipeline(
  params: {
    scene_id: string;
    inject_fault: boolean;
    provider: string;
    quick_ppo?: boolean;
  },
  onEvent: (ev: PipelineStageEvent) => void,
  onError: (err: any) => void,
  onDone?: () => void,
): () => void {
  const query = new URLSearchParams({
    scene_id: params.scene_id,
    inject_fault: String(params.inject_fault),
    provider: params.provider,
    quick_ppo: String(params.quick_ppo ?? true),
  });

  const eventSource = new EventSource(`${API_BASE}/api/run/stream?${query.toString()}`);

  eventSource.onmessage = (e) => {
    try {
      const data: PipelineStageEvent = JSON.parse(e.data);
      onEvent(data);
      if (data.stage === "complete" || data.stage === "done") {
        eventSource.close();
        if (onDone) onDone();
      }
    } catch (err) {
      console.error("Failed to parse SSE payload", err);
    }
  };

  eventSource.onerror = (err) => {
    onError(err);
    eventSource.close();
  };

  return () => {
    eventSource.close();
  };
}
