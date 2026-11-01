const fs = require('fs');
const path = require('path');
const { chromium } = require(path.resolve(__dirname, '../frontend/node_modules/playwright'));

const ARTIFACTS_DIR = 'C:\\Users\\Kazali\\.gemini\\antigravity\\brain\\0154df68-8610-446c-9915-ce7aa48fea02';

async function run() {
  console.log('=== WorldForge STUDIO 2.0 End-to-End Automated Browser Verification ===');
  const browser = await chromium.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 }
  });

  const page = await context.newPage();

  const consoleLogs = [];
  const networkErrors = [];

  page.on('console', msg => {
    const text = msg.text();
    consoleLogs.push({ type: msg.type(), text });
    if (msg.type() === 'error') {
      console.error('[Browser Console Error]', text);
    }
  });

  page.on('response', resp => {
    if (resp.status() >= 400) {
      networkErrors.push({ url: resp.url(), status: resp.status() });
      console.error('[Network Error]', resp.status(), resp.url());
    }
  });

  // Step 0: Open http://localhost:5173
  console.log('\n[Step 0] Boot & Health Check...');
  await page.goto('http://localhost:5173', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);
  console.log('Page loaded successfully!');

  // Checkpoint 1: Layout & 8-Stage Pipeline Bar Visual Audit
  console.log('\n[Checkpoint 1] Visual Audit of 8-Stage Pipeline Bar & Telemetry...');
  const cp1Path = path.join(ARTIFACTS_DIR, 'checkpoint1_layout.png');
  await page.screenshot({ path: cp1Path, fullPage: false });
  console.log('Saved Checkpoint 1 screenshot to:', cp1Path);

  // Checkpoint 2: Scene Preset Switching & Left/Right Image Sync
  console.log('\n[Checkpoint 2] Scene Preset Switching...');
  
  // Scene 1
  const sceneSelect = page.locator('select').first();
  await sceneSelect.selectOption('scene1_pick_place');
  await page.waitForTimeout(1500);
  const cp2s1Path = path.join(ARTIFACTS_DIR, 'checkpoint2_scene1.png');
  await page.screenshot({ path: cp2s1Path });
  console.log('Saved Checkpoint 2 (Scene 1) screenshot to:', cp2s1Path);

  // Scene 2
  await sceneSelect.selectOption('scene2_push_obstacle');
  await page.waitForTimeout(1500);
  const cp2s2Path = path.join(ARTIFACTS_DIR, 'checkpoint2_scene2.png');
  await page.screenshot({ path: cp2s2Path });
  console.log('Saved Checkpoint 2 (Scene 2) screenshot to:', cp2s2Path);

  // Scene 3
  await sceneSelect.selectOption('scene3_articulated_drawer');
  await page.waitForTimeout(1500);
  const cp2s3Path = path.join(ARTIFACTS_DIR, 'checkpoint2_scene3.png');
  await page.screenshot({ path: cp2s3Path });
  console.log('Saved Checkpoint 2 (Scene 3) screenshot to:', cp2s3Path);

  // Checkpoint 3: Full 8-Stage Pipeline Execution
  console.log('\n[Checkpoint 3] Full 8-Stage Pipeline Execution on Scene 1...');
  await sceneSelect.selectOption('scene1_pick_place');
  await page.waitForTimeout(1000);

  // Ensure Inject Fault is unchecked
  const faultCheckbox = page.locator('input[type="checkbox"]');
  if (await faultCheckbox.isChecked()) {
    await faultCheckbox.uncheck();
  }

  // Click Forge & Validate Environment
  const forgeBtn = page.getByRole('button', { name: /Forge & Validate Environment/i });
  console.log('Clicking Forge & Validate Environment...');
  await forgeBtn.click();

  // Wait for completion (up to 120 seconds)
  console.log('Waiting for pipeline to complete...');
  await page.waitForFunction(() => {
    const forge = Array.from(document.querySelectorAll('button')).find(b => b.textContent && b.textContent.includes('Forge & Validate'));
    return forge && !forge.disabled;
  }, null, { timeout: 120000 });

  await page.waitForTimeout(3000);
  const cp3Complete = path.join(ARTIFACTS_DIR, 'checkpoint3_stage_complete.png');
  await page.screenshot({ path: cp3Complete });
  console.log('Saved Checkpoint 3 Complete screenshot to:', cp3Complete);

  // Click center tabs
  console.log('Checking Center Panel tabs...');
  const sparkTab = page.getByRole('button', { name: /Spark Reasoning/i });
  const sgTab = page.getByRole('button', { name: /SceneGraph/i });
  const mjcfTab = page.getByRole('button', { name: /MJCF XML/i });
  const rewardTab = page.getByRole('button', { name: /Reward Code/i });

  if (await sparkTab.count() > 0) {
    await sparkTab.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'checkpoint3_reasoning.png') });
  }

  if (await sgTab.count() > 0) {
    await sgTab.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'checkpoint3_scenegraph.png') });
  }

  if (await mjcfTab.count() > 0) {
    await mjcfTab.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'checkpoint3_mjcf.png') });
  }

  if (await rewardTab.count() > 0) {
    await rewardTab.click();
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'checkpoint3_reward.png') });
  }

  // Checkpoint 4: Rollout Video Playback
  console.log('\n[Checkpoint 4] Rollout Video Playback...');
  const ppoBtn = page.getByRole('button', { name: /PPO Policy/i });
  const oracleBtn = page.getByRole('button', { name: /Oracle Baseline/i });
  const percBtn = page.getByRole('button', { name: /Perception/i });

  if (await ppoBtn.count() > 0) {
    await ppoBtn.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'checkpoint4_ppo_video.png') });
  }

  if (await oracleBtn.count() > 0) {
    await oracleBtn.click();
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(ARTIFACTS_DIR, 'checkpoint4_oracle_video.png') });
  }

  // Checkpoint 5: Fast Self-Healing Loop on Scene 3
  console.log('\n[Checkpoint 5] Fast Self-Healing Loop on Scene 3...');
  await sceneSelect.selectOption('scene3_articulated_drawer');
  await page.waitForTimeout(1000);

  // Check Inject Gate 3 Fault
  await faultCheckbox.check();
  await page.waitForTimeout(500);

  const tStart = Date.now();
  console.log('Clicking Forge & Validate Environment with Fault Injection...');
  await forgeBtn.click();

  await page.waitForFunction(() => {
    const forge = Array.from(document.querySelectorAll('button')).find(b => b.textContent && b.textContent.includes('Forge & Validate'));
    return forge && !forge.disabled;
  }, null, { timeout: 60000 });

  const durationSec = (Date.now() - tStart) / 1000;
  console.log(`Self-healing finished in: ${durationSec.toFixed(2)}s`);

  // Switch to Reward Code tab to show diff
  if (await rewardTab.count() > 0) {
    await rewardTab.click();
    await page.waitForTimeout(1000);
  }
  const cp5DiffPath = path.join(ARTIFACTS_DIR, 'checkpoint5_self_healing_diff.png');
  await page.screenshot({ path: cp5DiffPath });
  console.log('Saved Checkpoint 5 Self-Healing Diff to:', cp5DiffPath);

  // Checkpoint 6: Interactive 3D PBR Workbench
  console.log('\n[Checkpoint 6] Interactive 3D PBR Workbench Verification...');
  const workbenchBtn = page.getByRole('button', { name: /3D PBR Workbench/i }).first();
  await workbenchBtn.click();
  await page.waitForTimeout(2000);

  const cp6Init = path.join(ARTIFACTS_DIR, 'checkpoint6_workbench_initial.png');
  await page.screenshot({ path: cp6Init });
  console.log('Saved Checkpoint 6 Initial Workbench to:', cp6Init);

  // Click camera presets
  const camPresets = ['Isometric Rig', 'MuJoCo render_cam', 'Top XY Plane', 'Front XZ'];
  for (const cam of camPresets) {
    const btn = page.getByRole('button', { name: new RegExp(cam, 'i') });
    if (await btn.count() > 0) {
      await btn.first().click();
      await page.waitForTimeout(300);
    }
  }

  // Click Mocap Actuator buttons
  const plusX = page.getByRole('button', { name: '+X', exact: true });
  if (await plusX.count() > 0) {
    await plusX.click();
    await page.waitForTimeout(200);
    await plusX.click();
    await page.waitForTimeout(200);
  }

  // Run Waypoint Oracle
  const oracleRunBtn = page.getByRole('button', { name: /Run Waypoint Oracle|Execute Oracle Rollout/i });
  if (await oracleRunBtn.count() > 0) {
    console.log('Executing Waypoint Oracle in 3D Workbench...');
    await oracleRunBtn.first().click();
    await page.waitForTimeout(4000);
  }

  const cp6Oracle = path.join(ARTIFACTS_DIR, 'checkpoint6_workbench_oracle.png');
  await page.screenshot({ path: cp6Oracle });
  console.log('Saved Checkpoint 6 Oracle to:', cp6Oracle);

  await browser.close();
  console.log('\n=== All Checkpoints 1-6 Successfully Executed & Screen Captured! ===');
}

run().catch(err => {
  console.error('E2E Verification Failed:', err);
  process.exit(1);
});
