// =============================================
//  Medical Image Super Resolution
//  script.js — All JavaScript logic
// =============================================

// Global variable to store results and original image
let results = {};
let origSrc  = '';

// Processing steps shown in progress bar
const steps = [
  [10,  'Loading models...'],
  [25,  'Processing Original baseline...'],
  [40,  'Running Vision Transformer (ViT)...'],
  [60,  'Running Diffusion Model (T=1000)...'],
  [80,  'Running ViT + Diffusion pipeline...'],
  [92,  'Calculating PSNR / SSIM metrics...'],
  [100, 'All models complete!']
];

// =============================================
//  HANDLE FILE UPLOAD
// =============================================
function handleFile(e) {
  const file = e.target.files[0];
  if (!file) return;

  // Show filename in upload zone
  document.getElementById('uploadText').textContent = file.name + ' — Ready!';

  // Enable run button
  document.getElementById('runBtn').disabled = false;

  // Read original image for display
  const reader = new FileReader();
  reader.onload = ev => { origSrc = ev.target.result; };
  reader.readAsDataURL(file);
}

// =============================================
//  RUN ALL MODELS
// =============================================
async function runAll() {
  const file  = document.getElementById('fileInput').files[0];
  if (!file) return;

  const scale = document.getElementById('scaleSelect').value;

  // Show progress bar
  document.getElementById('progressCard').style.display = 'block';
  document.getElementById('runBtn').disabled = true;

  // Animate progress bar steps
  let i = 0;
  const interval = setInterval(() => {
    if (i < steps.length) {
      document.getElementById('barFill').style.width  = steps[i][0] + '%';
      document.getElementById('progStep').textContent = steps[i][1];
      i++;
    }
  }, 400);

  // Send image to Flask backend
  const formData = new FormData();
  formData.append('image', file);
  formData.append('scale', scale);

  try {
    const response = await fetch('/enhance_all', { method: 'POST', body: formData });
    const data     = await response.json();

    clearInterval(interval);
    document.getElementById('barFill').style.width  = '100%';
    document.getElementById('progStep').textContent = 'Done!';

    setTimeout(() => {
      document.getElementById('progressCard').style.display = 'none';
      document.getElementById('runBtn').disabled = false;
      results = data;
      updateUI(data, scale);
    }, 500);

  } catch (err) {
    clearInterval(interval);
    alert('Error: ' + err.message);
    document.getElementById('runBtn').disabled = false;
  }
}

// =============================================
//  UPDATE ALL UI WITH RESULTS
// =============================================
function updateUI(data, scale) {

  // --- HOME TAB: 4 cards ---
  showResult('orig', data, scale);
  showResult('vit',  data, scale);
  showResult('diff', data, scale);
  showResult('comb', data, scale);

  // --- BOTTOM METRICS BAR ---
  document.getElementById('bm-psnr').textContent  = data.psnr_comb || '—';
  document.getElementById('bm-ssim').textContent  = data.ssim_comb || '—';
  document.getElementById('bm-scale').textContent = scale + 'x';
  const acc = data.ssim_comb ? (data.ssim_comb * 100).toFixed(1) + '%' : '—';
  document.getElementById('bm-acc').textContent   = acc;

  // --- SINGLE TABS: Original ---
  setImg('so-orig', origSrc);
  setImg('so-res',  'data:image/png;base64,' + data.orig);
  hideEl('sph-orig'); hideEl('sph-orig-r');
  document.getElementById('sm-psnr-orig').textContent  = data.psnr_orig || '—';
  document.getElementById('sm-ssim-orig').textContent  = data.ssim_orig || '—';
  document.getElementById('sm-scale-orig').textContent = scale + 'x';

  // --- SINGLE TABS: ViT ---
  setImg('sv-orig', origSrc);
  setImg('sv-res',  'data:image/png;base64,' + data.vit);
  hideEl('sph-vit-o'); hideEl('sph-vit-r');
  document.getElementById('sm-psnr-vit').textContent  = data.psnr_vit || '—';
  document.getElementById('sm-ssim-vit').textContent  = data.ssim_vit || '—';
  document.getElementById('sm-scale-vit').textContent = scale + 'x';

  // --- SINGLE TABS: Diffusion ---
  setImg('sd-orig', origSrc);
  setImg('sd-res',  'data:image/png;base64,' + data.diff);
  hideEl('sph-diff-o'); hideEl('sph-diff-r');
  document.getElementById('sm-psnr-diff').textContent  = data.psnr_diff || '—';
  document.getElementById('sm-ssim-diff').textContent  = data.ssim_diff || '—';
  document.getElementById('sm-scale-diff').textContent = scale + 'x';

  // --- SINGLE TABS: Combined ---
  setImg('sc-orig', origSrc);
  setImg('sc-res',  'data:image/png;base64,' + data.comb);
  hideEl('sph-comb-o'); hideEl('sph-comb-r');
  document.getElementById('sm-psnr-comb').textContent  = data.psnr_comb || '—';
  document.getElementById('sm-ssim-comb').textContent  = data.ssim_comb || '—';
  document.getElementById('sm-scale-comb').textContent = scale + 'x';

  // --- COMPARE TAB ---
  updateCompare('orig', data);
  updateCompare('vit',  data);
  updateCompare('diff', data);
  updateCompare('comb', data);
}

// Helper: update one home tab card
function showResult(model, data, scale) {
  const src = 'data:image/png;base64,' + data[model];
  setImg('img-' + model, src);
  hideEl('ph-' + model);
  document.getElementById('psnr-' + model).textContent = data['psnr_' + model] || '—';
  document.getElementById('ssim-' + model).textContent = data['ssim_' + model] || '—';
}

// Helper: update compare tab bar
function updateCompare(model, data) {
  document.getElementById('cmp-psnr-' + model).textContent = data['psnr_' + model] || '—';
  document.getElementById('cmp-ssim-' + model).textContent = 'SSIM: ' + (data['ssim_' + model] || '—');
  const pct = data['psnr_' + model]
    ? Math.min(100, (data['psnr_' + model] / 40) * 100).toFixed(0)
    : 0;
  document.getElementById('cmp-bar-' + model).style.width = pct + '%';
}

// =============================================
//  DOWNLOAD ENHANCED IMAGE
// =============================================
function download(model) {
  if (!results[model]) {
    alert('Please run the models first!');
    return;
  }
  const a      = document.createElement('a');
  a.href       = 'data:image/png;base64,' + results[model];
  a.download   = 'medsr_' + model + '_enhanced.png';
  a.click();
}

// =============================================
//  TAB NAVIGATION
// =============================================
function showTab(name, btn) {
  // Hide all tabs
  document.getElementById('tab-home').style.display = 'none';
  document.querySelectorAll('.single-tab').forEach(t => t.classList.remove('active'));
  // Remove active from all nav buttons
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  // Activate clicked button
  btn.classList.add('active');
  // Show selected tab
  if (name === 'home') {
    document.getElementById('tab-home').style.display = 'block';
  } else {
    document.getElementById('tab-' + name).classList.add('active');
  }
}

// =============================================
//  UTILITY FUNCTIONS
// =============================================

// Set image src and show it
function setImg(id, src) {
  const el = document.getElementById(id);
  if (el) {
    el.src           = src;
    el.style.display = 'block';
  }
}

// Hide an element
function hideEl(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = 'none';
}
