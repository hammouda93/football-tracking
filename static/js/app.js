(() => {
  "use strict";

  const videoInput = document.querySelector('.drop-zone input[type="file"]');
  const dropZone = document.querySelector('.drop-zone');
  const fileName = document.querySelector('[data-file-name]');
  if (videoInput && dropZone && fileName) {
    const showFile = () => {
      const file = videoInput.files && videoInput.files[0];
      if (!file) return;
      const megabytes = (file.size / 1024 / 1024).toFixed(1);
      fileName.textContent = `${file.name} · ${megabytes} Mo`;
    };
    videoInput.addEventListener('change', showFile);
    ['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, () => dropZone.classList.add('dragover')));
    ['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragover')));
  }

  const video = document.querySelector('#match-video');
  const seekVideo = (seconds) => {
    if (!video) return;
    const seek = () => {
      video.currentTime = Math.max(0, Number(seconds || 0));
      video.play().catch(() => {});
      video.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
    if (video.readyState >= 1) {
      seek();
      return;
    }
    video.addEventListener('loadedmetadata', seek, { once: true });
    video.load();
  };
  document.querySelectorAll('[data-video-ms]').forEach((button) => {
    button.addEventListener('click', () => {
      seekVideo(Number(button.dataset.videoMs || 0) / 1000);
    });
  });

  const parseTimecode = (value) => {
    const parts = String(value || '').trim().replace(',', '.').split(':');
    if (!parts.length || parts.some((part) => part === '' || Number.isNaN(Number(part)))) return null;
    let seconds = 0;
    parts.forEach((part) => { seconds = seconds * 60 + Number(part); });
    return seconds;
  };

  const formatTimecode = (seconds) => {
    const totalMs = Math.max(0, Math.round(Number(seconds || 0) * 1000));
    const millis = totalMs % 1000;
    const totalSeconds = Math.floor(totalMs / 1000);
    const secs = totalSeconds % 60;
    const totalMinutes = Math.floor(totalSeconds / 60);
    const minutes = totalMinutes % 60;
    const hours = Math.floor(totalMinutes / 60);
    const base = hours
      ? `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
      : `${String(totalMinutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    return `${base}.${String(millis).padStart(3, '0')}`;
  };

  document.querySelectorAll('[data-preview-input]').forEach((button) => {
    button.addEventListener('click', () => {
      const input = document.querySelector(`[name="${button.dataset.previewInput}"]`);
      const seconds = parseTimecode(input?.value);
      if (seconds === null) return;
      seekVideo(seconds);
    });
  });

  document.querySelectorAll('[data-set-period]').forEach((button) => {
    button.addEventListener('click', () => {
      if (!video) return;
      const input = document.querySelector(`[name="${button.dataset.setPeriod}"]`);
      if (input) input.value = formatTimecode(video.currentTime);
    });
  });

  const banner = document.querySelector('[data-analysis-status]');
  if (banner && banner.dataset.terminal !== 'true') {
    const livePreview = banner.querySelector('[data-live-preview]');
    const livePreviewContainer = banner.querySelector('[data-live-preview-container]');
    const refreshLivePreview = (url) => {
      if (!livePreview || !url) return;
      livePreview.onload = () => {
        if (livePreviewContainer) livePreviewContainer.hidden = false;
      };
      livePreview.src = `${url}${url.includes('?') ? '&' : '?'}v=${Date.now()}`;
    };
    const poll = async () => {
      try {
        const response = await fetch(banner.dataset.analysisStatus, { headers: { Accept: 'application/json' } });
        if (!response.ok) return;
        const data = await response.json();
        const progress = banner.querySelector('[data-progress-bar]');
        const progressValue = banner.querySelector('[data-progress-value]');
        const stage = banner.querySelector('[data-stage-label]');
        const progressDetail = banner.querySelector('[data-progress-detail]');
        const status = banner.querySelector('[data-status-label]');
        const error = banner.querySelector('[data-analysis-error]');
        if (progress) progress.style.width = `${data.progress}%`;
        if (progressValue) progressValue.textContent = `${data.progress}%`;
        if (stage) stage.textContent = data.stage_label;
        if (progressDetail) progressDetail.textContent = data.progress_detail?.label || '';
        if (status) status.textContent = data.status_label;
        if (error && data.error) error.textContent = data.error;
        if (data.stage === 'tracking') {
          refreshLivePreview(data.live_preview_url || livePreview?.dataset.livePreviewUrl);
        }
        if (['completed', 'review', 'failed', 'cancelled'].includes(data.status)) {
          window.setTimeout(() => window.location.reload(), 600);
          return;
        }
        window.setTimeout(poll, 2000);
      } catch (_) {
        window.setTimeout(poll, 5000);
      }
    };
    window.setTimeout(poll, 900);
  }

  window.setTimeout(() => {
    document.querySelectorAll('.message').forEach((message) => {
      message.style.transition = 'opacity .4s ease, transform .4s ease';
      message.style.opacity = '0';
      message.style.transform = 'translateY(-8px)';
      window.setTimeout(() => message.remove(), 450);
    });
  }, 5000);
})();
