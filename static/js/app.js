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
  document.querySelectorAll('[data-video-ms]').forEach((button) => {
    button.addEventListener('click', () => {
      if (!video) return;
      video.currentTime = Number(button.dataset.videoMs || 0) / 1000;
      video.play().catch(() => {});
      video.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });
  });

  const banner = document.querySelector('[data-analysis-status]');
  if (banner && banner.dataset.terminal !== 'true') {
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
