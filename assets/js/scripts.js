// === 공통: 연도 출력 ===
const yearEl = document.getElementById('year');
if (yearEl) yearEl.textContent = new Date().getFullYear();


// === 테마 고정: 무조건 light ===
const root = document.documentElement;

// 혹시 이전에 저장된 테마가 남아있으면 제거
localStorage.removeItem('theme');

// 항상 라이트로 고정
root.setAttribute('data-theme', 'light');


// === 스크롤 리빌 ===
const revealEls = document.querySelectorAll('.reveal');
const io = new IntersectionObserver((entries, obs) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('show');
      obs.unobserve(entry.target);
    }
  });
}, { threshold: .12 });
revealEls.forEach(el => io.observe(el));


// === AI 학습 타임라인 데이터(예시) ===
const AI_TIMELINE = [
  {
    date: '2025-09',
    title: 'Python 알고리즘 집중',
    detail: '그리디/투포인터/DP 기초, SWEA/백준 풀이 루틴 정착',
    link: 'https://github.com/Migong0311/ssafy-algorithm'
  },
  {
    date: '2025-10',
    title: 'Transfer Learning 실습',
    detail: 'ViT/ResNet 파이프라인, Confusion Matrix/ROC 리포트',
    link: '#'
  },
  {
    date: '2025-11',
    title: '모델 서빙 입문',
    detail: 'FastAPI 기반 추론 엔드포인트, 배포 스크립트 초안',
    link: '#'
  }
];

// === 타임라인 렌더링 ===
(function renderTimeline() {
  const wrap = document.getElementById('aiTimeline');
  if (!wrap) return;
  AI_TIMELINE.forEach((item, idx) => {
    const div = document.createElement('div');
    div.className = 'timeline-item reveal';
    div.style.setProperty('--delay', `${0.02 * idx}s`);
    div.innerHTML = `
      <div class="timeline-bullet"></div>
      <div class="small text-secondary">${item.date}</div>
      <div class="fw-semibold">${item.title}</div>
      <div class="small mb-1">${item.detail}</div>
      ${item.link ? `<a class="small" href="${item.link}" target="_blank" rel="noreferrer">관련 링크 →</a>` : ''}
    `;
    wrap.appendChild(div);
    io.observe(div);
  });
})();


// === 진행도 바(예시 값) ===
function setProgress(idBar, idLabel, val) {
  const bar = document.getElementById(idBar);
  const lab = document.getElementById(idLabel);
  if (!bar || !lab) return;
  const v = Math.max(0, Math.min(100, val));
  bar.style.width = v + '%';
  lab.textContent = v + '%';
}
setProgress('progAlgo', 'progAlgoLabel', 60);
setProgress('progDL', 'progDLLabel', 35);


// === 프로젝트 필터 ===
const grid = document.getElementById('projectGrid');
if (grid) {
  const btns = document.querySelectorAll('.filter-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const key = btn.dataset.filter;
      const items = grid.querySelectorAll('.project-item');
      items.forEach(it => {
        const tags = (it.getAttribute('data-tags') || '').split(',').map(s => s.trim());
        const on = key === 'all' || tags.includes(key);
        it.style.display = on ? '' : 'none';
      });
    });
  });
}


// === 내부 앵커 부드러운 스크롤 ===
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', (e) => {
    const id = a.getAttribute('href');
    const el = document.querySelector(id);
    if (el) {
      e.preventDefault();
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});


// === Hero right: 학습 진행도 원형 게이지 & 날짜 표시 ===
(function(){
  const nodes = document.querySelectorAll('.radial[data-progress]');
  nodes.forEach(node => {
    const val = Math.max(0, Math.min(100, Number(node.getAttribute('data-progress') || 0)));
    const deg = (val / 100) * 360;
    node.style.setProperty('--deg', deg + 'deg');
    const label = node.querySelector('.radial__label');
    if (label) label.textContent = val + '%';
  });
  const upd = document.getElementById('aiProgressUpdated');
  if (upd) {
    const d = new Date();
    const pad = n => String(n).padStart(2,'0');
    upd.textContent = `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}`;
  }
})();


// ==== Scroll Buttons (Top / Bottom) ====
(function(){
  const btnTop = document.getElementById('scrollTopBtn');
  const btnBottom = document.getElementById('scrollBottomBtn');
  if (!btnTop || !btnBottom) return;

  btnTop.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  btnBottom.addEventListener('click', () => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  });

  window.addEventListener('scroll', () => {
    const y = window.scrollY || document.documentElement.scrollTop;
    const show = y > 200;
    btnTop.classList.toggle('hide', !show);
    btnBottom.classList.toggle('hide', !show);
  });

  btnTop.classList.add('hide');
  btnBottom.classList.add('hide');
})();
