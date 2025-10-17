// 연도 자동 표기
document.getElementById("year").textContent = new Date().getFullYear();

// 데모 폼 메시지
const form = document.getElementById("contactForm");
form?.addEventListener("submit", (e) => {
  e.preventDefault();
  document.getElementById("formToast")?.classList.remove("d-none");
  form.reset();
});

// 🌙 다크모드 토글
const toggleBtn = document.getElementById("themeToggle");
const root = document.documentElement;

// 저장된 테마 불러오기
const savedTheme = localStorage.getItem("theme");
if (savedTheme) {
  root.setAttribute("data-theme", savedTheme);
  toggleBtn.textContent = savedTheme === "dark" ? "☀️" : "🌙";
}

// 버튼 클릭 시 전환
toggleBtn?.addEventListener("click", () => {
  const currentTheme = root.getAttribute("data-theme");
  const newTheme = currentTheme === "dark" ? "light" : "dark";
  root.setAttribute("data-theme", newTheme);
  toggleBtn.textContent = newTheme === "dark" ? "☀️" : "🌙";
  localStorage.setItem("theme", newTheme);
});
