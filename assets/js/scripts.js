document.addEventListener("DOMContentLoaded", function() {
      // Year Update
      document.getElementById('year').textContent = new Date().getFullYear();

      // Scroll Reveal Animation
      const reveals = document.querySelectorAll('.reveal');
      const revealOnScroll = () => {
        const windowHeight = window.innerHeight;
        const elementVisible = 100;
        reveals.forEach((reveal) => {
          const elementTop = reveal.getBoundingClientRect().top;
          if (elementTop < windowHeight - elementVisible) {
            reveal.classList.add('active');
          }
        });
      };
      window.addEventListener('scroll', revealOnScroll);
      revealOnScroll(); // Trigger once on load

      // Scroll Top/Bottom Buttons Logic
      const topBtn = document.getElementById("scrollTopBtn");
      const botBtn = document.getElementById("scrollBottomBtn");
      
      window.onscroll = function() {
        if (document.body.scrollTop > 300 || document.documentElement.scrollTop > 300) {
          topBtn.classList.add("show");
        } else {
          topBtn.classList.remove("show");
        }
      };
      
      topBtn.onclick = function() {
        window.scrollTo({top: 0, behavior: 'smooth'});
      };
      
      botBtn.onclick = function() {
        window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});
      };
    });