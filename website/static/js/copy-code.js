document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.code-wrapper').forEach(wrapper => {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', () => {
      const code = wrapper.querySelector('pre');
      const text = code.innerText;
      if (navigator.clipboard) {
        navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      btn.textContent = 'Copied!';
      setTimeout(() => btn.textContent = 'Copy', 2000);
    });
    wrapper.appendChild(btn);  // append instead of prepend → goes after the code block in DOM
  });
});