document.addEventListener('DOMContentLoaded', () => {
  const COPY_ICON = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false"><path fill="currentColor" d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1Zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Zm0 16H10V7h9v14Z"/></svg>';
  const COPIED_ICON = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false"><path fill="currentColor" d="M9 16.17 4.83 12 3.41 13.41 9 19l12-12-1.41-1.41z"/></svg>';

  document.querySelectorAll('.code-wrapper').forEach(wrapper => {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.type = 'button';
    btn.innerHTML = COPY_ICON;
    btn.setAttribute('aria-label', 'Copy code');
    btn.title = 'Copy code';
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
      btn.innerHTML = COPIED_ICON;
      btn.setAttribute('aria-label', 'Copied');
      btn.title = 'Copied';
      setTimeout(() => {
        btn.innerHTML = COPY_ICON;
        btn.setAttribute('aria-label', 'Copy code');
        btn.title = 'Copy code';
      }, 2000);
    });
    wrapper.appendChild(btn);  // append instead of prepend → goes after the code block in DOM
  });
});