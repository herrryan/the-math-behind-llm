function renderKaTeX(el) {
  const root = el || document.body;
  if (typeof renderMathInElement !== "undefined" && root) {
    renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true }
      ],
      throwOnError: false
    });
  }
}

if (typeof document$ !== "undefined") {
  document$.subscribe(({ body }) => {
    renderKaTeX(body);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => renderKaTeX(document.body));
} else {
  renderKaTeX(document.body);
}
