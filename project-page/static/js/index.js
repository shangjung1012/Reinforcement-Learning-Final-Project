function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: "smooth"
  });
}

function copyBibTeX() {
  const bibtexElement = document.getElementById("bibtex-code");
  const button = document.querySelector(".copy-bibtex-btn");

  if (!bibtexElement || !button) {
    return;
  }

  const text = bibtexElement.textContent;
  const markCopied = () => {
    button.classList.add("copied");
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.classList.remove("copied");
      button.textContent = "Copy BibTeX";
    }, 1800);
  };

  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(markCopied).catch(() => {
      fallbackCopy(text);
      markCopied();
    });
  } else {
    fallbackCopy(text);
    markCopied();
  }
}

function fallbackCopy(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "");
  textArea.style.position = "absolute";
  textArea.style.left = "-9999px";
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand("copy");
  document.body.removeChild(textArea);
}

document.addEventListener("DOMContentLoaded", () => {
  const scrollButton = document.querySelector(".scroll-to-top");
  const copyButton = document.querySelector(".copy-bibtex-btn");

  if (scrollButton) {
    scrollButton.addEventListener("click", scrollToTop);
    window.addEventListener("scroll", () => {
      scrollButton.classList.toggle("visible", window.scrollY > 320);
    });
  }

  if (copyButton) {
    copyButton.addEventListener("click", copyBibTeX);
  }
});
