// Motion and small interactions. The page works the same without this file.
(() => {
  const calm = matchMedia("(prefers-reduced-motion: reduce)");
  const fine = matchMedia("(hover: hover) and (pointer: fine)");
  const glowing = ".card, .limit-card, .tally-item, .drop, .auth-card";
  const rippling = ".ghost, .chip:not(.plain), .cta, .account-link";
  const TILT = 5;
  const TOAST_TIME = 2600;
  const ICONS = {
    success: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
    error: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 6v8M12 18h.01"/></svg>',
  };
  document.documentElement.classList.add("js");

  const reveal = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      entry.target.classList.add("in");
      reveal.unobserve(entry.target);
    }
  }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
  document.querySelectorAll("[data-reveal]").forEach((node) => reveal.observe(node));

  const title = document.getElementById("heroTitle");
  const splitTitle = () => {
    const words = title.textContent.trim().split(/\s+/);
    title.replaceChildren(...words.flatMap((word, index) => {
      const span = document.createElement("span");
      span.className = "word";
      span.style.setProperty("--i", index);
      span.textContent = word;
      return index < words.length - 1 ? [span, document.createTextNode(" ")] : [span];
    }));
  };
  splitTitle();
  document.addEventListener("vi:language", splitTitle);

  const hero = document.querySelector(".hero.single");
  const progress = document.getElementById("progress");
  const toTop = document.getElementById("toTop");
  let queued = false;
  addEventListener("scroll", () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      const room = document.documentElement.scrollHeight - innerHeight;
      progress.style.transform = `scaleX(${room > 0 ? scrollY / room : 0})`;
      toTop.classList.toggle("show", scrollY > 600);
      if (!calm.matches) hero.style.setProperty("--shift", Math.min(scrollY, 900).toFixed(0));
      queued = false;
    });
  }, { passive: true });
  toTop.addEventListener("click", () => scrollTo({ top: 0, behavior: calm.matches ? "auto" : "smooth" }));

  document.addEventListener("pointermove", (event) => {
    if (!fine.matches) return;
    const target = event.target.closest?.(glowing);
    if (!target) return;
    const box = target.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width;
    const y = (event.clientY - box.top) / box.height;
    target.style.setProperty("--x", `${100 * x}%`);
    target.style.setProperty("--y", `${100 * y}%`);
    if (target.classList.contains("card") && !calm.matches) {
      target.style.setProperty("--rx", `${(0.5 - y) * TILT}deg`);
      target.style.setProperty("--ry", `${(x - 0.5) * TILT}deg`);
    }
  }, { passive: true });
  document.addEventListener("pointerout", (event) => {
    const card = event.target.closest?.(".card");
    if (card && !card.contains(event.relatedTarget)) {
      card.style.removeProperty("--rx");
      card.style.removeProperty("--ry");
    }
  });

  document.addEventListener("pointerdown", (event) => {
    const button = event.target.closest?.(rippling);
    if (!button || calm.matches) return;
    const box = button.getBoundingClientRect();
    const wave = document.createElement("span");
    wave.className = "ripple";
    wave.style.setProperty("--x", `${event.clientX - box.left}px`);
    wave.style.setProperty("--y", `${event.clientY - box.top}px`);
    wave.addEventListener("animationend", () => wave.remove(), { once: true });
    button.append(wave);
  });

  const dialog = document.getElementById("detail");
  const shut = dialog.close.bind(dialog);
  dialog.close = () => {
    if (calm.matches || !dialog.open) return shut();
    dialog.classList.add("closing");
    dialog.addEventListener("animationend", () => {
      dialog.classList.remove("closing");
      shut();
    }, { once: true });
  };

  const shelf = document.getElementById("toasts");
  window.toast = (text, kind = "") => {
    const note = document.createElement("div");
    note.className = `toast ${kind}`.trim();
    if (ICONS[kind]) {
      const icon = document.createElement("span");
      icon.className = "toast-icon";
      icon.innerHTML = ICONS[kind];
      note.append(icon);
    }
    note.append(document.createTextNode(text));
    shelf.append(note);
    setTimeout(() => {
      note.classList.add("out");
      note.addEventListener("animationend", () => note.remove(), { once: true });
    }, TOAST_TIME);
  };
})();
