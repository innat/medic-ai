# Home

<div class="medicai-lp">
  <section class="medicai-hero">
    <h1>medic<span>ai</span></h1>
    <p class="medicai-tagline">A <code>Keras</code>-native library for medical image analysis - backend-agnostic, GPU/TPU-ready, and built for volumetric data.</p>
    <div class="medicai-pill-row" aria-label="Core capabilities">
      <span class="medicai-pill teal">TensorFlow</span>
      <span class="medicai-pill coral">PyTorch</span>
      <span class="medicai-pill amber">JAX</span>
      <span class="medicai-pill">2D + 3D</span>
      <span class="medicai-pill">Multi-GPU / TPU</span>
    </div>
    <div class="medicai-btn-row">
      <a class="medicai-btn primary" href="getting_started/quickstart.html" data-doc-path="getting_started/quickstart.html">Get started</a>
      <a class="medicai-btn" href="guides/example.html" data-doc-path="guides/example.html">View examples</a>
      <a class="medicai-btn" href="https://github.com/innat/medic-ai">GitHub</a>
    </div>
  </section>

  <section class="medicai-section">
    <div class="medicai-section-label">core strengths</div>
    <div class="medicai-cards">
      <article class="medicai-card">
        <div class="medicai-card-icon teal" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path d="M4 7h5l2 2h9"/><path d="M4 17h5l2-2h9"/><path d="M4 12h16"/><path d="M17 6l3 3-3 3"/><path d="M17 12l3 3-3 3"/></svg>
        </div>
        <h3>Backend agnostic</h3>
        <p>Runs identically on TensorFlow, PyTorch, and JAX - switch with one environment variable.</p>
      </article>
      <article class="medicai-card">
        <div class="medicai-card-icon blue" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path d="M8 8 4 12l4 4"/><path d="m16 8 4 4-4 4"/><path d="m14 5-4 14"/></svg>
        </div>
        <h3>High-level API</h3>
        <p>Consistent interface for transforms and model creation with minimal boilerplate.</p>
      </article>
      <article class="medicai-card">
        <div class="medicai-card-icon amber" aria-hidden="true">
          <svg viewBox="0 0 24 24"><rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v3M15 2v3M9 19v3M15 19v3M2 9h3M2 15h3M19 9h3M19 15h3"/></svg>
        </div>
        <h3>Scalable execution</h3>
        <p>Train and infer on single GPU, multi-GPU, and TPU-VM setups out of the box.</p>
      </article>
      <article class="medicai-card">
        <div class="medicai-card-icon purple" aria-hidden="true">
          <svg viewBox="0 0 24 24"><rect x="4" y="4" width="6" height="6" rx="1"/><rect x="14" y="4" width="6" height="6" rx="1"/><rect x="4" y="14" width="6" height="6" rx="1"/><rect x="14" y="14" width="6" height="6" rx="1"/></svg>
        </div>
        <h3>Model Zoo</h3>
        <p>Supports state-of-the-art classification and segmentation models for both 2D and 3D workloads.</p>
      </article>
      <article class="medicai-card">
        <div class="medicai-card-icon coral" aria-hidden="true">
          <svg viewBox="0 0 24 24"><path d="M12 3v18"/><path d="M3 12h18"/><circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/></svg>
        </div>
        <h3>Utility</h3>
        <p>Includes medical-specialized losses, metrics, and components such as Grad-CAM utilities.</p>
      </article>
    </div>
  </section>

</div>

<script>
(() => {
  const landing = document.querySelector(".medicai-lp");
  if (!landing) return;

  const prefix = window.location.pathname.includes("/getting_started/") ? "../" : "";
  landing.querySelectorAll("[data-doc-path]").forEach((link) => {
    link.href = prefix + link.dataset.docPath;
  });

})();
</script>
