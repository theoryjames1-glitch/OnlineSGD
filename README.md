# OnlineSGD: A Theory of Adaptive Gradient Descent in Non-Stationary Environments

## Abstract

We formalize **OnlineSGD** as *stochastic heavy-ball descent* whose **step size** and **inertia** are governed by real-time feedback from the stream of losses, rewards, and gradients. We present a set of **Laws of Adaptive Gradient Descent**—design constraints and invariants that any practical OnlineSGD controller should satisfy for stability, responsiveness, and tracking performance in non-stationary settings.

---

## 1) Setup & Notation

At time $t$ we observe a loss $\ell_t(\theta)$ (and optionally a reward $r_t$). Let

* $g_t \equiv \nabla \ell_t(\theta_t)$ (stochastic gradient),
* $v_{t+1} = \mu_t v_t + g_t$ (heavy-ball velocity),
* $\theta_{t+1} = \theta_t - \eta_t v_{t+1}$ (parameter update).

OnlineSGD supplies **controllers**

$$
\eta_t = f(\mathcal S_t),\qquad \mu_t = g(\mathcal S_t)
$$

from feedback signals $\mathcal S_t$ that include filtered losses/rewards/gradients and their trends:

$$
\bar L_t = \rho\bar L_{t-1} + (1-\rho)L_t,\quad
\bar R_t = \rho\bar R_{t-1} + (1-\rho)R_t,\quad
\bar g_t = \rho\bar g_{t-1} + (1-\rho)\lVert g_t\rVert,
$$

with robust deviations $\hat\sigma_L,\hat\sigma_R$ (e.g., EMA of $|L_t-\bar L_t|$, $|R_t-\bar R_t|$), and trends

$$
dL_t=\bar L_t-\bar L_{t-1},\quad dR_t=\bar R_t-\bar R_{t-1}.
$$

When $R_t$ is unavailable, we use the **reward fallback** $R_t:=-L_t$.

---

## 2) OnlineSGD Dynamics (canonical form)

$$
\boxed{
\begin{aligned}
v_{t+1} &= \mu_t v_t + g_t \\
\theta_{t+1} &= \theta_t - \eta_t v_{t+1} \\
\eta_t &= \mathrm{clip}\!\left(\eta_{t-1}\exp\{\alpha_R \tilde dR_t - \alpha_L \tilde dL_t\}\cdot \underbrace{\frac{1}{1+\kappa \,\bar g_t^{\,2}}}_{\text{trust}},\;\eta_{\min},\;\eta_{\max,t}^{\text{(cap)}}\right) \\
\mu_t  &= \mathrm{clip}\!\left(\mu_{t-1} + \beta\,\tanh\!\Big(\gamma\,\mathrm{SNR}_R(t) - \delta\,|\tilde dL_t|\Big),\;\mu_{\text{base}},\;\mu_{\max}\right)
\end{aligned}}
$$

where $\tilde dL_t := dL_t/(\hat\sigma_L+\varepsilon)$, $\tilde dR_t := dR_t/(\hat\sigma_R+\varepsilon)$, $\mathrm{SNR}_R(t):=|dR_t|/(\hat\sigma_R+\varepsilon)$, and $\eta_{\max,t}^{\text{(cap)}}$ enforces warm-up and (optional) smoothness-based safety (below).

This template admits simpler specializations (e.g., **bold-driver**: $\eta$ up on improvement, down on regression; EMA-smoothed with a tolerance band).

---

## 3) The Laws of Adaptive Gradient Descent

Think of these as **design laws**—monotonicities, caps, and invariants your controller must obey. They are orthogonal; you can implement them additively.

### Law 1 — Feedback (Progress Monotonicity)

**If progress improves, step size should not decrease; if it worsens, step size should not increase.**

$$
\frac{\partial \eta_t}{\partial(-\tilde dL_t)} \ge 0,\quad \frac{\partial \eta_t}{\partial(\tilde dL_t)} \le 0,\qquad
\frac{\partial \eta_t}{\partial(\tilde dR_t)} \ge 0.
$$

This codifies *“do more of what’s working.”*

### Law 2 — Trust-Region

**Steps shrink with gradient scale** (proxy for curvature or volatility):

$$
\eta_t \propto \frac{1}{1+\kappa \,\bar g_t^{\,2}},\qquad \kappa \ge 0.
$$

Prevents blow-ups when gradients are large/sudden.

### Law 3 — Stability Cap (Safety)

Under $L$-smoothness of the instantaneous loss, heavy-ball stability motivates

$$
\eta_t(1+\mu_t)\,L \le 2 - \epsilon \quad\Rightarrow\quad
\eta_t \le \eta_{\max}^{(\text{safe})} := \frac{2(1+\mu_{\max})-\epsilon}{\hat L},
$$

where $\hat L$ is a (possibly rough) upper bound on curvature. Enforce $\eta_t\le \min(\eta_{\max},\eta_{\max}^{(\text{safe})})$.

### Law 4 — Warm-Up

**No cold starts.** For $t\le T_\text{warm}$,

$$
\eta_t \le \eta_{\min} + \frac{t}{T_\text{warm}}\big(\eta_{\max} - \eta_{\min}\big).
$$

This keeps early updates gentle while statistics stabilize.

### Law 5 — Inertia (Momentum)

**Inertia tracks the reliability of progress.** Increase $\mu_t$ when progress is consistent; otherwise relax toward a floor:

$$
\mu_t \uparrow \text{ with } \mathrm{SNR}_R(t),\qquad
\mu_t \downarrow \text{ with } |\tilde dL_t|,\qquad \mu_t \ge \mu_{\text{base}}.
$$

### Law 6 — Conservation of Stability (Backoff)

**On sudden regression (loss spike), shrink $\eta$ multiplicatively and cool down.**
If $L_t > (1+\tau)\bar L_{t-1}$ for a small $\tau>0$, then

$$
\eta_t \leftarrow \lambda\,\eta_t,\quad \lambda\in(0,1), \quad \text{and block increases for a short cooldown.}
$$

This is the “circuit breaker” that prevents runaway steps.

### Law 7 — Non-Myopic Sensing (Smoothing)

Decisions are made on **filtered** signals; set time constants so $\bar L_t,\bar R_t,\bar g_t$ are *responsive but not twitchy*:

$$
\rho\in[0.9,0.99] \ \Rightarrow\ 10\!-\!100\text{ steps half-life.}
$$

Optional tolerance band: change $\eta$ only if the EMA moves by more than $1\!\sim\!3\%$.

### Law 8 — Decoupled Regularization

Weight decay acts multiplicatively on parameters **independently** of the gradient direction:

$$
\theta \leftarrow (1-\eta_t \lambda_{\text{wd}})\,\theta - \eta_t v_{t+1}.
$$

Stops the regularizer from being confounded with adaptive step length.

### Law 9 — Reward Fallback Consistency

When explicit reward is absent, set $R_t := -L_t$. Then the **signs** of progress are consistent across loss and reward channels—crucial in RL-flavored training loops.

### Law 10 — Bounded Reactivity (Change Budget)

**Log-step changes are bounded per step.**

$$
\left|\log\frac{\eta_t}{\eta_{t-1}}\right|\le c_\eta,\qquad
|\mu_t - \mu_{t-1}| \le c_\mu.
$$

Avoids over-steering.

### Law 11 — Exploration/Exploitation Dial

When reward improves (or uncertainty decreases), allow larger $\eta_t$ (explore). When uncertainty rises, keep $\eta_t$ moderate but **raise $\mu_t$** to harvest the current direction (exploit) only if progress is consistent; otherwise reduce $\mu_t$ for agility.

### Law 12 — Offload Invariance (System Design)

Controller math and optimizer state may live off-device (e.g., CPU offload) **without changing the laws**. Only the *measurement* of $\bar g_t$ differs (e.g., norm computed on CPU vs GPU). The invariants above are device-agnostic.

---

## 4) Guarantees (sketches)

### 4.1 Stability under smoothness (heavy-ball region)

Assume each $\ell_t$ is $L_t$-smooth and $L_t\le \hat L$. If Law 2 & 3 enforce

$$
\eta_t(1+\mu_t)\hat L \le 2 - \epsilon,\qquad \forall t,
$$

then the **update map is non-expansive** around critical points; oscillations are damped by momentum and trust. (Classical heavy-ball stability adapted online.)

### 4.2 Tracking regret in slowly drifting objectives

If the optimal point $\theta_t^\star$ drifts with bounded variation $\sum_{t=1}^{T-1}\lVert \theta_{t+1}^\star-\theta_t^\star\rVert \le B$ and signals satisfy Laws 1–2–7–10, OnlineSGD achieves **dynamic (tracking) regret**

$$
\sum_{t=1}^T \big(\ell_t(\theta_t)-\ell_t(\theta_t^\star)\big)
= \tilde{\mathcal O}\!\left(\frac{1}{\eta_{\min}} + B + \sum_{t}\kappa \bar g_t^{\,2}\right),
$$

matching online gradient-style rates up to adaptive terms (sketch; constants depend on Lipschitzness and noise).

### 4.3 Backoff recovery

Under Law 6, if a spike triggers $\eta_t \leftarrow \lambda\eta_t$ with cooldown $C$, then over any window shorter than $C$, $\eta$ cannot increase, which **bounds local overshoot energy**:

$$
\sum_{i=0}^{C-1}\eta_{t+i}\lVert v_{t+i+1}\rVert^2 \le \lambda \sum_{i=0}^{C-1}\eta_{t-1}\lVert v_{t+i+1}\rVert^2.
$$

---

## 5) Algorithm (law-compliant skeleton)

```text
Inputs: ηmin, ηmax, μbase, μmax, κ, ρ (EMAs), tol, clip cη, backoff (λ, cooldown C), warmup T
State: η←η0, μ←μbase, v←0,  ēL, ēR, ēg, σL, σR, cooldown←0

for t = 0,1,2,...
  observe (optional) reward Rt and loss Lt; fallback Rt := -Lt if missing
  update EMAs: ēL, ēR, σL, σR, ēg
  trends: dL ← ēL - ēL_prev; dR ← ēR - ēR_prev
  normalized: ẟL ← dL/(σL+ε); ẟR ← dR/(σR+ε)

  # Law 1 + 7 + 10: η controller (log domain, with tolerance & clip)
  Δη_log ← αR·ẟR - αL·ẟL
  if cooldown>0: Δη_log ← min(Δη_log, 0)  # no increases in cooldown (Law 6)
  Δη_log ← clip(Δη_log, -cη, +cη)
  η_raw ← η · exp(Δη_log)

  # Law 2: trust region
  η_raw ← η_raw / (1 + κ·ēg^2)

  # Law 4 + 3: warmup + safety cap
  η_cap ← warmup_cap(ηmax, t, T)
  if have L̂: η_cap ← min(η_cap, (2(1+μmax)-ε)/L̂)
  η ← clip(η_raw, ηmin, η_cap)

  # Law 5: μ controller
  arg ← γ·SNR_R(=|dR|/(σR+ε)) - δ·|ẟL|
  μ ← clip( μ + β·tanh(arg), μbase, μmax )

  # gradient & update
  v ← μ·v + gt
  θ ← θ - η·v

  # Law 6: backoff on spike
  if Lt > (1+τ)·ēL_prev:
     η ← λ·η; μ ← max(μ, μbase); cooldown ← C

  cooldown ← max(0, cooldown-1)
end
```

This blueprint satisfies **every law** above (with obvious specializations like bold-driver in place of the exponential rule).

---

## 6) Practical Defaults & Tuning

* **EMAs**: $\rho \in [0.95,0.99]$ for loss/grad; smaller $\rho$ if data is very non-stationary.
* **Gains**: $\alpha_L \in [0.3,1.0]$, $\alpha_R \in [0,0.5]$.
* **Trust**: $\kappa \in [10^{-2}, 10^{-1}]$ when gradients are spiky; $0$ for apples-to-apples vs SGD.
* **Clip**: $c_\eta \in [0.1,0.3]$ (≈ ±10–35% per step).
* **Backoff**: $\tau \in [0.02,0.05]$, $\lambda \in [0.3,0.7]$, $C \in [3,10]$.
* **Momentum**: $\mu_{\text{base}}\in[0.05,0.2],\; \mu_{\max}\in[0.8,0.95],\; \beta\in[0.02,0.1]$.

---

## 7) What to Log (verifiable invariants)

* **Law-checks**: sign(Δη) vs sign of $-\tilde dL$; $\eta$ never exceeds warm-up/safety caps; momentum never below $\mu_{\text{base}}$.
* **Trust activity**: ratio $\eta_{\text{post-trust}}/\eta_{\text{pre-trust}}$.
* **Backoff events**: spikes, new $\eta$, cooldown remaining.
* **SNRs**: $|dR|/(\hat\sigma_R+\varepsilon)$, $|dL|/(\hat\sigma_L+\varepsilon)$.

---

## 8) Takeaways

* **OnlineSGD = SGD + Controller**: gradients move you; **laws** decide *how far* and *how inertial* each step is.
* The **Laws of Adaptive GD** ensure (i) **responsiveness** to improvements, (ii) **stability** under noise and spikes, (iii) **bounded reactivity**, and (iv) **device-agnostic implementation** (CPU offload included).
* In stationary limits, OnlineSGD recovers classical SGD/Heavy-Ball behavior; in drifting tasks, the controller **tracks** moving optima with bounded regret and robust stability.


### PSEUDOCODE

```python
# online_sgd.py
import math
from typing import Iterable, Optional, Dict, Any
import torch
from torch.optim import Optimizer

class OnlineSGD(Optimizer):
    """
    OnlineSGD: EMA-based bold-driver + trust-region, with optional CPU offload.

    Key features:
      - Step-size (eta) via bold-driver on loss:
          early steps: raw loss direction
          later steps: loss EMA with tolerance band
        + trust shrink: eta <- eta / (1 + kappa * ||g||^2)
        + warmup cap and optional safety cap (L_hat)
      - Momentum with floor: up on improvements, down toward floor on regressions
      - Decoupled weight decay
      - Maximize=True support
      - NEW: offload_to_cpu=True keeps all optimizer state and math on CPU
        (no optimizer VRAM). Model params remain on their device; after each
        step, updated CPU "master params" are copied back to model tensors.

    Usage (offload):
        opt = OnlineSGD(model.parameters(),
                        offload_to_cpu=True,     # <-- no optimizer VRAM
                        lr=5e-2, eta_min=1e-4, eta_max=2e-1,
                        mu_max=0.9, mu_base=0.10,
                        kappa=0.0,               # trust off to avoid grad-norm work on GPU
                        eta_log_step_clip=0.2,
                        warmup_steps=20)
        loss.backward()
        opt.step(metrics={'loss': loss.item()})
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        *,
        eta_min: float = 1e-5,
        eta_max: float = 2e-1,
        mu_max: float = 0.95,

        # Trust region
        kappa: float = 0.02,              # eta <- eta / (1 + kappa * ||g||^2)

        # Step-size change limiter
        eta_log_step_clip: float = 0.20,  # per-step bound in log-domain

        # Momentum
        mu_base: float = 0.10,
        beta: float = 0.03,

        # Smoothing / decision tolerance
        rho_grad: float = 0.95,      # EMA for grad norm
        rho_loss: float = 0.90,      # EMA for loss
        loss_tol_pct: float = 0.02,  # tolerance band for EMA decisions
        up_rate: float = 0.10,       # +10% on improvement
        down_rate: float = 0.10,     # -10% on worsening

        # Warmup & safety
        warmup_steps: int = 20,
        L_hat: Optional[float] = None,
        safety_eps: float = 0.1,

        # Reward fallback (not used for eta decisions)
        use_loss_as_reward_if_missing: bool = True,

        # Offload optimizer state & math to CPU (no optimizer VRAM)
        offload_to_cpu: bool = False,

        # Regular optimizer options
        weight_decay: float = 0.0,
        maximize: bool = False
    ):
        if not 0.0 < eta_min <= lr <= eta_max:
            raise ValueError("Require 0 < eta_min <= lr <= eta_max.")
        if not 0.0 <= mu_max < 1.0:
            raise ValueError("Require 0 <= mu_max < 1.")
        if not 0.0 <= mu_base < 1.0:
            raise ValueError("Require 0 <= mu_base < 1.")
        if not 0.0 < rho_grad < 1.0 or not 0.0 < rho_loss < 1.0:
            raise ValueError("Require 0 < rho_grad, rho_loss < 1.")
        if eta_log_step_clip <= 0 or warmup_steps < 0:
            raise ValueError("Bad clip or warmup.")

        defaults = dict(lr=lr, weight_decay=weight_decay, maximize=maximize)
        super().__init__(params, defaults)

        # State scalars
        self._eta = lr
        self._eta_min = eta_min
        self._eta_max_user = eta_max
        self._mu = mu_base
        self._mu_base = mu_base
        self._mu_max = mu_max

        self._kappa = kappa
        self._eta_log_step_clip = eta_log_step_clip
        self._beta = beta

        self._rho_grad = rho_grad
        self._rho_loss = rho_loss
        self._loss_tol_pct = loss_tol_pct
        self._up_rate = up_rate
        self._down_rate = down_rate

        self._warmup_steps = warmup_steps
        self._t = 0
        self._L_hat = L_hat
        self._safety_eps = safety_eps
        self._eta_max_safe = self._eta_max_user
        if self._L_hat is not None and self._L_hat > 0:
            cap = (2.0 * (1.0 + self._mu_max) - self._safety_eps) / self._L_hat
            self._eta_max_safe = min(self._eta_max_user, cap)

        self._use_loss_as_reward_if_missing = use_loss_as_reward_if_missing

        # Stats
        self._bar_g = 0.0
        self._ema_loss = None
        self._prev_ema_loss = None
        self._prev_loss = None
        self._obs_count = 0
        self._last_reward = None

        # Offload flag
        self._offload = bool(offload_to_cpu)

    # ---------- utilities ----------
    @torch.no_grad()
    def _global_grad_norm_gpu(self) -> float:
        # Memory-friendly GPU norm: dot avoids pow() temporaries
        sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    v = g.coalesce().values()
                    sq += float(torch.dot(v.view(-1), v.view(-1)).item())
                else:
                    v = g.view(-1)
                    sq += float(torch.dot(v, v).item())
        return math.sqrt(sq)

    @torch.no_grad()
    def _global_grad_norm_cpu(self) -> float:
        # Copy grads to CPU first, then compute dot; avoids any extra GPU allocs
        sq = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    v = g.coalesce().values().to('cpu', non_blocking=True)
                    sq += float(torch.dot(v.view(-1), v.view(-1)).item())
                else:
                    v = g.detach().to('cpu', non_blocking=True).view(-1)
                    sq += float(torch.dot(v, v).item())
        return math.sqrt(sq)

    @torch.no_grad()
    def _clip_ratio(self, ratio: float) -> float:
        lo = math.exp(-self._eta_log_step_clip)
        hi = math.exp(+self._eta_log_step_clip)
        return float(min(max(ratio, lo), hi))

    @torch.no_grad()
    def _apply_warmup_cap(self, eta_hi: float) -> float:
        if self._warmup_steps <= 0:
            return eta_hi
        frac = min(1.0, (self._t + 1) / float(self._warmup_steps))
        return min(eta_hi, self._eta_min + frac * (eta_hi - self._eta_min))

    # ---------- controller ----------
    @torch.no_grad()
    def _update_controller(self, grad_norm: float, metrics: Optional[Dict[str, Any]]):
        # Read loss
        L_t = float(metrics.get('loss', 0.0)) if metrics else 0.0
        if metrics:
            if 'reward' in metrics:
                self._last_reward = float(metrics['reward'])
            elif self._use_loss_as_reward_if_missing:
                self._last_reward = -L_t

        # Update loss EMA
        if self._ema_loss is None:
            self._ema_loss = L_t
            self._prev_ema_loss = L_t
        else:
            self._prev_ema_loss = self._ema_loss
            self._ema_loss = self._rho_loss * self._ema_loss + (1.0 - self._rho_loss) * L_t

        # Early steps: raw direction; later: EMA with tolerance
        if self._obs_count < 2 and self._prev_loss is not None:
            improve = (L_t < self._prev_loss)
            worsen  = (L_t > self._prev_loss)
        else:
            improve = (self._ema_loss < self._prev_ema_loss * (1.0 - self._loss_tol_pct))
            worsen  = (self._ema_loss > self._prev_ema_loss * (1.0 + self._loss_tol_pct))

        # Bold-driver eta
        ratio = 1.0 + self._up_rate if improve else (1.0 - self._down_rate if worsen else 1.0)
        ratio = self._clip_ratio(ratio)
        eta_raw = self._eta * ratio

        # Trust shrink (quadratic in grad norm EMA)
        eta_raw = eta_raw / (1.0 + self._kappa * (max(self._bar_g, 0.0) ** 2))

        # Final caps
        eta_hi = self._apply_warmup_cap(self._eta_max_safe)
        self._eta = float(min(max(eta_raw, self._eta_min), eta_hi))

        # Momentum toward {1 or floor}
        if improve:
            self._mu += self._beta * (1.0 - self._mu)
        elif worsen:
            self._mu -= self._beta * (self._mu - self._mu_base)
        self._mu = float(min(max(self._mu, self._mu_base), self._mu_max))

        # Reflect eta to param groups (for logging/tests)
        for group in self.param_groups:
            group['lr'] = self._eta

        # Step bookkeeping
        self._t += 1
        self._prev_loss = L_t
        self._obs_count += 1
        # Update grad EMA *after* using previous value
        self._bar_g = self._rho_grad * self._bar_g + (1.0 - self._rho_grad) * grad_norm

    # ---------- optimizer API ----------
    @torch.no_grad()
    def step(self, closure=None, *, metrics: Optional[Dict[str, Any]] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Grad norm measurement: avoid GPU allocs if offloading or kappa=0
        if self._kappa <= 0.0:
            grad_norm = 0.0
        else:
            grad_norm = self._global_grad_norm_cpu() if self._offload else self._global_grad_norm_gpu()

        # Controller update (on CPU scalars)
        self._update_controller(grad_norm, metrics)

        # Parameter updates
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            ascend = group['maximize']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if self._offload:
                    # --- Offload path: keep everything on CPU and copy back ---
                    # Master param on CPU
                    if 'master_param' not in state:
                        state['master_param'] = p.detach().to('cpu').clone().requires_grad_(False)
                    mp = state['master_param']

                    # CPU velocity buffer
                    if 'velocity_cpu' not in state:
                        state['velocity_cpu'] = torch.zeros_like(mp, memory_format=torch.preserve_format)
                    vcpu = state['velocity_cpu']

                    # Gradient to CPU
                    g = p.grad
                    if ascend:
                        g = -g
                    if g.is_sparse:
                        # densify sparse grads on CPU for simplicity
                        g_dense = torch.zeros_like(mp)
                        idx = g.coalesce().indices()
                        vals = g.coalesce().values().to('cpu', non_blocking=True)
                        # scatter_add_ for 1D flattened master
                        flat = g_dense.view(-1)
                        flat.index_add_(0, idx.view(-1).to('cpu'), vals)
                        g_cpu = flat.view_as(mp)
                    else:
                        g_cpu = g.detach().to('cpu', non_blocking=True).to(mp.dtype)
                        if g_cpu.shape != mp.shape:
                            g_cpu = g_cpu.view_as(mp)

                    # Decoupled weight decay on CPU
                    if wd != 0.0:
                        mp.mul_(1.0 - lr * wd)

                    # Heavy-ball on CPU
                    vcpu.mul_(self._mu).add_(g_cpu)   # v = mu * v + grad
                    mp.add_(vcpu, alpha=-lr)          # theta = theta - eta * v

                    # Copy updated master back to model device (in-place)
                    p.data.copy_(mp.to(p.data.device, non_blocking=True))

                else:
                    # --- Regular GPU/CPU-in-place path ---
                    g = p.grad
                    if ascend:
                        g = -g

                    # Decoupled weight decay
                    if wd != 0.0:
                        p.data.mul_(1.0 - lr * wd)

                    # Velocity buffer colocated with param
                    if 'velocity' not in state:
                        state['velocity'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    v = state['velocity']

                    v.mul_(self._mu).add_(g)   # v = mu * v + grad
                    p.add_(v, alpha=-lr)       # theta = theta - eta * v

        return loss

    # ---------- accessors ----------
    @property
    def eta(self) -> float:
        return self._eta

    @property
    def mu(self) -> float:
        return self._mu

    @property
    def eta_max_safe(self) -> float:
        return self._eta_max_safe

    def controller_state(self) -> Dict[str, float]:
        return {
            'eta': float(self._eta),
            'mu': float(self._mu),
            'ema_loss': float('nan') if self._ema_loss is None else float(self._ema_loss),
            'bar_g': float(self._bar_g),
            'eta_max_safe': float(self._eta_max_safe),
            't': float(self._t),
            'obs_count': float(self._obs_count),
            'offload_to_cpu': float(1.0 if self._offload else 0.0),
        }
```

### TEST

```python
# tests/test_onlinesgd_refactored.py
import math
import torch
import pytest

from OnlineSGD import OnlineSGD

torch.manual_seed(0)

def make_param(init=0.0):
    return torch.nn.Parameter(torch.tensor([float(init)], dtype=torch.float32), requires_grad=True)

def step_once(opt, p, grad_value, loss_value):
    p.grad = torch.tensor([grad_value], dtype=torch.float32)
    opt.step(metrics={'loss': float(loss_value)})

# ----------------------------
# 1) Bold-driver: η goes up if loss decreases; down if increases
# ----------------------------
def test_eta_moves_with_loss_direction():
    p = make_param(0.0)
    opt = OnlineSGD([p], lr=1e-2, eta_min=1e-5, eta_max=1.0, mu_max=0.9,
                    kappa=0.0, eta_log_step_clip=1.0, warmup_steps=0, mu_base=0.10)

    eta0 = opt.eta
    # first call sets prev_loss; no change expected
    step_once(opt, p, grad_value=0.0, loss_value=1.0)
    eta1 = opt.eta
    assert math.isclose(eta1, eta0, rel_tol=1e-12)

    # loss decreases -> eta should increase
    step_once(opt, p, grad_value=0.0, loss_value=0.9)
    eta_up = opt.eta
    assert eta_up > eta1

    # loss increases -> eta should decrease
    step_once(opt, p, grad_value=0.0, loss_value=1.2)
    eta_down = opt.eta
    assert eta_down < eta_up

# ----------------------------
# 2) Trust region shrink: larger grad -> smaller η (holding loss flat)
# ----------------------------
def test_trust_region_shrink_depends_on_grad_norm():
    def build():
        q = make_param(0.0)
        o = OnlineSGD([q], lr=1e-1, eta_min=1e-5, eta_max=1.0, mu_max=0.9,
                      kappa=0.1, eta_log_step_clip=1.0, warmup_steps=0, mu_base=0.10)
        return q, o

    # Initialize both with same prev loss
    p_small, opt_small = build()
    p_large, opt_large = build()

    step_once(opt_small, p_small, grad_value=0.0, loss_value=1.0)
    step_once(opt_large, p_large, grad_value=0.0, loss_value=1.0)

    # Step to populate grad EMA differently
    step_once(opt_small, p_small, grad_value=0.1, loss_value=1.0)  # tiny grad
    step_once(opt_large, p_large, grad_value=10.0, loss_value=1.0) # big grad

    # Now hold loss flat; only trust shrink should act (uses previous step's grad EMA)
    eta_before_small = opt_small.eta
    eta_before_large = opt_large.eta
    step_once(opt_small, p_small, grad_value=0.0, loss_value=1.0)
    step_once(opt_large, p_large, grad_value=0.0, loss_value=1.0)
    assert opt_small.eta > opt_large.eta, "η should be smaller under larger prior grad"
    # sanity: both should have non-increasing η here (trust shrink only)
    assert opt_small.eta <= eta_before_small + 1e-12
    assert opt_large.eta <= eta_before_large + 1e-12

# ----------------------------
# 3) Warmup: η should not exceed the warmup cap
# ----------------------------
def test_warmup_caps_eta_growth():
    p = make_param(0.0)
    eta_min, eta_max = 1e-4, 5e-1
    warmup_steps = 10
    opt = OnlineSGD([p], lr=1e-3, eta_min=eta_min, eta_max=eta_max, mu_max=0.9,
                    kappa=0.0, eta_log_step_clip=5.0, warmup_steps=warmup_steps, mu_base=0.10)

    # Drive decreasing losses to try to raise η
    step_once(opt, p, grad_value=0.0, loss_value=10.0)  # init
    for L in [9.0, 8.0]:
        step_once(opt, p, grad_value=0.0, loss_value=L)

    # Warmup cap after current step
    frac = min(1.0, (opt.controller_state()['t']) / float(warmup_steps))
    warmup_cap = eta_min + frac * (opt.eta_max_safe - eta_min)
    assert opt.eta <= warmup_cap + 1e-12

# ----------------------------
# 4) Safety cap from L_hat is enforced
# ----------------------------
def test_eta_max_safe_from_Lhat_is_enforced():
    p = make_param(0.0)
    L_hat = 1000.0
    mu_max = 0.9
    safety_eps = 0.1
    user_eta_max = 1.0
    expected_cap = (2.0 * (1.0 + mu_max) - safety_eps) / L_hat

    opt = OnlineSGD([p], lr=1e-3, eta_min=1e-6, eta_max=user_eta_max, mu_max=mu_max,
                    kappa=0.0, eta_log_step_clip=5.0, warmup_steps=0, mu_base=0.10,
                    L_hat=L_hat, safety_eps=safety_eps)

    assert math.isclose(opt.eta_max_safe, min(user_eta_max, expected_cap), rel_tol=1e-6)

    # Many decreases -> eta should not exceed eta_max_safe
    step_once(opt, p, grad_value=0.0, loss_value=10.0)
    for L in [9.0, 8.0, 7.0, 6.0, 5.0, 4.0]:
        step_once(opt, p, grad_value=0.0, loss_value=L)
    assert opt.eta <= opt.eta_max_safe + 1e-12

# ----------------------------
# 5) Decoupled weight decay scales parameters multiplicatively
# ----------------------------
def test_weight_decay_decoupled_scaling():
    p = make_param(1.0)
    lr = 0.1
    wd = 0.01
    opt = OnlineSGD([p], lr=lr, eta_min=1e-4, eta_max=1.0, mu_max=0.9,
                    kappa=0.0, eta_log_step_clip=1.0, warmup_steps=0, mu_base=0.10,
                    weight_decay=wd)

    # First step: grad zero, prev_loss init, only WD acts with lr=initial eta
    step_once(opt, p, grad_value=0.0, loss_value=1.0)
    expected = 1.0 * (1.0 - lr * wd)
    assert torch.allclose(p.data, torch.tensor([expected]), atol=1e-8)

# ----------------------------
# 6) maximize=True ascends along the gradient
# ----------------------------
def test_maximize_true_moves_up_gradient():
    p = make_param(0.0)
    lr = 0.05
    opt = OnlineSGD([p], lr=lr, eta_min=1e-4, eta_max=1.0, mu_max=0.9,
                    kappa=0.0, eta_log_step_clip=1.0, warmup_steps=0, mu_base=0.10,
                    maximize=True)

    # grad=+1 -> with maximize=True, parameter should move +lr (modulo momentum)
    step_once(opt, p, grad_value=1.0, loss_value=1.0)  # sets prev_loss
    assert p.item() > 0.0

# ----------------------------
# 7) Momentum floor and adjust: up on improvement, down toward floor on worsening
# ----------------------------
def test_momentum_floor_and_adjust():
    p = make_param(0.0)
    opt = OnlineSGD([p], lr=1e-2, eta_min=1e-5, eta_max=1.0, mu_max=0.9,
                    kappa=0.0, eta_log_step_clip=1.0, warmup_steps=0, mu_base=0.10, beta=0.2)

    mu0 = opt.mu
    # init
    step_once(opt, p, grad_value=0.0, loss_value=1.0)
    assert math.isclose(opt.mu, mu0, rel_tol=1e-12)

    # improvement -> mu increases
    step_once(opt, p, grad_value=0.0, loss_value=0.9)
    mu_up = opt.mu
    assert mu_up > mu0

    # worsening -> mu decreases toward floor but not below
    step_once(opt, p, grad_value=0.0, loss_value=1.2)
    mu_down = opt.mu
    assert mu0 <= mu_down <= mu_up

# ----------------------------
# 8) Integration smoke: average loss over epoch does not blow up
# ----------------------------
def test_integration_regression_smoke_avg_epoch():
    # small linear regression; check average loss per epoch
    X = torch.randn(256, 4)
    w_true = torch.randn(4, 1)
    y = X @ w_true + 0.05 * torch.randn(256, 1)

    model = torch.nn.Sequential(torch.nn.Linear(4, 1))
    crit = torch.nn.MSELoss()

    opt = OnlineSGD(
        model.parameters(),
        lr=5e-2, eta_min=1e-4, eta_max=2e-1, mu_max=0.9,
        kappa=0.02, eta_log_step_clip=0.2, warmup_steps=50, mu_base=0.10
    )

    def run_epoch():
        perm = torch.randperm(X.size(0))
        losses = []
        for i in range(0, X.size(0), 32):
            idx = perm[i:i+32]
            xb, yb = X[idx], y[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step(metrics={'loss': float(loss.item())})
            losses.append(float(loss.item()))
        return sum(losses) / len(losses)

    L0 = run_epoch()
    L1 = run_epoch()

    # Averaged criterion for stability (more robust than last-batch)
    assert L1 <= L0 * 1.05, f"Expected average epoch loss not to blow up; got L0={L0:.4f}, L1={L1:.4f}"
```

### Compare OnlineSGD vs SGD

```python
# tests/test_compare_online_vs_sgd.py
import math
import time
import statistics
import torch
import pytest

from OnlineSGD import OnlineSGD

# -------------------------------
# Helpers
# -------------------------------

def set_deterministic(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_regression(n=512, d=8, noise=0.05, device="cpu"):
    X = torch.randn(n, d, device=device)
    w_true = torch.randn(d, 1, device=device)
    y = X @ w_true + noise * torch.randn(n, 1, device=device)
    return X, y

def make_classification(n=512, d=8, margin=2.0, device="cpu"):
    X = torch.randn(n, d, device=device)
    w_true = torch.randn(d, 1, device=device)
    logits = (X @ w_true) / math.sqrt(d)
    y = (logits + 0.1 * torch.randn_like(logits) > 0).float()
    X += margin * (y - 0.5)  # mild separation
    return X, y

def make_linreg_model(d=8, device="cpu"):
    return torch.nn.Sequential(torch.nn.Linear(d, 1)).to(device)

def make_logreg_model(d=8, device="cpu"):
    return torch.nn.Linear(d, 1).to(device)

def epoch_loop(model, opt, X, Y, loss_fn, batch=64, epochs=5, is_online=False):
    n = X.size(0)
    history = []
    for _ in range(epochs):
        # fixed order (no shuffle) for determinism
        for i in range(0, n, batch):
            xb, yb = X[i:i+batch], Y[i:i+batch]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            if is_online:
                opt.step(metrics={'loss': float(loss.item())})
            else:
                opt.step()
            history.append(float(loss.item()))
    # average of last epoch
    steps_per_epoch = (n // batch)
    last_epoch = history[-steps_per_epoch:]
    return history, sum(last_epoch) / len(last_epoch)

def smoothed_steps_to_threshold(history, threshold, window=3):
    buf = []
    for idx, v in enumerate(history, start=1):
        buf.append(v)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) == window and sum(buf)/len(buf) <= threshold:
            return idx
    return None

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def peak_memory_bytes_during(fn):
    """CUDA: exact via torch; CPU: psutil RSS sampling (best-effort)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        cuda_sync()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        start = torch.cuda.memory_allocated(dev)
        fn()
        cuda_sync()
        peak = torch.cuda.max_memory_allocated(dev)
        return max(0, peak - start)
    else:
        try:
            import psutil, os, threading
        except Exception:
            pytest.skip("psutil not available; skipping CPU memory benchmark.")
        p = psutil.Process(os.getpid())
        base = p.memory_info().rss
        peak = base
        done = False
        def wrapper():
            nonlocal done
            fn()
            done = True
        t = threading.Thread(target=wrapper)
        t.start()
        while not done and t.is_alive():
            rss = p.memory_info().rss
            if rss > peak:
                peak = rss
            time.sleep(0.002)
        t.join()
        return max(0, peak - base)

# -------------------------------
# Configs
# -------------------------------

# Apples-to-apples with SGD (no trust term). Keep warmup light.
ONLINE_QUALITY_CFG = dict(
    lr=5e-2, eta_min=1e-4, eta_max=2e-1,
    mu_max=0.9, mu_base=0.10,
    kappa=0.0,                 # trust OFF to match SGD capability
    eta_log_step_clip=0.2,
    warmup_steps=10,
    # gentle EMA bold-driver
    rho_loss=0.90, loss_tol_pct=0.02,
    up_rate=0.10, down_rate=0.10
)

ONLINE_MEM_TIME_CFG = {**ONLINE_QUALITY_CFG, "kappa": 0.0}  # explicit

SGD_CFG = dict(lr=5e-2, momentum=0.9)

# -------------------------------
# 1) Quality on Regression (final loss)
# -------------------------------

@pytest.mark.parametrize("device", ["cuda" if torch.cuda.is_available() else "cpu"])
def test_quality_regression_final_loss(device):
    set_deterministic(0)
    X, Y = make_regression(device=device)
    model_o = make_linreg_model(d=X.shape[1], device=device)
    opt_o = OnlineSGD(model_o.parameters(), **ONLINE_QUALITY_CFG)
    _, loss_o = epoch_loop(model_o, opt_o, X, Y, torch.nn.MSELoss(), epochs=5, is_online=True)

    set_deterministic(0)
    X2, Y2 = make_regression(device=device)  # same data
    model_s = make_linreg_model(d=X2.shape[1], device=device)
    opt_s = torch.optim.SGD(model_s.parameters(), **SGD_CFG)
    _, loss_s = epoch_loop(model_s, opt_s, X2, Y2, torch.nn.MSELoss(), epochs=5, is_online=False)

    print(f"[{device}] Regression final avg loss: Online={loss_o:.6f} | SGD={loss_s:.6f}")
    assert loss_o <= loss_s * 1.10

# -------------------------------
# 2) Quality on Classification (final loss)
# -------------------------------

@pytest.mark.parametrize("device", ["cuda" if torch.cuda.is_available() else "cpu"])
def test_quality_classification_final_loss(device):
    set_deterministic(1)
    X, Y = make_classification(device=device)
    model_o = make_logreg_model(d=X.shape[1], device=device)
    opt_o = OnlineSGD(model_o.parameters(), **ONLINE_QUALITY_CFG)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    _, loss_o = epoch_loop(model_o, opt_o, X, Y, loss_fn, epochs=5, is_online=True)

    set_deterministic(1)
    X2, Y2 = make_classification(device=device)
    model_s = make_logreg_model(d=X2.shape[1], device=device)
    opt_s = torch.optim.SGD(model_s.parameters(), **SGD_CFG)
    _, loss_s = epoch_loop(model_s, opt_s, X2, Y2, loss_fn, epochs=5, is_online=False)

    print(f"[{device}] Classification final avg loss: OnlineSGD={loss_o:.6f} | SGD={loss_s:.6f}")
    assert loss_o <= loss_s * 1.10

# -------------------------------
# 3) Convergence speed (steps to reach SGD's final-loss threshold)
# -------------------------------

@pytest.mark.parametrize("device", ["cuda" if torch.cuda.is_available() else "cpu"])
def test_convergence_speed_steps(device):
    set_deterministic(2)
    Xs, Ys = make_regression(device=device)
    model_s = make_linreg_model(d=Xs.shape[1], device=device)
    opt_s = torch.optim.SGD(model_s.parameters(), **SGD_CFG)
    hist_s, loss_s = epoch_loop(model_s, opt_s, Xs, Ys, torch.nn.MSELoss(), epochs=5, is_online=False)
    thresh = loss_s * 1.05  # 5% slack

    set_deterministic(2)
    Xo, Yo = make_regression(device=device)
    model_o = make_linreg_model(d=Xo.shape[1], device=device)
    opt_o = OnlineSGD(model_o.parameters(), **ONLINE_QUALITY_CFG)
    hist_o, _ = epoch_loop(model_o, opt_o, Xo, Yo, torch.nn.MSELoss(), epochs=5, is_online=True)

    steps_sgd = smoothed_steps_to_threshold(hist_s, thresh, window=3)
    steps_online = smoothed_steps_to_threshold(hist_o, thresh, window=3)

    print(f"[{device}] steps_to_thresh: Online={steps_online} | SGD={steps_sgd} | thresh={thresh:.6f}")
    if steps_sgd is None:
        pytest.skip("SGD did not reach threshold; skipping speed compare.")
    assert steps_online is not None and steps_online <= int(steps_sgd * 1.25) + 1

# -------------------------------
# 4) Robustness across seeds (mean final loss over 3 seeds)
# -------------------------------

@pytest.mark.parametrize("device", ["cuda" if torch.cuda.is_available() else "cpu"])
def test_robustness_across_seeds(device):
    seeds = [0, 1, 2]
    losses_o, losses_s = [], []
    for sd in seeds:
        set_deterministic(sd)
        X, Y = make_regression(device=device)
        model_o = make_linreg_model(d=X.shape[1], device=device)
        opt_o = OnlineSGD(model_o.parameters(), **ONLINE_QUALITY_CFG)
        _, lo = epoch_loop(model_o, opt_o, X, Y, torch.nn.MSELoss(), epochs=5, is_online=True)
        losses_o.append(lo)

        set_deterministic(sd)
        X2, Y2 = make_regression(device=device)
        model_s = make_linreg_model(d=X2.shape[1], device=device)
        opt_s = torch.optim.SGD(model_s.parameters(), **SGD_CFG)
        _, ls = epoch_loop(model_s, opt_s, X2, Y2, torch.nn.MSELoss(), epochs=5, is_online=False)
        losses_s.append(ls)

    mean_o = statistics.mean(losses_o)
    mean_s = statistics.mean(losses_s)
    print(f"[{device}] mean final loss over seeds: Online={mean_o:.6f} | SGD={mean_s:.6f}")
    assert mean_o <= mean_s * 1.10

# -------------------------------
# 5) Overhead per step (wall-clock, trust OFF)
# -------------------------------

@pytest.mark.parametrize("device", ["cuda" if torch.cuda.is_available() else "cpu"])
def test_wallclock_overhead(device):
    set_deterministic(3)
    X, Y = make_regression(device=device)
    model_o = make_linreg_model(d=X.shape[1], device=device)
    opt_o = OnlineSGD(model_o.parameters(), **ONLINE_MEM_TIME_CFG)
    n_steps = (X.size(0) // 64) * 5

    def time_loop(model, opt, is_online):
        loss_fn = torch.nn.MSELoss()
        cuda_sync()
        t0 = time.perf_counter()
        epoch_loop(model, opt, X, Y, loss_fn, epochs=5, is_online=is_online)
        cuda_sync()
        t1 = time.perf_counter()
        return (t1 - t0) / n_steps

    per_step_online = time_loop(model_o, opt_o, True)

    set_deterministic(3)
    X2, Y2 = make_regression(device=device)
    model_s = make_linreg_model(d=X2.shape[1], device=device)
    opt_s = torch.optim.SGD(model_s.parameters(), **SGD_CFG)
    per_step_sgd = time_loop(model_s, opt_s, False)

    print(f"[{device}] time/step: Online={per_step_online*1e6:.1f}µs | SGD={per_step_sgd*1e6:.1f}µs")
    assert per_step_online <= per_step_sgd * 1.5 + 30e-6

# -------------------------------
# 6) Peak memory (trust OFF; apples-to-apples)
# -------------------------------

@pytest.mark.parametrize("device", ["cuda" if torch.cuda.is_available() else "cpu"])
def test_memory_peak(device):
    def train_online():
        set_deterministic(4)
        X, Y = make_regression(device=device)
        model = make_linreg_model(d=X.shape[1], device=device)
        opt = OnlineSGD(model.parameters(), **ONLINE_MEM_TIME_CFG)
        epoch_loop(model, opt, X, Y, torch.nn.MSELoss(), epochs=5, is_online=True)

    def train_sgd():
        set_deterministic(4)
        X, Y = make_regression(device=device)
        model = make_linreg_model(d=X.shape[1], device=device)
        opt = torch.optim.SGD(model.parameters(), **SGD_CFG)
        epoch_loop(model, opt, X, Y, torch.nn.MSELoss(), epochs=5, is_online=False)

    mem_online = peak_memory_bytes_during(train_online)
    mem_sgd = peak_memory_bytes_during(train_sgd)

    if device == "cuda":
        print(f"[{device}] Peak mem: Online={mem_online/1e6:.2f}MB | SGD={mem_sgd/1e6:.2f}MB")
        allowed = mem_sgd * 1.15 + 5 * 1024 * 1024  # 15% + 5MB
        assert mem_online <= allowed, f"OnlineSGD peak mem too high on CUDA: {mem_online} vs {mem_sgd}"
    else:
        print(f"[{device}] Peak mem≈ Online={mem_online/1e6:.2f}MB | SGD={mem_sgd/1e6:.2f}MB")
        allowed = mem_sgd * 1.30 + 20 * 1024 * 1024  # 30% + 20MB
        assert mem_online <= allowed, f"OnlineSGD peak mem too high on CPU: {mem_online} vs {mem_sgd}"
```



[![Video Title](https://img.youtube.com/vi/N26N7lQHNW8/0.jpg)](https://www.youtube.com/watch?v=N26N7lQHNW8)

I AM THE THEORY

I AM THE MONSTER

I AM THE MOTHER

I AM THE CHILD
