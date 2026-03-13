import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import sympy as sp

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ----------------------------
# Sympy parsing + safe eval
# ----------------------------
x, y, z, lam = sp.symbols("x y z lam", real=True)
dx_sym, dy_sym, dz_sym = sp.symbols("dx dy dz", real=True)

SAFE_FUNCS = {
    "x": x, "y": y, "z": z, "lambda": lam, "lam": lam,
    "pi": sp.pi, "e": sp.E,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "sqrt": sp.sqrt, "abs": sp.Abs,
    "ln": sp.log, "log": sp.log, "exp": sp.exp,
    "min": sp.Min, "max": sp.Max
}


def parse_f(expr_str: str) -> sp.Expr:
    s = expr_str.strip()
    if not s:
        raise ValueError("f(x,y) expression is empty.")
    s = s.replace("^", "**")
    return sp.simplify(sp.sympify(s, locals=SAFE_FUNCS))


def parse_g(expr_str: str) -> sp.Expr:
    s = expr_str.strip()
    if not s:
        raise ValueError("g(x,y) expression is empty.")
    s = s.replace("^", "**")
    return sp.simplify(sp.sympify(s, locals=SAFE_FUNCS))


def parse_general(expr_str: str) -> sp.Expr:
    s = expr_str.strip()
    if not s:
        raise ValueError("Expression is empty.")
    s = s.replace("^", "**")
    return sp.simplify(sp.sympify(s, locals=SAFE_FUNCS))


def lambdify_xy(expr: sp.Expr):
    return sp.lambdify((x, y), expr, "numpy")


def finite_mask(*arrays):
    m = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


# ----------------------------
# Numerical helpers
# ----------------------------
def make_grid(xmin, xmax, ymin, ymax, n):
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def classify_critical_point(H: np.ndarray):
    eig = np.linalg.eigvals(H)
    eig = np.real_if_close(eig)
    if np.all(eig > 1e-8):
        return "local min"
    if np.all(eig < -1e-8):
        return "local max"
    if (eig[0] > 1e-8 and eig[1] < -1e-8) or (eig[0] < 1e-8 and eig[1] > 1e-8):
        return "saddle"
    return "inconclusive"


def uniq_points(points, tol=1e-4):
    uniq = []
    for px, py in points:
        ok = True
        for ux, uy in uniq:
            if (px - ux) ** 2 + (py - uy) ** 2 < tol ** 2:
                ok = False
                break
        if ok:
            uniq.append((px, py))
    return uniq


def grid_seeds_from_grad(FX, FY, X, Y, k=10):
    G = np.sqrt(FX * FX + FY * FY)
    if not np.any(np.isfinite(G)):
        return []
    flat = G.ravel()
    idx = np.argsort(flat)
    seeds = []
    for i in idx[: min(k, idx.size)]:
        r = i // G.shape[1]
        c = i % G.shape[1]
        if np.isfinite(G[r, c]):
            seeds.append((float(X[r, c]), float(Y[r, c])))
    return uniq_points(seeds, tol=1e-2)


def safe_nsolve(system, vars_, guess, maxsteps=60):
    try:
        sol = sp.nsolve(system, vars_, guess, tol=1e-14, maxsteps=maxsteps, prec=50)
        sol = [float(sol[i]) for i in range(len(vars_))]
        if all(np.isfinite(sol)):
            return sol
    except Exception:
        return None
    return None


# ----------------------------
# GUI App
# ----------------------------
class SDL4OptimizationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SDL 4 — Optimization Visualizer (Multivariable Calculus)")
        self.geometry("1360x820")

        self.f_str = tk.StringVar(value="x^2 + y^2")
        self.g_str = tk.StringVar(value="x^2 + y^2")
        self.c_str = tk.StringVar(value="1")

        self._build_ui()

    def _build_ui(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_surface = ttk.Frame(self.nb, padding=10)
        self.tab_grad = ttk.Frame(self.nb, padding=10)
        self.tab_dir = ttk.Frame(self.nb, padding=10)
        self.tab_opt = ttk.Frame(self.nb, padding=10)
        self.tab_lagr = ttk.Frame(self.nb, padding=10)
        self.tab_sym = ttk.Frame(self.nb, padding=10)

        self.nb.add(self.tab_surface, text="1) Surface + Contours")
        self.nb.add(self.tab_grad, text="2) Gradient Explorer")
        self.nb.add(self.tab_dir, text="3) Directional Derivative")
        self.nb.add(self.tab_opt, text="4) Optimization (∇f=0)")
        self.nb.add(self.tab_lagr, text="5) Lagrange Multipliers")
        self.nb.add(self.tab_sym, text="6) Derivatives + Eigenvalues")

        self._build_surface_tab()
        self._build_grad_tab()
        self._build_dir_tab()
        self._build_opt_tab()
        self._build_lagr_tab()
        self._build_symbolic_tab()

    # ---- shared figure embedding
    def _embed_figure(self, parent, is3d=False, title=""):
        fig = Figure(figsize=(8.2, 6.4), dpi=100)
        if is3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)
        ax.set_title(title)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, parent)
        toolbar.update()

        return fig, ax, canvas

    def _get_ranges(self, xmin_var, xmax_var, ymin_var, ymax_var, n_var):
        xmin = float(xmin_var.get())
        xmax = float(xmax_var.get())
        ymin = float(ymin_var.get())
        ymax = float(ymax_var.get())
        n = int(float(n_var.get()))
        if n < 50:
            n = 50
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("Invalid ranges. Need xmin<xmax and ymin<ymax.")
        return xmin, xmax, ymin, ymax, n

    def _get_f_expr(self):
        return parse_f(self.f_str.get())

    def _get_g_expr(self):
        return parse_g(self.g_str.get())

    # =========================
    # Tab 1: Surface + Contours
    # =========================
    def _build_surface_tab(self):
        left = ttk.Frame(self.tab_surface)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right = ttk.Frame(self.tab_surface)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        right_top = ttk.Frame(right)
        right_top.pack(fill=tk.BOTH, expand=True)
        right_bot = ttk.Frame(right)
        right_bot.pack(fill=tk.BOTH, expand=True)

        self.figS2, self.axS2, self.canvasS2 = self._embed_figure(
            right_top, is3d=False, title="Contours of f(x,y)"
        )
        self.figS3, self.axS3, self.canvasS3 = self._embed_figure(
            right_bot, is3d=True, title="Surface of f(x,y)"
        )

        ttk.Label(left, text="Function").pack(anchor="w")
        ttk.Label(left, text="f(x,y) =").pack(anchor="w")
        ttk.Entry(left, textvariable=self.f_str, width=34).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Plot window").pack(anchor="w", pady=(6, 0))
        self.s_xmin = tk.StringVar(value="-3")
        self.s_xmax = tk.StringVar(value="3")
        self.s_ymin = tk.StringVar(value="-3")
        self.s_ymax = tk.StringVar(value="3")
        self.s_n = tk.StringVar(value="200")

        grid = ttk.Frame(left)
        grid.pack(anchor="w", pady=(2, 8))
        ttk.Label(grid, text="x:").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.s_xmin, width=8).grid(row=0, column=1)
        ttk.Label(grid, text="to").grid(row=0, column=2)
        ttk.Entry(grid, textvariable=self.s_xmax, width=8).grid(row=0, column=3)

        ttk.Label(grid, text="y:").grid(row=1, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.s_ymin, width=8).grid(row=1, column=1)
        ttk.Label(grid, text="to").grid(row=1, column=2)
        ttk.Entry(grid, textvariable=self.s_ymax, width=8).grid(row=1, column=3)

        ttk.Label(left, text="Grid samples").pack(anchor="w")
        ttk.Entry(left, textvariable=self.s_n, width=10).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Render Contours + Surface", command=self.render_surface).pack(anchor="w", pady=(6, 0))

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Quick examples").pack(anchor="w")
        ex = ttk.Frame(left)
        ex.pack(anchor="w", pady=4)
        ttk.Button(ex, text="Bowl", command=lambda: self._set_f_and_render("x^2 + y^2")).pack(side=tk.LEFT)
        ttk.Button(ex, text="Saddle", command=lambda: self._set_f_and_render("x^2 - y^2")).pack(side=tk.LEFT, padx=6)
        ttk.Button(ex, text="Sin bumps", command=lambda: self._set_f_and_render("sin(x) + cos(y)")).pack(side=tk.LEFT)

        self.statusS = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.statusS, foreground="#005").pack(anchor="w", pady=(10, 0))

        self.render_surface()

    def _set_f_and_render(self, s):
        self.f_str.set(s)
        self.render_surface()

    def render_surface(self):
        try:
            expr = self._get_f_expr()
            f = lambdify_xy(expr)

            xmin, xmax, ymin, ymax, n = self._get_ranges(
                self.s_xmin, self.s_xmax, self.s_ymin, self.s_ymax, self.s_n
            )
            X, Y = make_grid(xmin, xmax, ymin, ymax, n)
            Z = np.asarray(f(X, Y), dtype=float)

            m = finite_mask(Z)
            if not np.any(m):
                raise ValueError("f(x,y) produced no finite values on this window.")

            self.axS2.clear()
            self.axS2.grid(True)
            self.axS2.set_aspect("equal", adjustable="box")
            self.axS2.set_title("Contours of f(x,y)")
            self.axS2.set_xlabel("x")
            self.axS2.set_ylabel("y")
            self.axS2.contour(X, Y, Z, levels=18)
            self.canvasS2.draw()

            self.axS3.clear()
            self.axS3.set_title("Surface of f(x,y)")
            self.axS3.set_xlabel("x")
            self.axS3.set_ylabel("y")
            self.axS3.set_zlabel("f")
            step = max(1, n // 120)
            self.axS3.plot_surface(
                X[::step, ::step],
                Y[::step, ::step],
                Z[::step, ::step],
                linewidth=0,
                antialiased=True,
                alpha=0.9
            )
            self.canvasS3.draw()

            self.statusS.set("Rendered.")
        except Exception as e:
            self.statusS.set("Error.")
            messagebox.showerror("Render Error", str(e))

    # =========================
    # Tab 2: Gradient Explorer
    # =========================
    def _build_grad_tab(self):
        left = ttk.Frame(self.tab_grad)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right = ttk.Frame(self.tab_grad)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figG, self.axG, self.canvasG = self._embed_figure(
            right, is3d=False, title="Contours + Gradient Field + ∇f(x0,y0)"
        )

        ttk.Label(left, text="Function").pack(anchor="w")
        ttk.Label(left, text="f(x,y) =").pack(anchor="w")
        ttk.Entry(left, textvariable=self.f_str, width=34).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Point").pack(anchor="w")
        self.g_x0 = tk.StringVar(value="1")
        self.g_y0 = tk.StringVar(value="1")
        row = ttk.Frame(left)
        row.pack(anchor="w", pady=(2, 8))
        ttk.Label(row, text="x0").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.g_x0, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="y0").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.g_y0, width=8).pack(side=tk.LEFT, padx=6)

        ttk.Label(left, text="Plot window").pack(anchor="w", pady=(6, 0))
        self.g_xmin = tk.StringVar(value="-3")
        self.g_xmax = tk.StringVar(value="3")
        self.g_ymin = tk.StringVar(value="-3")
        self.g_ymax = tk.StringVar(value="3")
        self.g_n = tk.StringVar(value="120")
        self.g_vec_stride = tk.StringVar(value="8")

        grid = ttk.Frame(left)
        grid.pack(anchor="w", pady=(2, 8))
        ttk.Label(grid, text="x:").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.g_xmin, width=8).grid(row=0, column=1)
        ttk.Label(grid, text="to").grid(row=0, column=2)
        ttk.Entry(grid, textvariable=self.g_xmax, width=8).grid(row=0, column=3)

        ttk.Label(grid, text="y:").grid(row=1, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.g_ymin, width=8).grid(row=1, column=1)
        ttk.Label(grid, text="to").grid(row=1, column=2)
        ttk.Entry(grid, textvariable=self.g_ymax, width=8).grid(row=1, column=3)

        ttk.Label(left, text="Grid samples").pack(anchor="w")
        ttk.Entry(left, textvariable=self.g_n, width=10).pack(anchor="w", pady=(0, 6))
        ttk.Label(left, text="Vector stride (bigger = fewer arrows)").pack(anchor="w")
        ttk.Entry(left, textvariable=self.g_vec_stride, width=10).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Render Gradient View", command=self.render_gradient).pack(anchor="w", pady=(6, 0))

        self.grad_out = tk.StringVar(value="∇f(x0,y0) = ")
        ttk.Label(left, textvariable=self.grad_out, wraplength=280).pack(anchor="w", pady=(10, 0))

        self.statusG = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.statusG, foreground="#005").pack(anchor="w", pady=(6, 0))

        self.render_gradient()

    def render_gradient(self):
        try:
            expr = self._get_f_expr()
            fx = sp.diff(expr, x)
            fy = sp.diff(expr, y)

            f = lambdify_xy(expr)
            fx_f = lambdify_xy(fx)
            fy_f = lambdify_xy(fy)

            xmin, xmax, ymin, ymax, n = self._get_ranges(
                self.g_xmin, self.g_xmax, self.g_ymin, self.g_ymax, self.g_n
            )
            X, Y = make_grid(xmin, xmax, ymin, ymax, n)
            Z = np.asarray(f(X, Y), dtype=float)
            FX = np.asarray(fx_f(X, Y), dtype=float)
            FY = np.asarray(fy_f(X, Y), dtype=float)

            x0 = float(self.g_x0.get())
            y0 = float(self.g_y0.get())
            gx0 = float(fx_f(x0, y0))
            gy0 = float(fy_f(x0, y0))
            self.grad_out.set(f"∇f({x0:.4g},{y0:.4g}) = <{gx0:.6g}, {gy0:.6g}>")

            self.axG.clear()
            self.axG.grid(True)
            self.axG.set_aspect("equal", adjustable="box")
            self.axG.set_title("Contours + Gradient Field + ∇f(x0,y0)")
            self.axG.set_xlabel("x")
            self.axG.set_ylabel("y")
            self.axG.contour(X, Y, Z, levels=18)

            stride = max(1, int(float(self.g_vec_stride.get())))
            self.axG.quiver(
                X[::stride, ::stride], Y[::stride, ::stride],
                FX[::stride, ::stride], FY[::stride, ::stride],
                angles="xy", scale_units="xy", scale=25
            )

            self.axG.scatter([x0], [y0], s=40)
            self.axG.quiver([x0], [y0], [gx0], [gy0], angles="xy", scale_units="xy", scale=1)
            self.axG.text(x0, y0, "  (x0,y0)", fontsize=9)

            self.canvasG.draw()
            self.statusG.set("Rendered.")
        except Exception as e:
            self.statusG.set("Error.")
            messagebox.showerror("Gradient Error", str(e))

    # =========================
    # Tab 3: Directional Derivative
    # =========================
    def _build_dir_tab(self):
        left = ttk.Frame(self.tab_dir)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right = ttk.Frame(self.tab_dir)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figD, self.axD, self.canvasD = self._embed_figure(
            right, is3d=False, title="Directional Derivative at (x0,y0)"
        )

        ttk.Label(left, text="Function").pack(anchor="w")
        ttk.Label(left, text="f(x,y) =").pack(anchor="w")
        ttk.Entry(left, textvariable=self.f_str, width=34).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Point").pack(anchor="w")
        self.d_x0 = tk.StringVar(value="1")
        self.d_y0 = tk.StringVar(value="1")
        row = ttk.Frame(left)
        row.pack(anchor="w", pady=(2, 8))
        ttk.Label(row, text="x0").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.d_x0, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Label(row, text="y0").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.d_y0, width=8).pack(side=tk.LEFT, padx=6)

        ttk.Label(left, text="Direction vector v = <vx, vy>").pack(anchor="w")
        self.vx = tk.StringVar(value="1")
        self.vy = tk.StringVar(value="0")
        row2 = ttk.Frame(left)
        row2.pack(anchor="w", pady=(2, 8))
        ttk.Label(row2, text="vx").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.vx, width=8).pack(side=tk.LEFT, padx=6)
        ttk.Label(row2, text="vy").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.vy, width=8).pack(side=tk.LEFT, padx=6)

        ttk.Label(left, text="Plot window").pack(anchor="w", pady=(6, 0))
        self.d_xmin = tk.StringVar(value="-3")
        self.d_xmax = tk.StringVar(value="3")
        self.d_ymin = tk.StringVar(value="-3")
        self.d_ymax = tk.StringVar(value="3")
        self.d_n = tk.StringVar(value="160")

        grid = ttk.Frame(left)
        grid.pack(anchor="w", pady=(2, 8))
        ttk.Label(grid, text="x:").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.d_xmin, width=8).grid(row=0, column=1)
        ttk.Label(grid, text="to").grid(row=0, column=2)
        ttk.Entry(grid, textvariable=self.d_xmax, width=8).grid(row=0, column=3)
        ttk.Label(grid, text="y:").grid(row=1, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.d_ymin, width=8).grid(row=1, column=1)
        ttk.Label(grid, text="to").grid(row=1, column=2)
        ttk.Entry(grid, textvariable=self.d_ymax, width=8).grid(row=1, column=3)

        ttk.Label(left, text="Grid samples").pack(anchor="w")
        ttk.Entry(left, textvariable=self.d_n, width=10).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Compute Directional Derivative", command=self.render_directional).pack(anchor="w", pady=(6, 0))

        self.dir_out = tk.StringVar(value="D_v f = ")
        ttk.Label(left, textvariable=self.dir_out, wraplength=280).pack(anchor="w", pady=(10, 0))

        self.statusD = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.statusD, foreground="#005").pack(anchor="w", pady=(6, 0))

        self.render_directional()

    def render_directional(self):
        try:
            expr = self._get_f_expr()
            fx = sp.diff(expr, x)
            fy = sp.diff(expr, y)
            fx_f = lambdify_xy(fx)
            fy_f = lambdify_xy(fy)
            f = lambdify_xy(expr)

            x0 = float(self.d_x0.get())
            y0 = float(self.d_y0.get())
            gx0 = float(fx_f(x0, y0))
            gy0 = float(fy_f(x0, y0))

            vx = float(self.vx.get())
            vy = float(self.vy.get())
            norm = float(np.hypot(vx, vy))
            if norm < 1e-12:
                raise ValueError("Direction vector cannot be <0,0>.")
            vhat = (vx / norm, vy / norm)

            Dv = gx0 * vhat[0] + gy0 * vhat[1]
            self.dir_out.set(
                f"∇f({x0:.4g},{y0:.4g})=< {gx0:.6g}, {gy0:.6g} >;  "
                f"v̂=< {vhat[0]:.6g}, {vhat[1]:.6g} >;  D_v f = {Dv:.6g}"
            )

            xmin, xmax, ymin, ymax, n = self._get_ranges(
                self.d_xmin, self.d_xmax, self.d_ymin, self.d_ymax, self.d_n
            )
            X, Y = make_grid(xmin, xmax, ymin, ymax, n)
            Z = np.asarray(f(X, Y), dtype=float)

            self.axD.clear()
            self.axD.grid(True)
            self.axD.set_aspect("equal", adjustable="box")
            self.axD.set_title("Directional Derivative at (x0,y0)")
            self.axD.set_xlabel("x")
            self.axD.set_ylabel("y")
            self.axD.contour(X, Y, Z, levels=18)

            self.axD.scatter([x0], [y0], s=45)
            self.axD.quiver([x0], [y0], [gx0], [gy0], angles="xy", scale_units="xy", scale=1, width=0.008)
            self.axD.quiver([x0], [y0], [vhat[0]], [vhat[1]], angles="xy", scale_units="xy", scale=1, width=0.008)
            self.axD.text(x0, y0, "  (x0,y0)", fontsize=9)

            self.canvasD.draw()
            self.statusD.set("Computed.")
        except Exception as e:
            self.statusD.set("Error.")
            messagebox.showerror("Directional Derivative Error", str(e))

    # =========================
    # Tab 4: Optimization (critical points)
    # =========================
    def _build_opt_tab(self):
        left = ttk.Frame(self.tab_opt)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right = ttk.Frame(self.tab_opt)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figO, self.axO, self.canvasO = self._embed_figure(
            right, is3d=False, title="Contours + Critical Points"
        )

        ttk.Label(left, text="Function").pack(anchor="w")
        ttk.Label(left, text="f(x,y) =").pack(anchor="w")
        ttk.Entry(left, textvariable=self.f_str, width=34).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Search window").pack(anchor="w", pady=(6, 0))
        self.o_xmin = tk.StringVar(value="-3")
        self.o_xmax = tk.StringVar(value="3")
        self.o_ymin = tk.StringVar(value="-3")
        self.o_ymax = tk.StringVar(value="3")
        self.o_n = tk.StringVar(value="160")
        self.o_seedk = tk.StringVar(value="10")

        grid = ttk.Frame(left)
        grid.pack(anchor="w", pady=(2, 8))
        ttk.Label(grid, text="x:").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.o_xmin, width=8).grid(row=0, column=1)
        ttk.Label(grid, text="to").grid(row=0, column=2)
        ttk.Entry(grid, textvariable=self.o_xmax, width=8).grid(row=0, column=3)
        ttk.Label(grid, text="y:").grid(row=1, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.o_ymin, width=8).grid(row=1, column=1)
        ttk.Label(grid, text="to").grid(row=1, column=2)
        ttk.Entry(grid, textvariable=self.o_ymax, width=8).grid(row=1, column=3)

        ttk.Label(left, text="Grid samples").pack(anchor="w")
        ttk.Entry(left, textvariable=self.o_n, width=10).pack(anchor="w", pady=(0, 6))
        ttk.Label(left, text="Seeds for nsolve (top-k)").pack(anchor="w")
        ttk.Entry(left, textvariable=self.o_seedk, width=10).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Find & Classify Critical Points", command=self.find_critical_points).pack(anchor="w", pady=(6, 0))

        ttk.Label(left, text="Results").pack(anchor="w", pady=(10, 0))
        self.opt_results = tk.Text(left, height=14, width=38, wrap="word")
        self.opt_results.pack(anchor="w")
        self.opt_results.insert("1.0", "Critical points will appear here.\n")
        self.opt_results.config(state="disabled")

        self.statusO = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.statusO, foreground="#005").pack(anchor="w", pady=(6, 0))

        self.find_critical_points()

    def _set_opt_results(self, text):
        self.opt_results.config(state="normal")
        self.opt_results.delete("1.0", tk.END)
        self.opt_results.insert("1.0", text)
        self.opt_results.config(state="disabled")

    def find_critical_points(self):
        try:
            expr = self._get_f_expr()
            fx = sp.diff(expr, x)
            fy = sp.diff(expr, y)
            f = lambdify_xy(expr)
            fx_f = lambdify_xy(fx)
            fy_f = lambdify_xy(fy)

            xmin, xmax, ymin, ymax, n = self._get_ranges(
                self.o_xmin, self.o_xmax, self.o_ymin, self.o_ymax, self.o_n
            )
            X, Y = make_grid(xmin, xmax, ymin, ymax, n)
            Z = np.asarray(f(X, Y), dtype=float)
            FX = np.asarray(fx_f(X, Y), dtype=float)
            FY = np.asarray(fy_f(X, Y), dtype=float)

            k = int(float(self.o_seedk.get()))
            seeds = grid_seeds_from_grad(FX, FY, X, Y, k=k)

            sols = []
            try:
                sol_sym = sp.solve([sp.Eq(fx, 0), sp.Eq(fy, 0)], [x, y], dict=True)
                for s in sol_sym:
                    sx = sp.N(s[x])
                    sy = sp.N(s[y])
                    if sx.is_real and sy.is_real:
                        sols.append((float(sx), float(sy)))
            except Exception:
                pass

            system = (fx, fy)
            for seed in seeds:
                sol = safe_nsolve(system, (x, y), seed)
                if sol is not None:
                    sols.append((sol[0], sol[1]))

            sols = uniq_points(sols, tol=1e-3)

            fxx = sp.diff(fx, x)
            fxy = sp.diff(fx, y)
            fyy = sp.diff(fy, y)
            fxx_f = lambdify_xy(fxx)
            fxy_f = lambdify_xy(fxy)
            fyy_f = lambdify_xy(fyy)

            lines = []
            self.axO.clear()
            self.axO.grid(True)
            self.axO.set_aspect("equal", adjustable="box")
            self.axO.set_title("Contours + Critical Points")
            self.axO.set_xlabel("x")
            self.axO.set_ylabel("y")
            self.axO.contour(X, Y, Z, levels=18)

            if not sols:
                lines.append("No critical points found in the window.\nTip: Try a different window or a simpler function.\n")
            else:
                for i, (cx, cy) in enumerate(sols):
                    H = np.array([
                        [float(fxx_f(cx, cy)), float(fxy_f(cx, cy))],
                        [float(fxy_f(cx, cy)), float(fyy_f(cx, cy))]
                    ], dtype=float)
                    eigvals = np.linalg.eigvals(H)
                    eigvals = np.real_if_close(eigvals)
                    kind = classify_critical_point(H)
                    val = float(f(cx, cy))
                    lines.append(
                        f"{i+1}) (x,y)=({cx:.6g}, {cy:.6g}), f={val:.6g}, "
                        f"type: {kind}, Hessian eigenvalues={eigvals}"
                    )
                    self.axO.scatter([cx], [cy], s=55)
                    self.axO.text(cx, cy, f"  {i+1}", fontsize=9)

            self.canvasO.draw()
            self._set_opt_results("\n".join(lines))
            self.statusO.set(f"Found {len(sols)} candidate critical point(s).")
        except Exception as e:
            self.statusO.set("Error.")
            messagebox.showerror("Optimization Error", str(e))

    # =========================
    # Tab 5: Lagrange multipliers
    # =========================
    def _build_lagr_tab(self):
        left = ttk.Frame(self.tab_lagr)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right = ttk.Frame(self.tab_lagr)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.figL, self.axL, self.canvasL = self._embed_figure(
            right, is3d=False, title="Constraint curve + f contours + Lagrange points"
        )

        ttk.Label(left, text="Objective").pack(anchor="w")
        ttk.Label(left, text="f(x,y) =").pack(anchor="w")
        ttk.Entry(left, textvariable=self.f_str, width=34).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Constraint").pack(anchor="w")
        ttk.Label(left, text="g(x,y) =").pack(anchor="w")
        ttk.Entry(left, textvariable=self.g_str, width=34).pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Constraint value").pack(anchor="w")
        rowc = ttk.Frame(left)
        rowc.pack(anchor="w", pady=(2, 8))
        ttk.Label(rowc, text="c =").pack(side=tk.LEFT)
        ttk.Entry(rowc, textvariable=self.c_str, width=10).pack(side=tk.LEFT, padx=6)

        ttk.Label(left, text="Plot window").pack(anchor="w", pady=(6, 0))
        self.l_xmin = tk.StringVar(value="-3")
        self.l_xmax = tk.StringVar(value="3")
        self.l_ymin = tk.StringVar(value="-3")
        self.l_ymax = tk.StringVar(value="3")
        self.l_n = tk.StringVar(value="220")

        grid = ttk.Frame(left)
        grid.pack(anchor="w", pady=(2, 8))
        ttk.Label(grid, text="x:").grid(row=0, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.l_xmin, width=8).grid(row=0, column=1)
        ttk.Label(grid, text="to").grid(row=0, column=2)
        ttk.Entry(grid, textvariable=self.l_xmax, width=8).grid(row=0, column=3)
        ttk.Label(grid, text="y:").grid(row=1, column=0, sticky="w")
        ttk.Entry(grid, textvariable=self.l_ymin, width=8).grid(row=1, column=1)
        ttk.Label(grid, text="to").grid(row=1, column=2)
        ttk.Entry(grid, textvariable=self.l_ymax, width=8).grid(row=1, column=3)

        ttk.Label(left, text="Grid samples").pack(anchor="w")
        ttk.Entry(left, textvariable=self.l_n, width=10).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Solve Lagrange System + Plot", command=self.solve_lagrange).pack(anchor="w", pady=(6, 0))

        ttk.Label(left, text="Candidates").pack(anchor="w", pady=(10, 0))
        self.lagr_results = tk.Text(left, height=14, width=38, wrap="word")
        self.lagr_results.pack(anchor="w")
        self.lagr_results.insert("1.0", "Solutions will appear here.\n")
        self.lagr_results.config(state="disabled")

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Quick examples").pack(anchor="w")
        ex = ttk.Frame(left)
        ex.pack(anchor="w", pady=4)
        ttk.Button(ex, text="Max xy on circle", command=self._ex_xy_circle).pack(side=tk.LEFT)
        ttk.Button(ex, text="Min distance to point", command=self._ex_dist_circle).pack(side=tk.LEFT, padx=6)

        self.statusL = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.statusL, foreground="#005").pack(anchor="w", pady=(6, 0))

        self._ex_xy_circle()

    def _set_lagr_results(self, text):
        self.lagr_results.config(state="normal")
        self.lagr_results.delete("1.0", tk.END)
        self.lagr_results.insert("1.0", text)
        self.lagr_results.config(state="disabled")

    def _ex_xy_circle(self):
        self.f_str.set("x*y")
        self.g_str.set("x^2 + y^2")
        self.c_str.set("1")
        self.solve_lagrange()

    def _ex_dist_circle(self):
        self.f_str.set("(x-1)^2 + (y-2)^2")
        self.g_str.set("x^2 + y^2")
        self.c_str.set("1")
        self.solve_lagrange()

    def solve_lagrange(self):
        try:
            f_expr = self._get_f_expr()
            g_expr = self._get_g_expr()
            c_val = float(self.c_str.get())

            fx = sp.diff(f_expr, x)
            fy = sp.diff(f_expr, y)
            gx = sp.diff(g_expr, x)
            gy = sp.diff(g_expr, y)

            eqs = [
                sp.Eq(fx, lam * gx),
                sp.Eq(fy, lam * gy),
                sp.Eq(g_expr, c_val)
            ]

            sols = []
            try:
                sol_sym = sp.solve(eqs, [x, y, lam], dict=True)
                for s in sol_sym:
                    sx = sp.N(s[x])
                    sy = sp.N(s[y])
                    sl = sp.N(s[lam])
                    if sx.is_real and sy.is_real:
                        sols.append((float(sx), float(sy), float(sl)))
            except Exception:
                pass

            xmin, xmax, ymin, ymax, n = self._get_ranges(
                self.l_xmin, self.l_xmax, self.l_ymin, self.l_ymax, self.l_n
            )
            X, Y = make_grid(xmin, xmax, ymin, ymax, max(80, n // 2))
            g_f = lambdify_xy(g_expr)
            G = np.asarray(g_f(X, Y), dtype=float)
            diff = np.abs(G - c_val)
            if np.any(np.isfinite(diff)):
                idx = np.argsort(diff.ravel())[:12]
                for i in idx:
                    r = i // diff.shape[1]
                    c = i % diff.shape[1]
                    seed = (float(X[r, c]), float(Y[r, c]), 1.0)
                    sol = safe_nsolve(
                        (fx - lam * gx, fy - lam * gy, g_expr - c_val),
                        (x, y, lam),
                        seed,
                        maxsteps=80
                    )
                    if sol is not None:
                        sols.append((sol[0], sol[1], sol[2]))

            uniq = []
            for sx, sy, sl in sols:
                ok = True
                for ux, uy, ul in uniq:
                    if (sx - ux) ** 2 + (sy - uy) ** 2 < 1e-6:
                        ok = False
                        break
                if ok:
                    uniq.append((sx, sy, sl))
            sols = uniq

            f_f = lambdify_xy(f_expr)
            cand = []
            for sx, sy, sl in sols:
                val = float(f_f(sx, sy))
                if np.isfinite(val):
                    cand.append((sx, sy, sl, val))
            cand.sort(key=lambda t: t[3])

            f_plot = lambdify_xy(f_expr)
            xmin, xmax, ymin, ymax, n = self._get_ranges(
                self.l_xmin, self.l_xmax, self.l_ymin, self.l_ymax, self.l_n
            )
            X, Y = make_grid(xmin, xmax, ymin, ymax, n)
            Z = np.asarray(f_plot(X, Y), dtype=float)
            G = np.asarray(g_f(X, Y), dtype=float)

            self.axL.clear()
            self.axL.grid(True)
            self.axL.set_aspect("equal", adjustable="box")
            self.axL.set_title("Constraint curve + f contours + Lagrange points")
            self.axL.set_xlabel("x")
            self.axL.set_ylabel("y")

            self.axL.contour(X, Y, Z, levels=18)
            self.axL.contour(X, Y, G, levels=[c_val], linewidths=2)

            lines = []
            if not cand:
                lines.append(
                    "No candidate solutions found.\nTips:\n"
                    "- Try a larger window\n"
                    "- Use a simpler f/g\n"
                    "- Ensure the constraint has points in the window\n"
                )
            else:
                for i, (sx, sy, sl, val) in enumerate(cand, start=1):
                    self.axL.scatter([sx], [sy], s=55)
                    self.axL.text(sx, sy, f"  {i}", fontsize=9)
                    lines.append(f"{i}) (x,y)=({sx:.6g}, {sy:.6g}), λ={sl:.6g}, f={val:.6g}")

                minp = cand[0]
                maxp = cand[-1]
                lines.append("")
                lines.append(f"Smallest f among candidates: f={minp[3]:.6g} at ({minp[0]:.6g}, {minp[1]:.6g})")
                lines.append(f"Largest  f among candidates: f={maxp[3]:.6g} at ({maxp[0]:.6g}, {maxp[1]:.6g})")

            self.canvasL.draw()
            self._set_lagr_results("\n".join(lines))
            self.statusL.set(f"Found {len(cand)} candidate(s).")
        except Exception as e:
            self.statusL.set("Error.")
            messagebox.showerror("Lagrange Error", str(e))

    # =========================
    # Tab 6: Derivatives + Eigenvalues
    # =========================
    def _build_symbolic_tab(self):
        left = ttk.Frame(self.tab_sym)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right = ttk.Frame(self.tab_sym)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="Function").pack(anchor="w")
        self.sym_expr = tk.StringVar(value="x^2 + y^2")
        ttk.Entry(left, textvariable=self.sym_expr, width=36).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Variables used").pack(anchor="w")
        self.sym_mode = tk.StringVar(value="x,y")
        ttk.Combobox(
            left,
            textvariable=self.sym_mode,
            values=["x,y", "x,y,z"],
            state="readonly",
            width=10
        ).pack(anchor="w", pady=(0, 8))

        ttk.Label(left, text="Optional evaluation point").pack(anchor="w")
        self.sym_x0 = tk.StringVar(value="1")
        self.sym_y0 = tk.StringVar(value="1")
        self.sym_z0 = tk.StringVar(value="1")

        row = ttk.Frame(left)
        row.pack(anchor="w", pady=(2, 8))
        ttk.Label(row, text="x0").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.sym_x0, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="y0").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.sym_y0, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(row, text="z0").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.sym_z0, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Button(left, text="Analyze Function", command=self.analyze_symbolic).pack(anchor="w", pady=(6, 0))

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Label(left, text="Quick examples").pack(anchor="w")
        ex = ttk.Frame(left)
        ex.pack(anchor="w", pady=4)
        ttk.Button(ex, text="2D bowl", command=lambda: self._load_symbolic_example("x^2 + y^2", "x,y")).pack(side=tk.LEFT)
        ttk.Button(ex, text="2D saddle", command=lambda: self._load_symbolic_example("x^2 - y^2", "x,y")).pack(side=tk.LEFT, padx=6)
        ttk.Button(ex, text="3D quadric", command=lambda: self._load_symbolic_example("x^2 + y^2 + z^2", "x,y,z")).pack(side=tk.LEFT)

        self.sym_status = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.sym_status, foreground="#005").pack(anchor="w", pady=(8, 0))

        self.sym_text = tk.Text(right, wrap="word", font=("Consolas", 11))
        self.sym_text.pack(fill=tk.BOTH, expand=True)
        self.sym_text.insert("1.0", "Symbolic results will appear here.\n")
        self.sym_text.config(state="disabled")

    def _load_symbolic_example(self, expr, mode):
        self.sym_expr.set(expr)
        self.sym_mode.set(mode)
        self.analyze_symbolic()

    def _set_sym_text(self, text):
        self.sym_text.config(state="normal")
        self.sym_text.delete("1.0", tk.END)
        self.sym_text.insert("1.0", text)
        self.sym_text.config(state="disabled")

    def analyze_symbolic(self):
        try:
            expr = parse_general(self.sym_expr.get())
            mode = self.sym_mode.get()

            if mode == "x,y":
                fx = sp.diff(expr, x)
                fy = sp.diff(expr, y)

                df_expr = fx * dx_sym + fy * dy_sym
                grad = sp.Matrix([fx, fy])
                hess = sp.hessian(expr, (x, y))
                eigs = hess.eigenvals()

                x0 = float(self.sym_x0.get())
                y0 = float(self.sym_y0.get())

                subs_map = {x: x0, y: y0}
                f_val = sp.N(expr.subs(subs_map))
                fx_val = sp.N(fx.subs(subs_map))
                fy_val = sp.N(fy.subs(subs_map))
                grad_val = sp.Matrix([fx_val, fy_val])
                hess_val = sp.Matrix(hess.subs(subs_map))
                eigvals_num = [sp.N(ev) for ev in hess_val.eigenvals().keys()]

                out = []
                out.append(f"f(x,y) = {sp.simplify(expr)}\n")
                out.append(f"df = {sp.simplify(df_expr)}\n")
                out.append(f"fx = {sp.simplify(fx)}")
                out.append(f"fy = {sp.simplify(fy)}\n")
                out.append(f"∇f = {grad}\n")
                out.append(f"Hessian =\n{hess}\n")
                out.append("Symbolic Hessian eigenvalues:")
                for ev, mult in eigs.items():
                    out.append(f"  {sp.simplify(ev)}   (mult={mult})")

                out.append("\nAt the point:")
                out.append(f"(x0,y0)=({x0}, {y0})")
                out.append(f"f(x0,y0) = {f_val}")
                out.append(f"fx(x0,y0) = {fx_val}")
                out.append(f"fy(x0,y0) = {fy_val}")
                out.append(f"∇f(x0,y0) = {grad_val}")
                out.append(f"Hessian(x0,y0) =\n{hess_val}")
                out.append(f"Numeric Hessian eigenvalues = {eigvals_num}")

                self._set_sym_text("\n".join(str(s) for s in out))

            else:
                fx = sp.diff(expr, x)
                fy = sp.diff(expr, y)
                fz = sp.diff(expr, z)

                df_expr = fx * dx_sym + fy * dy_sym + fz * dz_sym
                grad = sp.Matrix([fx, fy, fz])
                hess = sp.hessian(expr, (x, y, z))
                eigs = hess.eigenvals()

                x0 = float(self.sym_x0.get())
                y0 = float(self.sym_y0.get())
                z0 = float(self.sym_z0.get())

                subs_map = {x: x0, y: y0, z: z0}
                f_val = sp.N(expr.subs(subs_map))
                fx_val = sp.N(fx.subs(subs_map))
                fy_val = sp.N(fy.subs(subs_map))
                fz_val = sp.N(fz.subs(subs_map))
                grad_val = sp.Matrix([fx_val, fy_val, fz_val])
                hess_val = sp.Matrix(hess.subs(subs_map))
                eigvals_num = [sp.N(ev) for ev in hess_val.eigenvals().keys()]

                out = []
                out.append(f"f(x,y,z) = {sp.simplify(expr)}\n")
                out.append(f"df = {sp.simplify(df_expr)}\n")
                out.append(f"fx = {sp.simplify(fx)}")
                out.append(f"fy = {sp.simplify(fy)}")
                out.append(f"fz = {sp.simplify(fz)}\n")
                out.append(f"∇f = {grad}\n")
                out.append(f"Hessian =\n{hess}\n")
                out.append("Symbolic Hessian eigenvalues:")
                for ev, mult in eigs.items():
                    out.append(f"  {sp.simplify(ev)}   (mult={mult})")

                out.append("\nAt the point:")
                out.append(f"(x0,y0,z0)=({x0}, {y0}, {z0})")
                out.append(f"f(x0,y0,z0) = {f_val}")
                out.append(f"fx = {fx_val}")
                out.append(f"fy = {fy_val}")
                out.append(f"fz = {fz_val}")
                out.append(f"∇f = {grad_val}")
                out.append(f"Hessian =\n{hess_val}")
                out.append(f"Numeric Hessian eigenvalues = {eigvals_num}")

                self._set_sym_text("\n".join(str(s) for s in out))

            self.sym_status.set("Analyzed.")
        except Exception as e:
            self.sym_status.set("Error.")
            messagebox.showerror("Symbolic Analysis Error", str(e))


if __name__ == "__main__":
    app = SDL4OptimizationApp()
    app.mainloop()