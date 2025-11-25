import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import convolve2d

plt.rcParams["font.size"] = 9

# =======================================================================
# LIST SUMBER GRAVITASI (user yang mengisi)
# =======================================================================
# Setiap sumber: { "name":..., "x":..., "y":..., "z":..., "rho":... }
sources = []

# variabel global untuk menyimpan grid terakhir
last_X = None
last_Y = None
last_gz = None   # dalam mGal

# =======================================================================
# FUNGSI HITUNG g_z UNTUK BANYAK SUMBER
# =======================================================================
def compute_gz_grid(x_grid, y_grid, source_list):
    """
    Hitung g_z di grid akibat beberapa sumber bola (sphere approximation).
    Koordinat sumber: (x, y, z) dengan z = kedalaman (m, positif ke bawah).
    """
    G = 6.674e-11
    gz_total = np.zeros_like(x_grid, dtype=float)

    # Asumsi radius bola konstan (atau bisa ditambah input per sumber)
    R = 20.0  # m, misal

    for s in source_list:
        xs = s["x"]
        ys = s["y"]
        zs = s["z"]     # kedalaman
        rho = s["rho"]

        Xs = x_grid - xs
        Ys = y_grid - ys
        Zs = zs

        r = np.sqrt(Xs**2 + Ys**2 + Zs**2)
        # hindari pembagian 0
        r[r == 0] = 1e-6

        # gz sphere
        gz = (4/3) * np.pi * G * rho * (R**3) * (Zs / (r**3))
        gz_total += gz

    return gz_total * 1e5   # ke mGal

# =======================================================================
# FUNGSI INTERPOLASI BILINEAR DI GRID
# =======================================================================
def bilinear_interp(x_grid, y_grid, z_grid, xq, yq):
    """
    Interpolasi bilinear nilai z_grid pada titik (xq, yq).
    x_grid, y_grid: 2D meshgrid (ukuran sama dengan z_grid)
    xq, yq: skalar
    return: zq (float)
    """
    # anggap grid teratur
    x_vals = x_grid[0, :]      # sumbu X
    y_vals = y_grid[:, 0]      # sumbu Y

    # cek apakah di dalam domain
    if not (x_vals[0] <= xq <= x_vals[-1] and y_vals[0] <= yq <= y_vals[-1]):
        return None  # di luar grid

    # indeks kiri/kanan dan bawah/atas
    i = np.searchsorted(x_vals, xq) - 1
    j = np.searchsorted(y_vals, yq) - 1

    i = np.clip(i, 0, len(x_vals)-2)
    j = np.clip(j, 0, len(y_vals)-2)

    x1, x2 = x_vals[i], x_vals[i+1]
    y1, y2 = y_vals[j], y_vals[j+1]

    Q11 = z_grid[j,   i]
    Q21 = z_grid[j,   i+1]
    Q12 = z_grid[j+1, i]
    Q22 = z_grid[j+1, i+1]

    # koefisien normalisasi
    if x2 == x1 or y2 == y1:
        return float(Q11)

    tx = (xq - x1) / (x2 - x1)
    ty = (yq - y1) / (y2 - y1)

    # bilinear
    zq = (Q11 * (1-tx) * (1-ty) +
          Q21 * tx     * (1-ty) +
          Q12 * (1-tx) * ty     +
          Q22 * tx     * ty)
    return float(zq)

def hitung_interpolasi():
    global last_X, last_Y, last_gz
    if last_X is None or last_Y is None or last_gz is None:
        messagebox.showwarning("Peringatan", "Jalankan dulu model (Run Model) sebelum interpolasi.")
        return

    try:
        xq = float(entry_int_x.get())
        yq = float(entry_int_y.get())
    except ValueError:
        messagebox.showerror("Error", "X dan Y interpolasi harus berupa angka.")
        return

    zq = bilinear_interp(last_X, last_Y, last_gz, xq, yq)
    if zq is None:
        label_int_result.config(text="Di luar area grid.")
    else:
        label_int_result.config(text=f"g_z interpolasi: {zq:.4f} mGal")

# =======================================================================
# FUNGSI UPDATE PETA DAN PROFIL
# =======================================================================
def forward_model():
    global last_X, last_Y, last_gz

    # grid peta LEBIH LUAS biar anomali kelihatan menyebar
    x = np.linspace(-400, 400, 200)
    y = np.linspace(-400, 400, 200)
    X, Y = np.meshgrid(x, y)

    if len(sources) == 0:
        gz_mgal = np.zeros_like(X)
    else:
        gz_mgal = compute_gz_grid(X, Y, sources)

    # --- SMOOTHING TAMBAHAN supaya bentuk tidak terlalu "bola" tajam ---
    # kernel rata-rata besar (misal 21x21)
    smooth_kernel_size = 21
    kernel = np.ones((smooth_kernel_size, smooth_kernel_size), dtype=float)
    kernel = kernel / kernel.sum()
    gz_mgal = convolve2d(gz_mgal, kernel, mode="same", boundary="symm")

    # simpan grid terakhir untuk interpolasi
    last_X = X
    last_Y = Y
    last_gz = gz_mgal

    # PROFIL MODEL (Y = 0)
    mid = len(y) // 2
    profile_x = x
    profile_val = gz_mgal[mid, :]

    # UPDATE TABEL PARAMETER (info umum)
    df = []
    df.append(("Jumlah Sumber", len(sources)))
    df.append(("Ukuran grid", f"{len(x)} x {len(y)}"))
    df.append(("Smoothing kernel", f"{smooth_kernel_size} x {smooth_kernel_size}"))

    table_param.delete(*table_param.get_children())
    for p, v in df:
        table_param.insert("", "end", values=(p, v))

    # PLOTTING
    fig.clear()

    # ---- PETA ANOMALI ----
    ax1 = fig.add_subplot(2, 1, 1)
    p = ax1.contourf(X, Y, gz_mgal, levels=40)
    fig.colorbar(p, ax=ax1, label="g_z (mGal)")
    ax1.set_title("Peta Anomali Gravitasi (g_z) [mGal] (Smoothed)")
    ax1.set_xlabel("X (m) ~ Latitude")
    ax1.set_ylabel("Y (m) ~ Longitude")

    # plot titik sumber + nama
    for i, s in enumerate(sources, start=1):
        ax1.plot(s["x"], s["y"], "ko")
        txt = f"{s['name']} (z={s['z']:.0f} m, ρ={s['rho']:.0f})"
        ax1.text(s["x"] + 5, s["y"] + 5, txt, color="w", fontsize=7)

    if grid_var.get() == 1:
        ax1.grid(True, linestyle="--", alpha=0.4)

    # ---- PROFIL Y=0 ----
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(profile_x, profile_val, linewidth=2)
    ax2.set_title("Profil g_z Sepanjang X (Y=0) – Smoothed")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("g_z (mGal)")
    if grid_var.get() == 1:
        ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    canvas.draw()

# =======================================================================
# FUNGSI INPUT SUMBER (TITIK) DARI USER
# =======================================================================
def add_source():
    try:
        name = entry_name.get().strip()
        x_val = float(entry_x.get())
        y_val = float(entry_y.get())
        z_val = float(entry_z.get())
        rho_val = float(entry_rho.get())
    except ValueError:
        messagebox.showerror("Error", "Semua nilai (X, Y, kedalaman, densitas) harus berupa angka.")
        return

    if name == "":
        name = f"p{len(sources)+1}"

    sources.append({
        "name": name,
        "x": x_val,
        "y": y_val,
        "z": z_val,
        "rho": rho_val
    })
    refresh_table_sources()
    clear_input()

def refresh_table_sources():
    table_sources.delete(*table_sources.get_children())
    for i, s in enumerate(sources, start=1):
        table_sources.insert(
            "",
            "end",
            values=(s["name"], s["x"], s["y"], s["z"], s["rho"])
        )

def clear_input():
    entry_name.delete(0, tk.END)
    entry_x.delete(0, tk.END)
    entry_y.delete(0, tk.END)
    entry_z.delete(0, tk.END)
    entry_rho.delete(0, tk.END)

def delete_selected_source():
    sel = table_sources.selection()
    if not sel:
        return
    idx = table_sources.index(sel[0])
    del sources[idx]
    refresh_table_sources()

def load_selected_to_entry(event=None):
    sel = table_sources.selection()
    if not sel:
        return
    idx = table_sources.index(sel[0])
    s = sources[idx]
    entry_name.delete(0, tk.END); entry_name.insert(0, s["name"])
    entry_x.delete(0, tk.END); entry_x.insert(0, str(s["x"]))
    entry_y.delete(0, tk.END); entry_y.insert(0, str(s["y"]))
    entry_z.delete(0, tk.END); entry_z.insert(0, str(s["z"]))
    entry_rho.delete(0, tk.END); entry_rho.insert(0, str(s["rho"]))

def update_selected_source():
    sel = table_sources.selection()
    if not sel:
        return
    idx = table_sources.index(sel[0])
    try:
        name = entry_name.get().strip()
        x_val = float(entry_x.get())
        y_val = float(entry_y.get())
        z_val = float(entry_z.get())
        rho_val = float(entry_rho.get())
    except ValueError:
        messagebox.showerror("Error", "Semua nilai (X, Y, kedalaman, densitas) harus berupa angka.")
        return

    if name == "":
        name = f"p{idx+1}"

    sources[idx] = {
        "name": name,
        "x": x_val,
        "y": y_val,
        "z": z_val,
        "rho": rho_val
    }
    refresh_table_sources()

# =======================================================================
# GUI TKINTER
# =======================================================================
root = tk.Tk()
root.title("Pemodelan Maju Gravitasi – Multi Sumber (Smoothed)")
root.geometry("1400x800")

frame_left = tk.Frame(root)
frame_left.pack(side="left", fill="y", padx=10, pady=10)

frame_right = tk.Frame(root)
frame_right.pack(side="right", fill="both", expand=True, padx=5, pady=5)

# ------------------ INPUT TITIK SUMBER ------------------
tk.Label(frame_left, text="Input Sumber Gravitasi", font=("Segoe UI", 10, "bold")).pack(anchor="w")

tk.Label(frame_left, text="Nama titik:").pack(anchor="w")
entry_name = tk.Entry(frame_left, width=20)
entry_name.pack(anchor="w")

tk.Label(frame_left, text="X (m) ~ Latitude:").pack(anchor="w")
entry_x = tk.Entry(frame_left, width=20)
entry_x.pack(anchor="w")

tk.Label(frame_left, text="Y (m) ~ Longitude:").pack(anchor="w")
entry_y = tk.Entry(frame_left, width=20)
entry_y.pack(anchor="w")

tk.Label(frame_left, text="Kedalaman z (m, positif ke bawah):").pack(anchor="w")
entry_z = tk.Entry(frame_left, width=20)
entry_z.pack(anchor="w")

tk.Label(frame_left, text="Densitas ρ (kg/m³):").pack(anchor="w")
entry_rho = tk.Entry(frame_left, width=20)
entry_rho.pack(anchor="w", pady=(0, 5))

frame_btn_src = tk.Frame(frame_left)
frame_btn_src.pack(anchor="w", pady=5)
tk.Button(frame_btn_src, text="Tambah", width=10, command=add_source).grid(row=0, column=0, padx=2, pady=2)
tk.Button(frame_btn_src, text="Update", width=10, command=update_selected_source).grid(row=0, column=1, padx=2, pady=2)
tk.Button(frame_btn_src, text="Hapus", width=10, command=delete_selected_source).grid(row=0, column=2, padx=2, pady=2)

# ------------------ TABEL SUMBER ------------------
tk.Label(frame_left, text="Daftar Sumber").pack(anchor="w", pady=(8, 0))
table_sources = ttk.Treeview(frame_left,
                             columns=("name", "x", "y", "z", "rho"),
                             show="headings", height=7)
for col, txt in zip(("name", "x", "y", "z", "rho"),
                    ("Nama", "X (m)", "Y (m)", "z (m)", "ρ (kg/m³)")):
    table_sources.heading(col, text=txt)
    table_sources.column(col, width=80)
table_sources.pack(pady=5, fill="x")
table_sources.bind("<<TreeviewSelect>>", load_selected_to_entry)

# ------------------ Opsi Plot & Parameter Ringkas ------------------
grid_var = tk.IntVar(value=1)
tk.Checkbutton(frame_left, text="Grid pada plot", variable=grid_var).pack(anchor="w")

tk.Button(frame_left, text="Run Model", width=20, command=forward_model).pack(pady=10)

tk.Label(frame_left, text="Ringkasan", font=("Segoe UI", 9, "bold")).pack(anchor="w")
table_param = ttk.Treeview(frame_left, columns=("Param", "Value"),
                           show="headings", height=4)
table_param.heading("Param", text="Parameter")
table_param.heading("Value", text="Value")
table_param.pack(pady=5, fill="x")

# ------------------ FORM INTERPOLASI ------------------
tk.Label(frame_left, text="Interpolasi g_z pada titik (X,Y)", font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(10, 0))

frame_int = tk.Frame(frame_left)
frame_int.pack(anchor="w", pady=5)

tk.Label(frame_int, text="X (m):").grid(row=0, column=0, sticky="w")
entry_int_x = tk.Entry(frame_int, width=10)
entry_int_x.grid(row=0, column=1, padx=3)

tk.Label(frame_int, text="Y (m):").grid(row=1, column=0, sticky="w")
entry_int_y = tk.Entry(frame_int, width=10)
entry_int_y.grid(row=1, column=1, padx=3)

tk.Button(frame_int, text="Hitung", command=hitung_interpolasi, width=10).grid(row=2, column=0, columnspan=2, pady=4)

label_int_result = tk.Label(frame_left, text="g_z interpolasi: -", justify="left")
label_int_result.pack(anchor="w")

# ------------------ KANVAS MATPLOTLIB ------------------
fig = plt.Figure(figsize=(8, 6))
canvas = FigureCanvasTkAgg(fig, master=frame_right)
canvas.get_tk_widget().pack(fill="both", expand=True)

root.mainloop()
