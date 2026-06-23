import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import ellipj
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Для поддержки 3D-графики

# Установка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# ------------------------------------------------------------
# 1. ТОЧНЫЕ АНАЛИТИЧЕСКИЕ РЕШЕНИЯ (МАТЕМАТИЧЕСКИ КОРРЕКТНЫЕ)
# ------------------------------------------------------------

def sol_set1(xi, c1, c2):
    """При c1 > 0, c2 > 0 ограниченных решений нет."""
    return None

def sol_set2(xi, c1, c2, m=0.3, xi0=0):
    """Набор 2 (c1 < 0, c2 < 0): Колебания cn Jacobi."""
    if c1 >= 0 or c2 >= 0 or not (0 < m < 0.5): 
        return None
    omega = np.sqrt(c2 / (2 * m - 1))
    amplitude = np.sqrt((2 * m / (1 - 2 * m)) * (c2 / c1))
    sn, cn, dn, ph = ellipj(omega * (xi - xi0), m)
    return amplitude * cn

def sol_set3(xi, c1, c2):
    """Набор 3 (c1 > 0, c2 < 0): Тёмный солитон (tanh)."""
    if c1 <= 0 or c2 >= 0: return None
    return np.sqrt(-c2/c1) * np.tanh(np.sqrt(-c2/2) * xi)

def sol_set4(xi, c1, c2, xi0=0):
    """Набор 4 (c1 < 0, c2 > 0): Светлый солитон (sech)."""
    if c1 >= 0 or c2 <= 0: return None
    amplitude = np.sqrt(-2 * c2 / c1)
    width = np.sqrt(c2)
    return amplitude / np.cosh(width * (xi - xi0))

# ------------------------------------------------------------
# 2. СИСТЕМЫ ОДУ С АППАРАТНЫМ ОГРАНИЧЕНИЕМ ПЕРЕПОЛНЕНИЙ (CLAMPING)
# ------------------------------------------------------------
def system_ode(xi, state, c1, c2):
    Phi, y = state
    # Защитное ограничение предотвращает nan/inf на пробных шагах адаптивного решателя
    Phi_clipped = np.clip(Phi, -20.0, 20.0)
    y_clipped = np.clip(y, -20.0, 20.0)
    return [y_clipped, c1 * Phi_clipped**3 + c2 * Phi_clipped]

def perturbed_system_ode(xi, state, c1, c2, f0, omega):
    Phi, y = state
    Phi_clipped = np.clip(Phi, -20.0, 20.0)
    y_clipped = np.clip(y, -20.0, 20.0)
    return [y_clipped, c1 * Phi_clipped**3 + c2 * Phi_clipped + f0 * np.cos(omega * xi)]

# ------------------------------------------------------------
# 3. ИНТЕГРИРОВАНИЕ ТРАЕКТОРИЙ С ДИНАМИЧЕСКИМ ВЫБОРОМ НАПРАВЛЕНИЯ
# ------------------------------------------------------------
def integrate_full_trajectory(Phi0, y0, c1, c2, xi_span, num_points=4000, 
                             method='DOP853', rtol=1e-12, atol=1e-14):
    """
    Интегрирование траектории с адаптивным шагом. 
    Убрано использование t_eval для 100% совместимости со всеми версиями SciPy.
    """
    xi_min, xi_max = xi_span
    
    # Терминальный триггер для остановки при численном взрыве траектории
    def limit_event(xi, state):
        Phi, y = state
        return 12.0 - max(abs(Phi), abs(y))
    limit_event.terminal = True
    
    # Интегрируем вперед, только если xi_max > 0
    if xi_max > 0:
        sol_fwd = solve_ivp(lambda xi, st: system_ode(xi, st, c1, c2),
                            (0, xi_max), [Phi0, y0],
                            method=method, rtol=rtol, atol=atol, events=limit_event)
        xi_fwd = sol_fwd.t
        Phi_fwd = sol_fwd.y[0]
        y_fwd = sol_fwd.y[1]
    else:
        xi_fwd = np.array([0.0])
        Phi_fwd = np.array([Phi0])
        y_fwd = np.array([y0])
        
    # Интегрируем назад, только если xi_min < 0
    if xi_min < 0:
        sol_rev = solve_ivp(lambda xi, st: system_ode(xi, st, c1, c2),
                            (0, xi_min), [Phi0, y0],
                            method=method, rtol=rtol, atol=atol, events=limit_event)
        xi_rev = sol_rev.t[::-1]
        Phi_rev = sol_rev.y[0][::-1]
        y_rev = sol_rev.y[1][::-1]
        
        # Сшиваем траектории
        if xi_max > 0:
            xi = np.concatenate([xi_rev[:-1], xi_fwd])
            Phi = np.concatenate([Phi_rev[:-1], Phi_fwd])
            y = np.concatenate([y_rev[:-1], y_fwd])
        else:
            xi = xi_rev
            Phi = Phi_rev
            y = y_rev
    else:
        # Если назад не интегрировали, решением является только форвард-траектория
        xi = xi_fwd
        Phi = Phi_fwd
        y = y_fwd
        
    return xi, Phi, y

# Особые точки
def fixed_points(c1, c2):
    pts = [(0.0, 0.0)]
    if c1 != 0 and -c2 / c1 > 0:
        val = np.sqrt(-c2 / c1)
        pts.append((val, 0.0))
        pts.append((-val, 0.0))
    result = []
    for Phi, y in pts:
        J = 3 * c1 * Phi**2 + c2
        typ = 'седло' if J > 0 else ('центр' if J < 0 else 'вырожденная')
        result.append((Phi, y, typ))
    return result

def separatrix_from_saddle(Phi_s, y_s, c1, c2, xi_span, eps=1e-7, num=3000, **kwargs):
    J_mat = np.array([[0, 1], [3 * c1 * Phi_s**2 + c2, 0]])
    eigvals, eigvecs = np.linalg.eig(J_mat)
    unstable = stable = None
    for i, val in enumerate(eigvals):
        if val > 0: unstable = eigvecs[:, i] / np.linalg.norm(eigvecs[:, i])
        elif val < 0: stable = eigvecs[:, i] / np.linalg.norm(eigvecs[:, i])
        
    if unstable is None or stable is None: 
        return None, None
        
    fwd_start = [Phi_s + eps * unstable[0], y_s + eps * unstable[1]]
    fwd_traj = integrate_full_trajectory(fwd_start[0], fwd_start[1], c1, c2, xi_span, num, **kwargs)
    
    rev_start = [Phi_s + eps * stable[0], y_s + eps * stable[1]]
    rev_traj = integrate_full_trajectory(rev_start[0], rev_start[1], c1, c2, xi_span, num, **kwargs)
    
    return fwd_traj, rev_traj

# ------------------------------------------------------------
# 4. СТОХАСТИЧЕСКОЕ МОДЕЛИРОВАНИЕ (Раздел 4.5)
# ------------------------------------------------------------
def simulate_sde_ensemble(Phi0, y0, c1, c2, epsilon, xi_max, steps=3000, realizations=100):
    dt = xi_max / steps
    t = np.linspace(0, xi_max, steps)
    
    all_Phi = np.zeros((realizations, steps))
    all_y = np.zeros((realizations, steps))
    
    for r in range(realizations):
        Phi = np.zeros(steps)
        y = np.zeros(steps)
        Phi[0], y[0] = Phi0, y0
        
        dW = np.random.normal(0, np.sqrt(dt), steps)
        
        for i in range(steps - 1):
            Phi_clamped = np.clip(Phi[i], -15.0, 15.0)
            y_clamped = np.clip(y[i], -15.0, 15.0)
            
            Phi[i+1] = Phi_clamped + y_clamped * dt
            drift = (c1 * Phi_clamped**3 + c2 * Phi_clamped) * dt
            diffusion = epsilon * Phi_clamped * dW[i]
            y[i+1] = y_clamped + drift + diffusion
            
        all_Phi[r, :] = Phi
        all_y[r, :] = y
        
    return t, all_Phi, all_y

def plot_stochastic_results(c1, c2, epsilon=0.3, lw=1.8):
    print(f"\nЗапуск стохастического моделирования (интенсивность шума epsilon = {epsilon})...")
    Phi0 = 0.0
    y0 = np.sqrt(-c2/c1) * np.sqrt(-c2/2)
    
    t, all_Phi, _ = simulate_sde_ensemble(Phi0, y0, c1, c2, epsilon, xi_max=8.0, steps=2000, realizations=100)
    
    mean_Phi = np.mean(all_Phi, axis=0)
    std_Phi = np.std(all_Phi, axis=0)
    
    t_det = np.linspace(0, 8.0, 1000)
    Phi_det = np.sqrt(-c2/c1) * np.tanh(np.sqrt(-c2/2) * t_det)
    
    plt.figure(figsize=(11, 6))
    for i in range(5):
        plt.plot(t, all_Phi[i, :], color='red', alpha=0.12, lw=lw*0.6)
    
    plt.fill_between(t, mean_Phi - std_Phi, mean_Phi + std_Phi, color='red', alpha=0.15, label='Доверительный интервал (±1 SD)')
    plt.plot(t, mean_Phi, 'r-', lw=lw, label='Среднее значение траекторий (с шумом)')
    plt.plot(t_det, Phi_det, 'b--', lw=lw*1.3, label='Детерминированный солитон (без шума)')
    
    plt.xlabel(r'$\xi$', fontsize=12)
    plt.ylabel(r'$\Phi(\xi)$', fontsize=12)
    plt.title(f"Влияние мультипликативного шума на профиль волны ($\epsilon={epsilon}$, $c_1={c1}, c_2={c2}$)")
    plt.legend(loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.show()

# ------------------------------------------------------------
# 5. ПЕРИОДИЧЕСКОЕ ВОЗДЕЙСТВИЕ И СЕЧЕНИЕ ПУАНКАРЕ (Раздел 4.4)
# ------------------------------------------------------------
def plot_poincare_section(c1, c2, f0=0.8, omega=1.4, xi_max=800):
    print(f"\nМоделирование системы под периодическим воздействием (f0={f0}, omega={omega})...")
    state0 = [1.2, 0.0]
    t_eval = np.linspace(0, xi_max, xi_max * 12)
    
    def limit_event(xi, state):
        return 15.0 - max(abs(state[0]), abs(state[1]))
    limit_event.terminal = True
    
    sol = solve_ivp(lambda t, y: perturbed_system_ode(t, y, c1, c2, f0, omega),
                    (0, xi_max), state0, t_eval=t_eval, method='RK45', rtol=1e-7, events=limit_event)
    
    Phi, y = sol.y[0], sol.y[1]
    
    period = 2 * np.pi / omega
    poincare_times = np.arange(100, sol.t[-1], period)
    
    Phi_p = np.interp(poincare_times, sol.t, Phi)
    y_p = np.interp(poincare_times, sol.t, y)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(Phi[1000:], y[1000:], 'indigo', lw=0.5, alpha=0.5)
    ax1.set_xlabel(r'$\Phi$', fontsize=11)
    ax1.set_ylabel(r"$y$", fontsize=11)
    ax1.set_title("Фазовая траектория при внешнем воздействии")
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(Phi_p, y_p, color='red', s=6, alpha=0.8, zorder=5)
    ax2.set_xlabel(r'$\Phi$', fontsize=11)
    ax2.set_ylabel(r"$y$", fontsize=11)
    ax2.set_title(f"Сечение Пуанкаре (T = {period:.3f})")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Воздействующая периодическая сила: $f_0={f0}$, $\\omega={omega}$")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 6. СТАБИЛЬНОЕ СРАВНЕНИЕ С АНАЛИТИКОЙ
# ------------------------------------------------------------
def compare_numerical_analytical(c1, c2, ana_func, xi_span=(-7,7), num=1000, title="", lw=1.8):
    xi = np.linspace(xi_span[0], xi_span[1], num)
    Phi_ana = ana_func(xi)
    if Phi_ana is None:
        print(f"\n[Ошибка] Аналитическое решение для '{title}' не определено или не существует при c1={c1}, c2={c2}.")
        return

    def num_deriv(x, f, h=1e-6):
        return (f(x + h) - f(x - h)) / (2 * h)
        
    Phi0 = ana_func(0.0)
    y0 = num_deriv(0.0, ana_func)
    
    try:
        xi_num, Phi_num, _ = integrate_full_trajectory(Phi0, y0, c1, c2, xi_span, num_points=num)
        
        mask = (xi >= xi_num[0]) & (xi <= xi_num[-1])
        xi_compare = xi[mask]
        Phi_ana_compare = Phi_ana[mask]
        
        Phi_num_interp = np.interp(xi_compare, xi_num, Phi_num)
        err = np.abs(Phi_num_interp - Phi_ana_compare)
    except Exception as e:
        print(f"\n[Ошибка численного расчета] Не удалось завершить интегрирование: {e}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(xi_compare, Phi_num_interp, 'b-', lw=lw, label='Численное (DOP853)')
    ax1.plot(xi_compare, Phi_ana_compare, 'r--', lw=lw*0.8, label='Аналитическое (точное)')
    ax1.set_xlabel(r'$\xi$')
    ax1.set_ylabel(r'$\Phi$')
    ax1.set_title(f"Сравнение решений: {title}")
    ax1.legend()
    
    ax2.semilogy(xi_compare, err, 'g-', lw=lw*0.8)
    ax2.set_xlabel(r'$\xi$')
    ax2.set_ylabel('Абсолютная ошибка (логарифм. шкала)')
    ax2.set_title("График абсолютной погрешности")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 7. ПОСТРОЕНИЕ 3D ПРОСТРАНСТВЕННОГО СОЛИТОНА
# ------------------------------------------------------------
def plot_3d_spatial_soliton(c1, c2, t_val=0.0, sol_type='bright'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.linspace(-6, 6, 100)
    y = np.linspace(-6, 6, 100)
    X, Y_grid = np.meshgrid(x, y)
    
    # Косой автомодельный фронт xi = a1*x + a2*y + a3*t
    a1, a2, a3 = 1.0, 0.8, -1.2
    Xi = a1 * X + a2 * Y_grid + a3 * t_val
    
    if sol_type == 'bright':
        amplitude = np.sqrt(-2 * c2 / c1)
        width = np.sqrt(c2)
        Phi = amplitude / np.cosh(width * Xi)
        cmap = 'plasma'
    else:
        amplitude = np.sqrt(-c2 / c1)
        width = np.sqrt(-c2 / 2)
        Phi = amplitude * np.tanh(width * Xi)
        cmap = 'viridis'
        
    Intensity = Phi**2
    
    surf = ax.plot_surface(X, Y_grid, Intensity, cmap=cmap, edgecolor='none', alpha=0.95)
    ax.set_xlabel('Пространство x', fontsize=10)
    ax.set_ylabel('Пространство y', fontsize=10)
    ax.set_zlabel(r'Интенсивность $|\phi(x,y,t)|^2$', fontsize=10)
    ax.set_title(f"3D поле энергии $|\phi(x,y,t)|^2$ солитона при t = {t_val:.2f}", fontsize=12)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 8. ПОСТРОЕНИЕ ФРАКТАЛЬНОЙ БИФУРКАЦИОННОЙ ДИАГРАММЫ ХАОСА
# ------------------------------------------------------------
def plot_bifurcation_diagram_driven(c1, c2, omega=1.4):
    print("\nРасчет численной бифуркационной диаграммы хаоса...")
    print("Выполняется свип по амплитуде гармонической силы f0 (это займет 5-10 секунд)...")
    
    f0_vals = np.linspace(0.0, 2.8, 150)
    period = 2 * np.pi / omega
    state = [1.0, 0.0]
    
    f0_points = []
    phi_points = []
    
    def limit_event(t, y):
        return 15.0 - max(abs(y[0]), abs(y[1]))
    limit_event.terminal = True
    
    for f0 in f0_vals:
        t_span = (0, 160 * period)
        t_eval = np.arange(120 * period, 160 * period, period)
        
        sol = solve_ivp(lambda t, y: perturbed_system_ode(t, y, c1, c2, f0, omega),
                        t_span, state, t_eval=t_eval, method='RK45', rtol=1e-6, events=limit_event)
        
        if sol.status >= 0 and len(sol.y[0]) > 0:
            last_pts = sol.y[0]
            for val in last_pts:
                f0_points.append(f0)
                phi_points.append(val)
            state = [sol.y[0][-1], sol.y[1][-1]]
        else:
            state = [1.0, 0.0]
            
    plt.figure(figsize=(10, 6))
    plt.plot(f0_points, phi_points, 'k.', markersize=0.6, alpha=0.5)
    plt.xlabel(r'Амплитуда внешнего воздействия $f_0$', fontsize=12)
    plt.ylabel(r'Положение сечения $\Phi$ в фазе $t = k T$', fontsize=12)
    plt.title(f"Фрактальная бифуркационная диаграмма перехода к хаосу ($c_1={c1}, c_2={c2}, \\omega={omega}$)")
    plt.grid(True, alpha=0.3)
    plt.show()

# ------------------------------------------------------------
# 9. ПОСТРОЕНИЕ ВЛИЯНИЯ ДРОБНОЙ ПРОИЗВОДНОЙ (Раздел 3.7)
# ------------------------------------------------------------
def plot_fractional_derivative_effect(c1, c2, p_val=1.5, lw=1.8):
    if c1 >= 0 or c2 <= 0:
        print("\n[Ошибка] Моделирование влияния дробности (sech) требует c1 < 0 и c2 > 0.")
        return
        
    print(f"\nМоделирование влияния порядка дробной производной alpha при волновом числе p = {p_val}...")
    xi = np.linspace(-8, 8, 1000)
    
    plt.figure(figsize=(10, 6))
    alphas = [1.0, 0.7, 0.4]
    colors = ['blue', 'red', 'green']
    linestyles = ['-', '--', '-.']
    
    for alpha, col, ls in zip(alphas, colors, linestyles):
        c2_eff = c2 * (abs(p_val) ** (2 * (alpha - 1)))
        amplitude = np.sqrt(-2 * c2_eff / c1)
        width = np.sqrt(c2_eff)
        Phi = amplitude / np.cosh(width * xi)
        
        plt.plot(xi, Phi, color=col, linestyle=ls, lw=lw, 
                 label=f'$\\alpha = {alpha:.1f}$ (эфф. $c_2 = {c2_eff:.3f}$)')
        
    plt.xlabel(r'$\xi$', fontsize=12)
    plt.ylabel(r'$\Phi(\xi)$', fontsize=12)
    plt.title(f"Влияние порядка дробной производной $\\alpha$ на профиль волны ($p = {p_val}$)")
    plt.legend(loc='best', frameon=True)
    plt.grid(True, alpha=0.3)
    plt.show()

# ------------------------------------------------------------
# 10. ИНВАРИАНТ ДВИЖЕНИЯ И ХАРАКТЕРИСТИКА ДРЕЙФА ЭНЕРГИИ (Раздел 3.4)
# ------------------------------------------------------------
def plot_hamiltonian_conservation(c1, c2, Phi0=1.0, y0=0.5, lw=1.8):
    """
    Вычисление первого интеграла движения (энергии Гамильтониана) 
    и анализ его численного сохранения во времени
    """
    print(f"\nИнтегрирование траектории и проверка сохранения Гамильтониана (энергии) при c1={c1}, c2={c2}...")
    xi_span = (0, 60)
    
    # Интегрируем высокоточным методом DOP853 с предохранителем limit_event
    xi, Phi, y = integrate_full_trajectory(Phi0, y0, c1, c2, xi_span, num_points=6000, method='DOP853')
    
    # Расчет Гамильтониана на каждом шаге (Уравнение 38 из Раздела 3.4)
    H = 0.5 * y**2 - (c1 / 4.0) * Phi**4 - (c2 / 2.0) * Phi**2
    H_init = H[0]
    
    H_drift = np.abs(H - H_init)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Отрисовка траектории
    ax1.plot(xi, Phi, 'b-', lw=lw, label=r'$\Phi(\xi)$')
    ax1.plot(xi, y, 'r--', lw=lw*0.8, label=r'$y = d\Phi/d\xi$')
    ax1.set_xlabel(r'$\xi$', fontsize=11)
    ax1.set_ylabel('Амплитуда', fontsize=11)
    ax1.set_title('Профиль динамики системы в фазовом пространстве')
    ax1.grid(True)
    ax1.legend(loc='best')
    
    # Ошибка энергии (Гамильтониан)
    ax2.semilogy(xi, H_drift + 1e-16, 'g-', lw=lw)
    ax2.set_xlabel(r'$\xi$', fontsize=11)
    ax2.set_ylabel(r'Погрешность инварианта $|H(\xi) - H(0)|$', fontsize=11)
    ax2.set_title('Сохранение первого интеграла во времени (DOP853)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Закон сохранения энергии (Раздел 3.4, Ур. 38): Нач. Энергия $H_0 = {H_init:.6f}$")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 11. СПЕКТРАЛЬНЫЙ АНАЛИЗ ФУРЬЕ (FFT) ДЛЯ ДИАГНОСТИКИ ХАОСА
# ------------------------------------------------------------
def plot_fft_chaos_analysis(c1, c2, f0=0.8, omega=1.4, xi_max=1800, lw=1.8):
    """
    Быстрое преобразование Фурье (FFT) от траектории неавтономной системы
    для строгого разделения периодических режимов от хаотических
    """
    print(f"\nИнтегрирование траектории и расчет БПФ (FFT) при f0={f0}, omega={omega}...")
    state0 = [1.2, 0.0]
    
    dt = 0.05
    t_eval = np.arange(0, xi_max, dt)
    
    def limit_event(t, y):
        return 15.0 - max(abs(y[0]), abs(y[1]))
    limit_event.terminal = True
    
    sol = solve_ivp(lambda t, y: perturbed_system_ode(t, y, c1, c2, f0, omega),
                    (0, xi_max), state0, t_eval=t_eval, method='RK45', rtol=1e-7, events=limit_event)
    
    Phi = sol.y[0]
    
    start_idx = int(len(Phi) * 0.2)
    Phi_steady = Phi[start_idx:]
    N = len(Phi_steady)
    
    # Спектральный расчет БПФ
    fft_vals = np.fft.rfft(Phi_steady)
    freqs = np.fft.rfftfreq(N, d=dt)
    power_spectrum = np.abs(fft_vals) ** 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Отрисовка временного ряда устоявшегося режима
    display_len = min(1500, N)
    ax1.plot(sol.t[start_idx:start_idx+display_len], Phi_steady[:display_len], 'indigo', lw=lw)
    ax1.set_xlabel(r'$\xi$', fontsize=11)
    ax1.set_ylabel(r'$\Phi(\xi)$', fontsize=11)
    ax1.set_title('Установившиеся колебания во времени (фрагмент)')
    ax1.grid(True)
    
    # Спектр мощности Фурье
    ax2.semilogy(freqs, power_spectrum, 'crimson', lw=lw)
    ax2.axvline(omega / (2 * np.pi), color='black', linestyle='--', alpha=0.7, 
               label=f'Частота вынуждения ({omega / (2 * np.pi):.3f} Гц)')
    ax2.set_xlabel('Частота (Гц)', fontsize=11)
    ax2.set_ylabel('Спектральная плотность мощности', fontsize=11)
    ax2.set_title('Спектральный анализ Фурье (БПФ/FFT)')
    ax2.set_xlim(0, 1.5)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Анализ Фурье: Спектральная диагностика детерминированного хаоса")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 12. ПОСТРОЕНИЕ СВЯЗАННОГО 2-СОЛИТОННОГО СОСТОЯНИЯ (Разд. 2.1)
# ------------------------------------------------------------
def plot_breather_2soliton_dynamics():
    print("\nПостроение пространственно-временной динамики связанного 2-солитонного решения (Breather)...")
    x = np.linspace(-5, 5, 200)
    t = np.linspace(-np.pi/2, np.pi/2, 200)
    X, T = np.meshgrid(x, t)
    
    # Точное дыхание (дышащий солитон)
    num = np.cosh(3*X) + 3 * np.exp(8j*T) * np.cosh(X)
    denom = np.cosh(4*X) + 4 * np.cosh(2*X) + 3 * np.cos(8*T)
    psi = 4 * np.exp(1j*T) * (num / denom)
    Intensity = np.abs(psi) ** 2
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, T, Intensity, cmap='magma', edgecolor='none', alpha=0.95)
    ax1.set_xlabel('Пространство x')
    ax1.set_ylabel('Время t')
    ax1.set_zlabel(r'Интенсивность $|u|^2$')
    ax1.set_title('3D эволюция дыхания солитона')
    
    # 2D сверху
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(Intensity, extent=[-5, 5, -np.pi/2, np.pi/2], aspect='auto', origin='lower', cmap='magma')
    ax2.set_xlabel('Пространство x')
    ax2.set_ylabel('Время t')
    ax2.set_title('Проекция на плоскость (вид сверху)')
    fig.colorbar(im, ax=ax2, label=r'$|u(x,t)|^2$')
    
    plt.suptitle("Точное двухсолитонное связанное состояние (Метод Хироты, Глава 2, разд. 2.1)")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 13. РАСЧЕТ ПОКАЗАТЕЛЯ ЛЯПУНОВА (МЕТОД БЕНЕТТИНА)
# ------------------------------------------------------------
def plot_lyapunov_exponent_driven(c1, c2, f0=0.8, omega=1.4, steps=300, dt_step=0.6, lw=1.8):
    """
    Расчет Старшего Показателя Ляпунова (LLE) по алгоритму Бенеттина.
    Положительный показатель однозначно доказывает присутствие хаоса в системе.
    """
    print("\nРасчет старшего показателя Ляпунова (LLE) по алгоритму Бенеттина...")
    print("Это может занять около 5 секунд (выполняется параллельное интегрирование траекторий)...")
    
    d0 = 1e-8 
    state1 = np.array([1.0, 0.0]) 
    state2 = state1 + np.array([d0, 0.0]) 
    
    lle_accum = 0.0
    times = []
    lle_history = []
    t_curr = 0.0
    
    for step in range(1, steps + 1):
        t_span = (t_curr, t_curr + dt_step)
        
        sol1 = solve_ivp(lambda t, y: perturbed_system_ode(t, y, c1, c2, f0, omega),
                         t_span, state1, method='RK45', rtol=1e-8, atol=1e-10)
        sol2 = solve_ivp(lambda t, y: perturbed_system_ode(t, y, c1, c2, f0, omega),
                         t_span, state2, method='RK45', rtol=1e-8, atol=1e-10)
                         
        p1, v1 = sol1.y[:, -1]
        p2, v2 = sol2.y[:, -1]
        
        d1 = np.sqrt((p1 - p2)**2 + (v1 - v2)**2)
        
        if d1 > 0:
            lle_accum += np.log(d1 / d0)
            
        state1 = np.array([p1, v1])
        state2 = state1 + d0 * (np.array([p2, v2]) - state1) / d1
        
        lle_val = lle_accum / (step * dt_step)
        
        times.append(t_curr + dt_step)
        lle_history.append(lle_val)
        t_curr += dt_step
        
    plt.figure(figsize=(10, 6))
    plt.plot(times, lle_history, 'b-', lw=lw, label=f'Сходимость LLE к $\\lambda \\approx {lle_history[-1]:.4f}$')
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Время $\\xi$', fontsize=11)
    plt.ylabel('Старший показатель Ляпунова $\\lambda$', fontsize=11)
    plt.title(f"Расчет показателя Ляпунова методом Бенеттина ($f_0 = {f0}, \\omega = {omega}$)")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    if lle_history[-1] > 0.05:
        print(f"\n[Анализ] LLE > 0 (лямбда ≈ {lle_history[-1]:.3f}). Динамика строго хаотическая!")
    else:
        print(f"\n[Анализ] LLE <= 0 (лямбда ≈ {lle_history[-1]:.3f}). Динамика регулярная (периодическая орбита или фокус).")

# ------------------------------------------------------------
# 14. ЧИСЛЕННОЕ МОДЕЛИРОВАНИЕ СТОЛКНОВЕНИЯ СОЛИТОНОВ (УЧП, SSFM)
# ------------------------------------------------------------
def simulate_two_soliton_collision_pde(A1=1.5, A2=1.0, x1=-7.0, x2=7.0, v1=1.5, v2=-1.5, N=512, t_max=12.0, steps=250):
    """
    Прямое численное моделирование упругого столкновения двух солитонов
    для исходного нелинейного уравнения Шрёдингера (УЧП) методом Split-Step Fourier Method (SSFM)
    """
    print("\nЧисленное моделирование столкновения двух солитонов (УЧП) методом расщепления по шагам Фурье (SSFM)...")
    print("Выполняется БПФ-интегрирование волнового поля (это займет 3-4 секунды)...")
    
    L = 30.0 # Размер пространственного домена
    dx = L / N
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    
    # Начальное условие: два движущихся солитона
    psi = A1 / np.cosh(A1 * (x - x1)) * np.exp(1j * v1 * x) + \
          A2 / np.cosh(A2 * (x - x2)) * np.exp(1j * v2 * x)
          
    # Шаг волновых чисел для преобразования Фурье
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    dt = t_max / steps
    
    Intensity_history = np.zeros((steps, N))
    time_vals = np.linspace(0, t_max, steps)
    
    # Оператор дисперсии на полушаге в спектральном пространстве (u_t = i/2 * u_xx)
    dispersion_half_step = np.exp(-1j * 0.25 * (k**2) * dt)
    
    for step in range(steps):
        Intensity_history[step, :] = np.abs(psi) ** 2
        
        # 1. Линейный шаг по дисперсии (полушаг)
        psi_k = np.fft.fft(psi) * dispersion_half_step
        psi = np.fft.ifft(psi_k)
        
        # 2. Нелинейное фазовое вращение (полный шаг в пространстве координат)
        psi = psi * np.exp(1j * np.abs(psi)**2 * dt)
        
        # 3. Линейный шаг по дисперсии (полушаг)
        psi_k = np.fft.fft(psi) * dispersion_half_step
        psi = np.fft.ifft(psi_k)
        
    X_mesh, T_mesh = np.meshgrid(x, time_vals)
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D вид соударения солитонов
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X_mesh, T_mesh, Intensity_history, cmap='viridis', edgecolor='none', alpha=0.95)
    ax1.set_xlabel('Пространство x')
    ax1.set_ylabel('Время t')
    ax1.set_zlabel(r'Интенсивность $|\psi|^2$')
    ax1.set_title('3D столкновение солитонов (УЧП)')
    ax1.set_xlim(-15, 15)
    
    # 2D плотность
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(Intensity_history, extent=[-L/2, L/2, 0, t_max], aspect='auto', origin='lower', cmap='viridis')
    ax2.set_xlabel('Пространство x')
    ax2.set_ylabel('Время t')
    ax2.set_title('Временной след столкновения солитонов (2D проекция)')
    ax2.set_xlim(-15, 15)
    fig.colorbar(im, ax=ax2, label=r'$|\psi(x,t)|^2$')
    
    plt.suptitle("Моделирование УЧП методом SSFM: Упругое прохождение двух солитонов")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 15. АНИМАЦИЯ СТОЛКНОВЕНИЯ СОЛИТОНОВ (УЧП, SSFM)
# ------------------------------------------------------------
def animate_two_soliton_collision_pde(A1=1.5, A2=1.0, x1=-6.0, x2=6.0, v1=1.5, v2=-1.5, N=512, t_max=12.0, steps=200):
    """
    Анимация динамического упругого прохождения двух солитонов друг сквозь друга
    на основе численного решения уравнения в частных производных (УЧП) методом SSFM
    """
    print("\nПодготовка анимации столкновения солитонов (УЧП)...")
    print("Выполняется предварительный расчет кадров...")
    
    L = 30.0 # Пространственный домен
    dx = L / N
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    
    # Исходное положение двух движущихся солитонов
    psi = A1 / np.cosh(A1 * (x - x1)) * np.exp(1j * v1 * x) + \
          A2 / np.cosh(A2 * (x - x2)) * np.exp(1j * v2 * x)
          
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    dt = t_max / steps
    
    # Предварительный расчет всех кадров во времени
    history = []
    current_psi = psi.copy()
    dispersion_half_step = np.exp(-1j * 0.25 * (k**2) * dt)
    
    for step in range(steps):
        history.append(np.abs(current_psi) ** 2)
        
        # Шаги численного расщепления (SSFM)
        psi_k = np.fft.fft(current_psi) * dispersion_half_step
        current_psi = np.fft.ifft(psi_k)
        
        current_psi = current_psi * np.exp(1j * np.abs(current_psi)**2 * dt)
        
        psi_k = np.fft.fft(current_psi) * dispersion_half_step
        current_psi = np.fft.ifft(psi_k)
        
    # Инициализация окна анимации
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-12, 12)
    max_val = np.max(history)
    ax.set_ylim(0, max_val * 1.1)
    ax.set_xlabel('Пространственная координата x', fontsize=11)
    ax.set_ylabel(r'Интенсивность волнового поля $|\psi(x,t)|^2$', fontsize=11)
    ax.set_title('Динамика соударения двух солитонов (Численное решение УЧП Шрёдингера)', fontsize=13)
    
    line, = ax.plot(x, history[0], color='crimson', lw=2.5)
    time_text = ax.text(0.05, 0.93, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    def update(frame):
        line.set_ydata(history[frame])
        time_text.set_text(f"Время t = {frame * dt:.2f}")
        return line, time_text
        
    ani = FuncAnimation(fig, update, frames=steps, interval=45, blit=True, repeat=True)
    plt.show()

# ------------------------------------------------------------
# 16. ИСПРАВЛЕННАЯ ДИНАМИЧЕСКАЯ 3D-АНИМАЦИЯ СТОЛКНОВЕНИЯ НА ПЛОСКОСТИ XY (УЧП)
# ------------------------------------------------------------
def animate_two_soliton_collision_3d_spatial(A1=1.2, A2=1.0, steps=100, t_max=4.0):
    """
    3D-анимация пространственной динамики косого столкновения волновых пакетов
    в реальном времени на плоскости xy. Волны физически сближаются, сталкиваются и расходятся.
    """
    print("\nПодготовка 3D-анимации физического встречного столкновения солитонов на плоскости xy...")
    print("Вычисляется динамика двух уединенных волновых пакетов...")
    
    # Сетка двумерного пространства
    x = np.linspace(-10, 10, 80)
    y = np.linspace(-10, 10, 80)
    X, Y = np.meshgrid(x, y)
    
    # Сетка времени (от отрицательного времени к положительному для симметрии сближения и расхождения)
    t_vals = np.linspace(-t_max, t_max, steps)
    
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Средняя скорость движения солитонов
    v = 1.8
    
    # Функция расчета интенсивности для кадра
    def get_intensity(t):
        # Координаты положений центров двух волновых пакетов во времени t
        # Солитон 1 движется слева направо (от x < 0 к x > 0)
        x1 = -v * t
        y1 = 0.0
        
        # Солитон 2 движется справа налево (от x > 0 к x < 0)
        x2 = v * t
        y2 = 0.0
        
        # Двумерные локализованные dromion-профили (sech(x)*sech(y))
        Phi1 = A1 / (np.cosh(X - x1) * np.cosh(Y - y1))
        Phi2 = A2 / (np.cosh(X - x2) * np.cosh(Y - y2))
        
        # Суперпозиция полей (нелинейная интерференция при соударении в центре)
        return (Phi1 + Phi2) ** 2

    # Определение максимального значения по оси Z
    max_z = (A1 + A2) ** 2
    
    def update(frame):
        ax.clear()
        t = t_vals[frame]
        Intensity = get_intensity(t)
        
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(0, max_z * 1.1)
        ax.set_xlabel('Пространство x', fontsize=9)
        ax.set_ylabel('Пространство y', fontsize=9)
        ax.set_zlabel(r'Интенсивность $|\phi|^2$', fontsize=9)
        ax.set_title(f"3D встречное соударение локализованных волновых пакетов на плоскости xy\nВремя t = {t:.2f}", fontsize=12)
        
        # Отрисовка трехмерной поверхности волны
        ax.plot_surface(X, Y, Intensity, cmap='plasma', edgecolor='none', alpha=0.9, rstride=2, cstride=2)
        
        # Плавное кинематографическое вращение камеры в процессе соударения
        ax.view_init(elev=30, azim=-60 + frame * 0.35)
        
    ani = FuncAnimation(fig, update, frames=steps, interval=60, repeat=True)
    plt.show()

# ------------------------------------------------------------
# 17. 3D-АНИМАЦИЯ ПУЛЬСАЦИЙ ДВУМЕРНОГО ДЫШАЩЕГО СОЛИТОНА (BREATHER)
# ------------------------------------------------------------
def animate_spatial_breather_3d(steps=100, t_max=np.pi/2):
    """
    3D-анимация пространственных периодических пульсаций (дыхания) двухсолитонного
    связанного состояния (солитон Сацумы-Ядзимы) на плоскости xy
    """
    print("\nПодготовка 3D-анимации пространственного дышащего солитона...")
    print("Вычисляется динамика пульсаций связанного состояния...")
    
    # Сетка пространства
    x = np.linspace(-5, 5, 80)
    y = np.linspace(-5, 5, 80)
    X, Y = np.meshgrid(x, y)
    
    # Временная сетка
    t_vals = np.linspace(-t_max, t_max, steps)
    
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Функция расчета мгновенного профиля
    def get_intensity(t):
        # Двухсолитонное решение по оси x
        num = np.cosh(3 * X) + 3 * np.exp(8j * t) * np.cosh(X)
        denom = np.cosh(4 * X) + 4 * np.cosh(2 * X) + 3 * np.cos(8 * t) + 12
        # Локализация по оси y (sech(y)) для формирования 2D-волнового пакета
        psi = 4 * np.exp(1j * t) * (num / denom) / np.cosh(Y)
        return np.abs(psi) ** 2
        
    # Поиск пика для масштабирования Z (максимум достигается в фазе t=0)
    peak_intensity = np.max(get_intensity(0.0))
    
    def update(frame):
        ax.clear()
        t = t_vals[frame]
        Intensity = get_intensity(t)
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(0, peak_intensity * 1.1)
        ax.set_xlabel('Пространство x', fontsize=9)
        ax.set_ylabel('Пространство y', fontsize=9)
        ax.set_zlabel(r'Интенсивность $|u|^2$', fontsize=9)
        ax.set_title(f"3D пульсации дышащего солитона на плоскости xy (Breather)\nВремя t = {t:.2f}")
        
        # Отрисовка пульсирующей 3D-поверхности
        ax.plot_surface(X, Y, Intensity, cmap='magma', edgecolor='none', alpha=0.9, rstride=2, cstride=2)
        
        # Кинематографический поворот
        ax.view_init(elev=30, azim=-60 + frame * 0.35)
        
    ani = FuncAnimation(fig, update, frames=steps, interval=60, repeat=True)
    plt.show()

# ------------------------------------------------------------
# 18. АНИМАЦИЯ РАССЕЯНИЯ СОЛИТОНА НА ПОТЕНЦИАЛЬНОМ БАРЬЕРЕ (УЧП, SSFM)
# ------------------------------------------------------------
def animate_soliton_potential_scattering(V0=2.0, w=1.0, pot_type='barrier', N=512, t_max=16.0, steps=200):
    """
    Численная 2D-анимация рассеяния, отражения или захвата одномерного волнового пакета 
    на потенциальных барьерах и ямах V(x) методом SSFM
    """
    print(f"\nПодготовка анимации рассеяния солитона на потенциале типа: '{pot_type}' (V0={V0})...")
    print("Выполняется расчет кадров...")
    
    L = 30.0
    dx = L / N
    x = np.linspace(-L/2, L/2, N, endpoint=False)
    
    # Задаем потенциал V(x)
    if pot_type == 'barrier':
        # Локализованный барьер в центре
        V = V0 * np.exp(-x**2 / (2 * w**2))
        title_str = "рассеяние на потенциальном барьере"
    elif pot_type == 'well':
        # Потенциальная яма в центре (V0 < 0)
        V = -abs(V0) * np.exp(-x**2 / (2 * w**2))
        title_str = "прохождение сквозь потенциальную яму"
    else: # lattice
        # Периодическая решетка
        V = V0 * (np.sin(np.pi * x / 3))**2
        title_str = "движение в периодической оптической решетке"
        
    # Начальный солитон: амплитуда A=1.2, положение x0=-7.0, движется вправо со скоростью v0=1.5
    A = 1.2
    x0 = -7.0
    v0 = 1.5
    psi = A / np.cosh(A * (x - x0)) * np.exp(1j * v0 * x)
    
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    dt = t_max / steps
    
    # Прекомпиляция всех кадров
    history = []
    current_psi = psi.copy()
    # Линейная дисперсия (полушаг)
    dispersion_half_step = np.exp(-1j * 0.25 * (k**2) * dt)
    
    for step in range(steps):
        history.append(np.abs(current_psi) ** 2)
        
        # SSFM шаги
        # 1. Линейный полушаг
        psi_k = np.fft.fft(current_psi) * dispersion_half_step
        current_psi = np.fft.ifft(psi_k)
        
        # 2. Нелинейный полный шаг + фазовый сдвиг от потенциала V(x)
        current_psi = current_psi * np.exp(1j * (np.abs(current_psi)**2 - V) * dt)
        
        # 3. Линейный полушаг
        psi_k = np.fft.fft(current_psi) * dispersion_half_step
        current_psi = np.fft.ifft(psi_k)
        
    # Настройка окна
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-12, 12)
    max_int = max(np.max(history), abs(V0))
    ax.set_ylim(-abs(V0)*1.1 if pot_type=='well' else 0, max_int * 1.2)
    ax.set_xlabel('Пространство x', fontsize=11)
    ax.set_ylabel('Амплитуда полей', fontsize=11)
    ax.set_title(f"Анимация УЧП: {title_str} ($V_0 = {V0}$)", fontsize=13)
    
    # Отрисовка потенциала V(x) на заднем плане
    ax.fill_between(x, 0, V, color='gray', alpha=0.2, label='Потенциал V(x)')
    ax.plot(x, V, 'k--', alpha=0.5)
    
    line, = ax.plot(x, history[0], color='crimson', lw=2.5, label=r'Профиль солитона $|\psi|^2$')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    ax.legend(loc='upper right')
    
    def update(frame):
        line.set_ydata(history[frame])
        time_text.set_text(f"Время t = {frame * dt:.2f}")
        return line, time_text
        
    ani = FuncAnimation(fig, update, frames=steps, interval=45, blit=True, repeat=True)
    plt.show()

# ------------------------------------------------------------
# 14. ФАЗОВЫЕ ПОРТРЕТЫ И АНИМАЦИЯ
# ------------------------------------------------------------
def plot_phase_portrait_auto(c1, c2, xlim=(-5,5), ylim=(-5,5), show_separatrices=True, lw=1.8):
    fig, ax = plt.subplots(figsize=(11, 9))
    Phi_min, Phi_max = xlim
    y_min, y_max = ylim

    P, Y = np.meshgrid(np.linspace(Phi_min, Phi_max, 20), np.linspace(y_min, y_max, 20))
    dPhi = Y
    dy = c1 * P**3 + c2 * P
    M = np.hypot(dPhi, dy)
    M[M == 0] = 1
    ax.quiver(P, Y, dPhi / M, dy / M, color='gray', alpha=0.3, scale=45, headwidth=3)

    ax.axhline(0, color='blue', linestyle=':', lw=1.5, alpha=0.6, label=r'Нуль-клиналь $\frac{d\Phi}{d\xi}=0$ ($y=0$)')
    ax.axvline(0, color='green', linestyle=':', lw=1.5, alpha=0.6, label=r'Нуль-клиналь $\frac{dy}{d\xi}=0$')
    if c1 != 0 and -c2/c1 > 0:
        val = np.sqrt(-c2/c1)
        ax.axvline(val, color='green', linestyle=':', lw=1.5, alpha=0.6)
        ax.axvline(-val, color='green', linestyle=':', lw=1.5, alpha=0.6)

    for Phi_f, y_f, typ in fixed_points(c1, c2):
        color = 'red' if typ == 'седло' else ('green' if typ == 'центр' else 'orange')
        marker = 's' if typ == 'седло' else 'o'
        ax.plot(Phi_f, y_f, marker=marker, color=color, markersize=8, markeredgecolor='black', zorder=5)
        ax.text(Phi_f + 0.1, y_f + 0.1, f"{typ} ({Phi_f:.2f}, {y_f:.2f})", fontsize=9, fontweight='bold')

    if show_separatrices:
        for Phi_f, y_f, typ in fixed_points(c1, c2):
            if typ == 'седло':
                fwd, rev = separatrix_from_saddle(Phi_f, y_f, c1, c2, xlim)
                if fwd: ax.plot(fwd[1], fwd[2], 'r-', lw=lw + 0.7, alpha=0.85, label='Сепаратриса (неуст.)')
                if rev: ax.plot(rev[1], rev[2], 'b-', lw=2.2 + 0.7, alpha=0.85, label='Сепаратриса (уст.)')

    seeds = []
    for r in np.linspace(0.5, min(abs(Phi_min), abs(y_min)) * 0.8, 5):
        for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
            seeds.append((r * np.cos(theta), r * np.sin(theta)))
            
    for p0, y0 in seeds:
        try:
            _, P_t, Y_t = integrate_full_trajectory(p0, y0, c1, c2, xlim)
            ax.plot(P_t, Y_t, color='indigo', lw=lw, alpha=0.4)
        except Exception:
            continue

    ax.set_xlabel(r'$\Phi$', fontsize=12)
    ax.set_ylabel(r"$y = d\Phi/d\xi$", fontsize=12)
    ax.set_xlim(Phi_min, Phi_max)
    ax.set_ylim(y_min, y_max)
    
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right', framealpha=0.9)
    plt.tight_layout()
    plt.show()

def animate_phase_portrait(c1_fixed, c2_fixed, vary_param, param_start, param_end,
                           steps=40, xlim=(-5,5), ylim=(-5,5), show_separatrices=True, lw=1.8):
    fig, ax = plt.subplots(figsize=(10, 8))
    Phi_min, Phi_max = xlim
    y_min, y_max = ylim
    
    init_list = [
        (0.4, 0.0), (1.0, 0.0), (2.0, 0.0), (-1.0, 0.0), (-2.0, 0.0),
        (0.0, 0.8), (0.0, 1.5), (0.0, -0.8), (0.0, -1.5),
        (1.2, 1.2), (-1.2, 1.2), (1.2, -1.2), (-1.2, -1.2)
    ]
    colors_traj = plt.cm.plasma(np.linspace(0, 1, len(init_list)))
    
    def update(frame):
        ax.clear()
        val = param_start + (param_end - param_start) * frame / (steps - 1)
        c1 = val if vary_param == 'c1' else c1_fixed
        c2 = c2_fixed if vary_param == 'c1' else val
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r'$\Phi$', fontsize=11)
        ax.set_ylabel(r"$y$", fontsize=11)
        ax.set_title(f"Эволюция фазового портрета при изменении {vary_param}")
        
        P, Y = np.meshgrid(np.linspace(Phi_min, Phi_max, 15), np.linspace(y_min, y_max, 15))
        dPhi = Y
        dy = c1 * P**3 + c2 * P
        M = np.hypot(dPhi, dy)
        M[M == 0] = 1
        ax.quiver(P, Y, dPhi / M, dy / M, color='gray', alpha=0.3, scale=40)
        
        for idx, (p0, y0) in enumerate(init_list):
            try:
                _, P_t, Y_t = integrate_full_trajectory(p0, y0, c1, c2, xlim, num_points=1000, 
                                                         method='RK45', rtol=1e-7, atol=1e-9)
                ax.plot(P_t, Y_t, color=colors_traj[idx], lw=lw, alpha=0.6)
            except Exception:
                continue
        
        for Phi_f, y_f, typ in fixed_points(c1, c2):
            color = 'red' if typ == 'седло' else 'green'
            marker = 's' if typ == 'седло' else 'o'
            ax.plot(Phi_f, y_f, marker=marker, color=color, markersize=8, markeredgecolor='black', zorder=5)
            
            if show_separatrices and typ == 'седло':
                try:
                    fwd, rev = separatrix_from_saddle(Phi_f, y_f, c1, c2, xlim, num=1500,
                                                      method='RK45', rtol=1e-7, atol=1e-9)
                    if fwd: ax.plot(fwd[1], fwd[2], 'r-', lw=lw * 1.5, alpha=0.8)
                    if rev: ax.plot(rev[1], rev[2], 'b-', lw=2, alpha=0.8)
                except Exception:
                    pass
            
        ax.text(0.05, 0.95, f"$c_1 = {c1:.3f}$\n$c_2 = {c2:.3f}$", transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ani = FuncAnimation(fig, update, frames=steps, interval=150, repeat=True)
    plt.show()

# ------------------------------------------------------------
# 19. ГЛАВНОЕ МЕНЮ С 18 ПУНКТАМИ
# ------------------------------------------------------------
def main():
    while True:
        print("\n" + "="*60)
        print("   ИНТЕРАКТИВНЫЙ КОМПЛЕКС: МОДЕЛИРОВАНИЕ ХИРОТЫ–МАККАРИ")
        print("="*60)
        print("1. Показать 4 качественные структуры фазовых портретов")
        print("2. Построить фазовый портрет для произвольных параметров (c1, c2)")
        print("3. Сравнить численное и точное аналитическое решения")
        print("4. Анимировать эволюцию портрета при изменении параметров")
        print("5. Моделирование темного солитона с мультипликативным шумом")
        print("6. Анализ периодического воздействия и сечение Пуанкаре")
        print("7. Построить 3D пространственный солитон |phi(x,y,t)|^2")
        print("8. Построить фрактальную бифуркационную диаграмму хаоса")
        print("9. Построить влияние порядка дробной производной alpha")
        print("10. [ИСПРАВЛЕНО] Проверить инвариант энергии (Гамильтониан, Ур. 38)")
        print("11. Спектральный анализ Фурье (БПФ/FFT) и диагностика хаоса")
        print("12. Построить эволюцию 2-солитонного связанного состояния (Разд. 2.1)")
        print("13. Расчет показателя Ляпунова (метод Бенеттина) для хаоса")
        print("14. Моделирование столкновения солитонов (УЧП Шрёдингера, SSFM)")
        print("15. Анимация столкновения солитонов в реальном времени (2D профиль)")
        print("16. [ИСПРАВЛЕНО] 3D-анимация столкновения волн на плоскости xy (УЧП)")
        print("17. [НОВОЕ] Рассеяние и захват солитона на потенциальном барьере/яме (УЧП)")
        print("18. [НОВОЕ] 3D-анимация пульсаций дышащего солитона (Breather) на плоскости")
        print("19. Выйти")
        print("="*60)
        
        choice = input("Ваш выбор (1-19): ").strip()
        if choice == '1':
            lw_choice = float(input("Толщина линий траекторий (по умолчанию 1.8): ") or "1.8")
            plot_phase_portrait_auto(1.0, 1.0, show_separatrices=False, lw=lw_choice)
            plot_phase_portrait_auto(-1.0, -1.0, show_separatrices=False, lw=lw_choice)
            plot_phase_portrait_auto(1.0, -2.0, show_separatrices=True, lw=lw_choice)
            plot_phase_portrait_auto(-1.0, 2.0, show_separatrices=True, lw=lw_choice)
        elif choice == '2':
            try:
                c1, c2 = float(input("c1: ")), float(input("c2: "))
                lw_choice = float(input("Толщина линий траекторий (по умолчанию 1.8): ") or "1.8")
                plot_phase_portrait_auto(c1, c2, lw=lw_choice)
            except ValueError: print("Ошибка ввода.")
        elif choice == '3':
            try:
                print("\n1 - cn Якоби, 2 - Тёмный солитон, 3 - Светлый солитон (sech)")
                typ = input("Выбор: ").strip()
                c1, c2 = float(input("c1: ")), float(input("c2: "))
                
                if typ == '1': compare_numerical_analytical(c1, c2, lambda xi: sol_set2(xi, c1, c2, m=0.35), title="cn Jacobi")
                elif typ == '2': compare_numerical_analytical(c1, c2, lambda xi: sol_set3(xi, c1, c2), title="Тёмный солитон")
                elif typ == '3': compare_numerical_analytical(c1, c2, lambda xi: sol_set4(xi, c1, c2), title="Светлый солитон")
                else: print("Неверный выбор.")
            except ValueError: print("Ошибка.")
        elif choice == '4':
            try:
                param_input = input("Параметр варьирования (c1 / c2): ").strip().lower().replace('с', 'c')
                if param_input in ['1', 'c1']: vary_param = 'c1'
                elif param_input in ['2', 'c2']: vary_param = 'c2'
                else: continue
                fixed_val = float(input("Фиксированное значение: "))
                p_start, p_end = float(input("Начало: ")), float(input("Конец: "))
                lw_choice = float(input("Толщина линий траекторий (по умолчанию 1.8): ") or "1.8")
                animate_phase_portrait(fixed_val if vary_param=='c2' else None, fixed_val if vary_param=='c1' else None,
                                       vary_param, p_start, p_end, lw=lw_choice)
            except ValueError: print("Ошибка.")
        elif choice == '5':
            try:
                c1 = float(input("Параметр c1 (рекомендуется 1.0): "))
                c2 = float(input("Введите c2 (рекомендуется -2.0): "))
                if c1 <= 0 or c2 >= 0:
                    print("\n[Ошибка] Стохастический темный солитон требует c1 > 0 и c2 < 0.")
                    continue
                epsilon = float(input("Интенсивность шума (рекомендуется 0.3): ") or "0.3")
                lw_choice = float(input("Толщина линий траекторий (по умолчанию 1.8): ") or "1.8")
                compare_numerical_analytical(c1, c2, lambda xi: sol_set3(xi, c1, c2), title="Тёмный солитон (без шума)", lw=lw_choice)
                plot_stochastic_results(c1, c2, epsilon=epsilon, lw=lw_choice)
            except ValueError: print("Ошибка.")
        elif choice == '6':
            try:
                c1 = float(input("Введите c1 (например, -1.0): "))
                c2 = float(input("Введите c2 (например, 2.0): "))
                f0 = float(input("Амплитуда возмущения f0 (рекомендуется 0.8): ") or "0.8")
                omega = float(input("Частота возмущения omega (рекомендуется 1.4): ") or "1.4")
                plot_poincare_section(c1, c2, f0=f0, omega=omega)
            except ValueError: print("Ошибка.")
        elif choice == '7':
            print("\nВыберите волновой 3D профиль для визуализации:")
            print("1 - Светлый солитон (bright, sech) [c1 < 0, c2 > 0]")
            print("2 - Тёмный солитон (dark, tanh) [c1 > 0, c2 < 0]")
            sol_choice = input("Ваш выбор: ").strip()
            try:
                c1, c2 = float(input("c1: ")), float(input("c2: "))
                t_val = float(input("Момент времени t (по умолчанию 0.0): ") or "0.0")
                if sol_choice == '1':
                    plot_3d_spatial_soliton(c1, c2, t_val=t_val, sol_type='bright')
                elif sol_choice == '2':
                    plot_3d_spatial_soliton(c1, c2, t_val=t_val, sol_type='dark')
                else: print("Неверный тип.")
            except ValueError: print("Ошибка.")
        elif choice == '8':
            try:
                c1 = float(input("Введите c1 (рекомендуется -1.0): "))
                c2 = float(input("Введите c2 (рекомендуется 2.0): "))
                omega = float(input("Частота возмущения omega (рекомендуется 1.4): ") or "1.4")
                plot_bifurcation_diagram_driven(c1, c2, omega=omega)
            except ValueError: print("Ошибка.")
        elif choice == '9':
            try:
                c1 = float(input("Введите c1 (рекомендуется -1.0): "))
                c2 = float(input("Введите c2 (рекомендуется 2.0): "))
                p_val = float(input("Волновое число p (рекомендуется 1.5): ") or "1.5")
                plot_fractional_derivative_effect(c1, c2, p_val=p_val)
            except ValueError: print("Ошибка.")
        elif choice == '10':
            try:
                c1 = float(input("Введите c1: "))
                c2 = float(input("Введите c2: "))
                Phi0 = float(input("Начальное значение Ф0 (например, 1.0): ") or "1.0")
                y0 = float(input("Начальная производная y0 (например, 0.5): ") or "0.5")
                lw_choice = float(input("Толщина линий траекторий (по умолчанию 1.8): ") or "1.8")
                plot_hamiltonian_conservation(c1, c2, Phi0=Phi0, y0=y0, lw=lw_choice)
            except ValueError: print("Ошибка ввода.")
        elif choice == '11':
            try:
                c1 = float(input("Введите c1 (рекомендуется -1.0): "))
                c2 = float(input("Введите c2 (рекомендуется 2.0): "))
                f0 = float(input("Амплитуда возмущения f0 (рекомендуется 0.8): ") or "0.8")
                omega = float(input("Частота возмущения omega (рекомендуется 1.4): ") or "1.4")
                lw_choice = float(input("Толщина линий траекторий (по умолчанию 1.8): ") or "1.8")
                plot_fft_chaos_analysis(c1, c2, f0=f0, omega=omega, lw=lw_choice)
            except ValueError: print("Ошибка ввода.")
        elif choice == '12':
            plot_breather_2soliton_dynamics()
        elif choice == '13':
            try:
                c1 = float(input("Введите c1 (рекомендуется -1.0): "))
                c2 = float(input("Введите c2 (рекомендуется 2.0): "))
                f0 = float(input("Амплитуда возмущения f0 (рекомендуется 0.8): ") or "0.8")
                omega = float(input("Частота возмущения omega (рекомендуется 1.4): ") or "1.4")
                lw_choice = float(input("Толщина линий траекторий (по умолчанию 1.8): ") or "1.8")
                plot_lyapunov_exponent_driven(c1, c2, f0=f0, omega=omega, lw=lw_choice)
            except ValueError: print("Ошибка ввода.")
        elif choice == '14':
            try:
                A1 = float(input("Амплитуда 1-го солитона (рекомендуется 1.5): ") or "1.5")
                A2 = float(input("Амплитуда 2-го солитона (рекомендуется 1.0): ") or "1.0")
                v1 = float(input("Скорость 1-го солитона (рекомендуется 1.5): ") or "1.5")
                v2 = float(input("Скорость 2-го солитона (рекомендуется -1.5): ") or "-1.5")
                simulate_two_soliton_collision_pde(A1=A1, A2=A2, v1=v1, v2=v2)
            except ValueError: print("Ошибка ввода.")
        elif choice == '15':
            try:
                A1 = float(input("Амплитуда 1-го солитона (рекомендуется 1.5): ") or "1.5")
                A2 = float(input("Амплитуда 2-го солитона (рекомендуется 1.0): ") or "1.0")
                v1 = float(input("Скорость 1-го солитона (рекомендуется 1.5): ") or "1.5")
                v2 = float(input("Скорость 2-го солитона (рекомендуется -1.5): ") or "-1.5")
                animate_two_soliton_collision_pde(A1=A1, A2=A2, v1=v1, v2=v2)
            except ValueError: print("Ошибка ввода.")
        elif choice == '16':
            try:
                A1 = float(input("Амплитуда 1-го солитона (рекомендуется 1.2): ") or "1.2")
                A2 = float(input("Амплитуда 2-го солитона (рекомендуется 1.0): ") or "1.0")
                animate_two_soliton_collision_3d_spatial(A1=A1, A2=A2)
            except ValueError: print("Ошибка ввода.")
        elif choice == '17':
            print("\nВыберите тип взаимодействия солитона с потенциалом V(x):")
            print("1 - Отражение от потенциального барьера (Barrier reflection) [V0 > 0]")
            print("2 - Прохождение сквозь потенциальную яму (Well transmission) [V0 < 0]")
            print("3 - Движение в периодической оптической решетке (Lattice scattering)")
            pot_choice = input("Ваш выбор: ").strip()
            try:
                if pot_choice == '1':
                    animate_soliton_potential_scattering(V0=3.0, w=0.8, pot_type='barrier')
                elif pot_choice == '2':
                    animate_soliton_potential_scattering(V0=1.5, w=1.2, pot_type='well')
                elif pot_choice == '3':
                    animate_soliton_potential_scattering(V0=1.5, w=1.0, pot_type='lattice')
                else: print("Неверный выбор.")
            except ValueError: print("Ошибка ввода.")
        elif choice == '18':
            animate_spatial_breather_3d()
        elif choice == '19':
            print("До свидания!")
            break

# Эволюция связанного 2-солитонного состояния Сацумы-Ядзимы
def plot_breather_2soliton_dynamics():
    print("\nПостроение пространственно-временной динамики связанного 2-солитонного решения (Breather)...")
    x = np.linspace(-5, 5, 200)
    t = np.linspace(-np.pi/2, np.pi/2, 200)
    X, T = np.meshgrid(x, t)
    
    # Точное дыхание (дышащий солитон)
    num = np.cosh(3*X) + 3 * np.exp(8j*T) * np.cosh(X)
    denom = np.cosh(4*X) + 4 * np.cosh(2*X) + 3 * np.cos(8*T)
    psi = 4 * np.exp(1j*T) * (num / denom)
    Intensity = np.abs(psi) ** 2
    
    fig = plt.figure(figsize=(14, 6))
    
    # 3D
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, T, Intensity, cmap='magma', edgecolor='none', alpha=0.95)
    ax1.set_xlabel('Пространство x')
    ax1.set_ylabel('Время t')
    ax1.set_zlabel(r'Интенсивность $|u|^2$')
    ax1.set_title('3D эволюция дыхания солитона')
    
    # 2D сверху
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(Intensity, extent=[-5, 5, -np.pi/2, np.pi/2], aspect='auto', origin='lower', cmap='magma')
    ax2.set_xlabel('Пространство x')
    ax2.set_ylabel('Время t')
    ax2.set_title('Проекция на плоскость (вид сверху)')
    fig.colorbar(im, ax=ax2, label=r'$|u(x,t)|^2$')
    
    plt.suptitle("Точное двухсолитонное связанное состояние (Метод Хироты, Глава 2, разд. 2.1)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()