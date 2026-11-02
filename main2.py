import numpy as np
import random
import matplotlib.pyplot as plt

x = np.array([
    1.0, 1.5, 2.0, 2.5, 3.0,
    3.5, 4.0, 4.5, 5.0, 5.5,
    6.0, 6.5, 7.0, 7.5, 8.0
])

y = np.array([
    2.20, 3.10, 14.50, 4.90, 5.80,
    -3.50, 7.90, 8.80, 19.20, 10.90,
    11.80, 1.00, 14.70, 22.50, 16.60
])


def ransac(t, rng):
    rand_num, rand_num2 = rng.sample(range(len(x)), 2)

    m = (y[rand_num2] - y[rand_num]) / (x[rand_num2] - x[rand_num])
    b = y[rand_num] - m * x[rand_num]

    count = 0

    for n in range(len(x)):
        distance = abs(y[n] - (m * x[n] + b))

        if distance <= t:
            count += 1

    return [[m, b], count]


def fit_ransac(t, S, seed):
    rng = random.Random(seed)
    counts = []

    for n in range(S):
        counts.append(ransac(t, rng))

    largest_count = max(counts, key=lambda result: result[1])
    m, b = largest_count[0]

    inliers = []

    for n in range(len(x)):
        distance = abs(y[n] - (m * x[n] + b))
        inliers.append(distance <= t)

    inliers = np.array(inliers)

    inlier_x = x[inliers]
    inlier_y = y[inliers]

    X = np.column_stack([inlier_x, np.ones_like(inlier_x)])
    m_final, c_final = np.linalg.lstsq(X, inlier_y, rcond=None)[0]

    return m_final, c_final, inliers


def plot_result(ax, result, m_ols, c_ols, title):
    m, c, inliers = result
    line_x = np.linspace(x.min(), x.max(), 200)

    ax.scatter(
        x[inliers], y[inliers],
        color="blue", label="Consensus inliers"
    )

    ax.scatter(
        x[~inliers], y[~inliers],
        color="red", marker="x", label="Consensus outliers"
    )

    ax.plot(
        line_x, m_ols * line_x + c_ols,
        "--", color="orange", label="OLS: all points"
    )

    ax.plot(
        line_x, m * line_x + c,
        color="green", label="RANSAC refit"
    )

    ax.set_title(
        title + f"\n{inliers.sum()} inliers, m={m:.3f}, c={c:.3f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def main():
    # OLS fitted to every point
    X = np.column_stack([x, np.ones_like(x)])
    m_ols, c_ols = np.linalg.lstsq(X, y, rcond=None)[0]

    print(f"OLS: m={m_ols:.6f}, c={c_ols:.6f}")

    # Threshold comparison: same iterations and sampled pairs
    print("\nThreshold comparison: S=50")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax, t in zip(axes, [0.5, 1.5, 4.0]):
        result = fit_ransac(t, 50, seed=0)
        m, c, inliers = result

        print(
            f"t={t}: inliers={inliers.sum()}, "
            f"m={m:.6f}, c={c:.6f}"
        )

        plot_result(ax, result, m_ols, c_ols, f"t={t}, S=50")

    fig.tight_layout()

    # Iteration comparison: same threshold and random seed
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, S in zip(axes, [5, 50]):
        result = fit_ransac(1.5, S, seed=0)
        plot_result(ax, result, m_ols, c_ols, f"t=1.5, S={S}")

    fig.tight_layout()

    # Repeat runs because one run cannot establish stability
    print("\nStability over 100 runs: t=1.5")

    for S in [5, 50]:
        slopes = []
        intercepts = []
        inlier_counts = []

        for seed in range(100):
            m, c, inliers = fit_ransac(1.5, S, seed)

            slopes.append(m)
            intercepts.append(c)
            inlier_counts.append(int(inliers.sum()))

        print(f"\nS={S}")
        print(f"Average inliers: {np.mean(inlier_counts):.2f}")
        print(f"Minimum inliers: {min(inlier_counts)}")
        print(f"Maximum inliers: {max(inlier_counts)}")
        print(f"Slope standard deviation: {np.std(slopes):.6f}")
        print(f"Intercept standard deviation: {np.std(intercepts):.6f}")

    plt.show()


if __name__ == "__main__":
    main()