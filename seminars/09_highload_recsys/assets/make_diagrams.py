"""Draw the seminar's algorithm diagrams with Matplotlib."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).parent
BLUE = '#2463a6'
GREEN = '#278568'
ORANGE = '#cb741c'
INK = '#233044'
PALE = ['#e4edf8', '#e1f1e9', '#fff0dc']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                     'text.color': INK, 'axes.titleweight': 'semibold', 'figure.facecolor': 'white'})


def save(fig, name):
    fig.savefig(ROOT / (name + '.png'), dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(ROOT / (name + '.svg'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def box(ax, x, y, w, h, text, color='#e4edf8', fontsize=12, edge=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
                               facecolor=color, edgecolor=edge, linewidth=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize)


def arrow(ax, start, end, color=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=15,
                                 color=color, linewidth=1.5))


def draw_blocked():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={'width_ratios': [2, 1, 0.8]})
    values = np.arange(8 * 12).reshape(8, 12) % 7
    axes[0].imshow(values, cmap='Blues', alpha=0.4, aspect='equal', vmin=0, vmax=8)
    axes[0].add_patch(Rectangle((3.5, 1.5), 4, 3, fill=False, edgecolor=ORANGE, linewidth=3))
    axes[0].set(xticks=range(12), yticks=range(8), xlabel='Объекты каталога', ylabel='Пользователи',
                title='Вся матрица скоров: Q × N')
    axes[0].tick_params(labelsize=9)
    for ax, shape, color, title in [
        (axes[1], (3, 4), ORANGE, 'Скоры текущей порции\nBᵤ × Bᵢ'),
        (axes[2], (3, 2), GREEN, 'Текущие лучшие\nBᵤ × K'),
    ]:
        ax.imshow(np.ones(shape), cmap=matplotlib.colors.ListedColormap([color]), alpha=0.23)
        for row in range(shape[0]):
            for col in range(shape[1]):
                ax.add_patch(Rectangle((col - .5, row - .5), 1, 1, fill=False,
                                       edgecolor='white', linewidth=2))
        ax.set(xticks=[], yticks=[], title=title)
        ax.spines[:].set_visible(False)
    fig.add_artist(ConnectionPatch(xyA=(7.7, 3), coordsA=axes[0].transData,
                                  xyB=(-.8, 1), coordsB=axes[1].transData,
                                  arrowstyle='-|>', mutation_scale=15, color=ORANGE, linewidth=1.7))
    fig.add_artist(ConnectionPatch(xyA=(3.7, 1), coordsA=axes[1].transData,
                                  xyB=(-.8, 1), coordsB=axes[2].transData,
                                  arrowstyle='-|>', mutation_scale=15, color=GREEN, linewidth=1.7))
    fig.text(.52, .07, 'После каждой порции обновляем top-K. Затем обрабатываем следующую порцию каталога.',
             ha='center', fontsize=11)
    fig.subplots_adjust(wspace=.6, bottom=.2, top=.8)
    save(fig, 'blocked_exact_search')


def draw_ivf():
    fig = plt.figure(figsize=(12, 5.2))
    left = fig.add_axes([.05, .13, .45, .77])
    right = fig.add_axes([.57, .08, .41, .84])
    right.set(xlim=(0, 1), ylim=(0, 1)); right.axis('off')
    rng = np.random.default_rng(21)
    angles = np.array([10, 65, 125, 185, 245, 305]) * np.pi / 180
    centers = np.column_stack([np.cos(angles), np.sin(angles)])
    query = np.array([1.2, .7])
    selected = set(np.argsort(centers @ query)[-2:])
    for i, center in enumerate(centers):
        active = i in selected
        color = BLUE if active else '#a8b0bc'
        pts = center + rng.normal(scale=.15, size=(20, 2))
        left.scatter(pts[:, 0], pts[:, 1], s=20, color=color, alpha=.7 if active else .35)
        left.scatter(*center, marker='X', s=130, color=color, edgecolors='white', linewidths=1)
        left.text(center[0] + .07, center[1] + .1, f'c{i}', fontsize=11, color=color)
        y = .86 - i * .14
        box(right, .13, y, .82, .095, f'Список {i}:  i₀, i₁, i₂, …',
            color='#e4edf8' if active else '#f4f5f7', edge=BLUE if active else '#c7ccd3')
        right.text(.05, y + .047, '✓' if active else '', color=BLUE, ha='center', va='center', fontsize=18)
    left.annotate('', xy=query, xytext=(0, 0), arrowprops={'arrowstyle': '-|>', 'lw': 2, 'color': ORANGE})
    left.scatter(*query, marker='*', s=200, color=ORANGE, zorder=5)
    left.text(query[0], query[1] + .3, 'Запрос x', color=ORANGE, ha='center')
    left.axhline(0, color='#e2e6eb', linewidth=.7); left.axvline(0, color='#e2e6eb', linewidth=.7)
    left.set(xlim=(-1.6, 1.6), ylim=(-1.55, 1.6), aspect='equal', xticks=[], yticks=[],
             title='Выбираем центроиды по xᵀc')
    left.spines[:].set_visible(False)
    right.set_title('Просматриваем только выбранные списки', pad=8)
    fig.text(.51, .025, 'Пример: nlist = 6, nprobe = 2. Серые списки в этом запросе пропускаем.',
             ha='center', fontsize=12)
    save(fig, 'ivf_search')


def draw_pq():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    vector = np.array([[.12, .78], [-.31, .52], [1.18, -.69]])
    books = np.array([
        [[-.8, -.2], [.1, .8], [.5, .5], [1, -.7]],
        [[0, 0], [.8, .2], [-.3, .5], [-.7, -.8]],
        [[1.2, -.7], [0, .1], [-1, .4], [.5, .8]],
    ])
    codes = np.argmin(np.sum((books - vector[:, None])**2, axis=2), axis=1)
    assert codes.tolist() == [1, 2, 0]
    ax.text(.5, 1.045, 'PQ: пример d = 6, m = 3, b = 2', ha='center', va='top', fontsize=16, weight='semibold')
    ax.text(.025, .91, 'Исходный\nвектор y', va='center', fontsize=11)
    ax.text(.025, .56, 'Codebook-и', va='center', fontsize=11)
    for j, left in enumerate([.17, .44, .71]):
        box(ax, left, .87, .23, .075, f'[{vector[j, 0]:.2f}, {vector[j, 1]:.2f}]', color=PALE[j])
        arrow(ax, (left + .115, .85), (left + .115, .765))
        ax.text(left + .115, .73, f'Часть {j + 1}: 4 центроида', ha='center', fontsize=10)
        for index, centroid in enumerate(books[j]):
            y = .64 - index * .073
            chosen = index == codes[j]
            box(ax, left, y, .23, .055,
                f'{index}: [{centroid[0]:.1f}, {centroid[1]:.1f}]',
                color=PALE[j] if chosen else '#f7f8fa',
                edge=GREEN if chosen else '#d6dce4', fontsize=11)
        arrow(ax, (left + .115, .39), (left + .115, .315))
        box(ax, left, .245, .23, .06, str(codes[j]), color=PALE[j], edge=GREEN, fontsize=16)
        arrow(ax, (left + .115, .23), (left + .115, .15))
        centroid = books[j, codes[j]]
        box(ax, left, .07, .23, .07, f'[{centroid[0]:.1f}, {centroid[1]:.1f}]', color=PALE[j], edge=GREEN)
    ax.text(.025, .275, 'Код q', va='center', fontsize=11)
    ax.text(.025, .105, 'Восстанов-\nленный ŷ', va='center', fontsize=11)
    ax.text(.565, .015, 'Квантование: выбираем ближайшие центроиды. Деквантизация: подставляем их по номерам.',
            ha='center', fontsize=11)
    save(fig, 'pq_encoding_decoding')


def draw_ivfpq():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.text(.015, .95, 'Построение IVF-PQ', fontsize=15, weight='semibold')
    top = [(0.025, 'Объект y'), (.265, 'Центроид\nсписка cℓ'), (.505, 'Остаток\nr = y − cℓ'), (.745, 'PQ-код\nостатка r')]
    for i, (x, text) in enumerate(top):
        box(ax, x, .65, .2, .18, text, color=PALE[min(i, 2)])
        if i < len(top) - 1: arrow(ax, (x + .208, .74), (top[i + 1][0] - .015, .74))
    ax.text(.015, .49, 'Поиск и точный пересчёт', fontsize=15, weight='semibold')
    bottom = [(0.025, 'Запрос x'), (.225, 'Выбор nprobe\nсписков'), (.425, 'Скоры\nпо PQ-кодам'),
              (.625, 'Top-L\nкандидатов'), (.825, 'Пересчёт FP32\n→ top-K')]
    for i, (x, text) in enumerate(bottom):
        box(ax, x, .17, .15, .2, text, color=PALE[min(i, 2)], fontsize=11)
        if i < len(bottom) - 1: arrow(ax, (x + .16, .27), (bottom[i + 1][0] - .015, .27))
    ax.text(.5, .025, 'До пересчёта скоры приближённые. Точные векторы нужны только для L кандидатов.',
            ha='center', fontsize=11)
    save(fig, 'ivfpq_pipeline')


if __name__ == '__main__':
    draw_blocked()
    draw_ivf()
    draw_pq()
    draw_ivfpq()
    print('Generated four PNG/SVG diagrams.')
