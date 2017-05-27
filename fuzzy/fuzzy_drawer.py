from pylab import *
import itertools

def draw(ax, l, m, u, alphas=[]):
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    ax.set_xticks([l,m,u])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_visible(False)
    ax.plot([l, m, u], [0,1,0], color='r')
    ax.fill_between([l, m, u], [0,0,0],[0,1,0], alpha=0.3, color='r')
    for alpha in alphas:
        assert(alpha > 0 and alpha < 1)
        ax.step([l + alpha * (m - l), u - alpha * (u - m), u - alpha * (u - m)], [0, alpha, 0], linestyle='--', color='b')

def draw_matrix(l, m, u, alphas=[]):
    fsize = (1.5 * l.shape[0], 2 * l.shape[1])
    fig, axes = subplots(*l.shape, figsize=fsize)
    for coords in itertools.product(*map(range, axes.shape)):
        draw(axes[coords], l[coords], m[coords], u[coords], alphas)
    fig.tight_layout()

nn = np.ones((4,3))
l, m, u = nn * 0.3, nn * 0.6, nn * 0.7
draw_matrix(l, m, u, [0.2, 0.4, 0.6])

show()
