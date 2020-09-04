import matplotlib
import matplotlib.pyplot as plt
from dotmap import DotMap

matplotlib.use('agg')


PLOT_CONFIG = DotMap({
    'TITLE_FONTSIZE': 19,
    'XLABEL_FONTSIZE': 18,
    'YLABEL_FONTSIZE': 18,
    'XTICKS_FONTSIZE': 18,
    'YTICKS_FONTSIZE': 18,
    'LEGEND_FONTSIZE': 16,
    'FIGSIZE': (9, 6),
    'DPI': 320
})


def get_ticks(data: int, interval: int):
    ticks = data
    for idx in range(len(ticks)):
        if idx % interval != 0:
            ticks[idx] = ''
    return ticks


def plot_rewards(saveto_filename: str,
                 data: list,
                 ylim: tuple,
                 xticks_interval: int = 100,
                 yticks_interval: int = 5,
                 mean_window_size: int = 100,
                 dpi = PLOT_CONFIG.DPI):

    print('\n\nGenerating Figure: Acc. Reward vs. Training Episodes')

    num_episodes = len(data)

    plt.figure(figsize=PLOT_CONFIG.FIGSIZE)

    for episode_idx, episode_reward in enumerate(data):
        mean_rewards = [sum(data[i:i + mean_window_size]) / mean_window_size
                        for i in range(len(data) - mean_window_size + 1)]

        plt.plot(range(num_episodes), data, linewidth=0.75, linestyle='solid', color='b')

    plot_legend_label = 'Moving Average (window size: {} episodes)'.format(mean_window_size)
    plt.plot(range(len(mean_rewards)), mean_rewards, label=plot_legend_label, linewidth=2.5, linestyle='dashed',
             color='r')

    title = 'Accumulated Reward over Training Episodes' + '\n' + \
            'Using Learned Benchmark Parameters ({} Episodes)'.format(num_episodes)
    plt.title(title, fontsize=PLOT_CONFIG.TITLE_FONTSIZE)

    plt.xlabel('Episode', fontsize=PLOT_CONFIG.XLABEL_FONTSIZE)
    plt.ylabel('Reward', fontsize=PLOT_CONFIG.YLABEL_FONTSIZE)

    plt.xticks(range(0, len(data) + xticks_interval, xticks_interval), fontsize=PLOT_CONFIG.XTICKS_FONTSIZE)
    plt.yticks(range(-25, 30, yticks_interval), fontsize=PLOT_CONFIG.YTICKS_FONTSIZE)

    plt.xlim(0, num_episodes)
    plt.ylim(ylim[0], ylim[1])

    plt.legend(fontsize=PLOT_CONFIG.LEGEND_FONTSIZE, loc='lower right')
    plt.savefig('./figures/' + saveto_filename + '.png', dpi=720)
    print('\n\nSaved ' + saveto_filename + '.png in dir ./figures')
    plt.close()
