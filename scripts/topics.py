"""Content characterisation: Korean tokenisation and the global LDA topic model.

One LDA model is fit over every video title/tag blob in the study period
(section 18h). Fitting one *global* model rather than a separate model per
community is what makes the topic axis shared: every dynamic community, and
every week, can then be projected onto the same fourteen topics and compared
directly. Per-community models would each invent their own topic numbering and
could not be placed side by side.

`plot_dc_topic_heatmap` and `plot_topic_prevalence` turn that shared axis into
the two topic figures: which topics each faction concentrated on across its
lifetime, and what the whole corpus was talking about each week.
"""

from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from . import config
from .style import (EVENT_COLOR, WING_LETTER, WING_ORDER, add_political_events,
                    format_month_axis, save_figure)
from .tracking import dominant_wing

# Generic newsroom vocabulary that appears across every channel regardless of
# topic -- keeping it would make several topics collapse onto "this is a news
# video" instead of what the video is about.
STOPWORDS_KO = {
    '오늘', '영상', '뉴스', '기자', '채널', '속보', '단독', '종합', '현장',
    '인터뷰', '방송', '풀영상', '다시보기', '라이브', '생방송', '클립',
    '하이라이트', '모음', '정리', '관련', '이번', '최신',
}

_kiwi = None


def _get_kiwi():
    """Lazily construct the Kiwi tokenizer -- it costs 1-2s and is reused."""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def kiwi_tokenize(text):
    """Extract noun-like keyword tokens from a Korean video title/tag blob.

    Korean is agglutinative, so whitespace splitting produces inflected junk;
    morphological analysis is required. Only common and proper nouns (NNG/NNP)
    are kept, single-character tokens are dropped as too ambiguous, and the
    newsroom stopwords above are removed.
    """
    if not text:
        return []
    text = text.replace('#', ' ')
    tokens = []
    for tok in _get_kiwi().tokenize(text):
        if tok.tag not in ('NNG', 'NNP'):
            continue
        if len(tok.form) < 2:
            continue
        if tok.form in STOPWORDS_KO:
            continue
        tokens.append(tok.form)
    return tokens


class GlobalLDA:
    """Fitted whole-period topic model plus the lookups the figures need."""

    def __init__(self, lda, vectorizer, vocab, doc_topic, topic_words,
                 topic_label, video_topic_of, n_topics):
        self.lda = lda
        self.vectorizer = vectorizer
        self.vocab = vocab
        self.doc_topic = doc_topic          # (n_videos, n_topics)
        self.topic_words = topic_words      # [[str] * TOPIC_TOP_WORDS] * n_topics
        self.topic_label = topic_label      # {topic_id: 'word/word'}
        self.video_topic_of = video_topic_of  # {video_id: topic distribution}
        self.n_topics = n_topics

    @property
    def overall_prevalence(self):
        """Corpus-wide mean weight of each topic."""
        return self.doc_topic.mean(axis=0)

    def describe(self):
        lines = [f'Fit {self.n_topics}-topic global LDA over {self.doc_topic.shape[0]:,} videos']
        for k, words in enumerate(self.topic_words):
            lines.append(f'  Topic {k:2d} ({self.topic_label[k]}): {", ".join(words)}')
        return '\n'.join(lines)


def fit_global_lda(videos_df, n_topics=None, random_state=None, min_df=None,
                   max_df=None, top_words=None, verbose=True):
    """Fit the whole-period LDA over video titles + tags."""
    n_topics = config.N_TOPICS_GLOBAL if n_topics is None else n_topics
    random_state = config.LDA_RANDOM_STATE if random_state is None else random_state
    min_df = config.LDA_MIN_DF if min_df is None else min_df
    max_df = config.LDA_MAX_DF if max_df is None else max_df
    top_words = config.TOPIC_TOP_WORDS if top_words is None else top_words

    docs = (videos_df['title'].fillna('') + ' ' +
            videos_df['tags'].fillna('').str.replace(',', ' ')).tolist()

    vectorizer = CountVectorizer(tokenizer=kiwi_tokenize, token_pattern=None,
                                 max_df=max_df, min_df=min_df)
    X = vectorizer.fit_transform(docs)
    vocab = np.array(vectorizer.get_feature_names_out())

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state,
                                    learning_method='batch')
    doc_topic = lda.fit_transform(X)

    topic_words = []
    for k in range(n_topics):
        top_idx = lda.components_[k].argsort()[::-1][:top_words]
        topic_words.append(list(vocab[top_idx]))
    topic_label = {k: '/'.join(words[:2]) for k, words in enumerate(topic_words)}

    model = GlobalLDA(lda, vectorizer, vocab, doc_topic, topic_words, topic_label,
                      dict(zip(videos_df['video_id'].values, doc_topic)), n_topics)
    if verbose:
        print(model.describe())
    return model


def dc_lifetime_corpus(dc, videos_df, snapshot_dates, window_days=None):
    """Every video the DC's member channels published while it held a front.

    Membership is evaluated per snapshot, so a channel only contributes the
    videos it published during the windows it was actually in the faction.
    """
    window_days = config.WINDOW_DAYS if window_days is None else window_days
    frames = []
    for t, front in dc.fronts.items():
        ws = snapshot_dates[t]
        we = ws + timedelta(days=window_days - 1)
        frames.append(videos_df[
            videos_df['channel_id'].isin(front)
            & (videos_df['published_at'] >= ws) & (videos_df['published_at'] <= we)
        ])
    if not frames:
        return pd.DataFrame(columns=videos_df.columns)
    return pd.concat(frames, ignore_index=True).drop_duplicates('video_id')


def dc_mean_topic_vector(dc, videos_df, snapshot_dates, video_topic_of,
                         window_days=None):
    """Mean topic distribution over the DC's whole lifetime corpus."""
    corpus = dc_lifetime_corpus(dc, videos_df, snapshot_dates, window_days)
    vecs = [video_topic_of[v] for v in corpus['video_id'] if v in video_topic_of]
    return np.mean(vecs, axis=0) if vecs else None


def dc_topic_matrix(dcs, videos_df, snapshot_dates, model, channel_info_map,
                    window_days=None):
    """(DC x topic) matrix of mean lifetime topic weights, plus row labels.

    Rows are ordered by (wing, dc_id) so same-wing factions sit together and the
    heatmap reads as two blocks.
    """
    rows, labels = [], []
    ordered = sorted(dcs.values(),
                     key=lambda d: (WING_ORDER[dominant_wing(d.latest_front(), channel_info_map)],
                                    d.id))
    for dc in ordered:
        vec = dc_mean_topic_vector(dc, videos_df, snapshot_dates,
                                   model.video_topic_of, window_days)
        if vec is None:
            continue
        rows.append(vec)
        wing = dominant_wing(dc.latest_front(), channel_info_map)
        labels.append(f'DC {dc.id} ({WING_LETTER[wing]})')

    return np.array(rows), labels


def topic_prevalence_matrix(snapshot_dates, videos_df, model, window_days=None):
    """(snapshot x topic) matrix of mean topic weight across *all* channels.

    Community-agnostic: this is what the whole corpus was talking about each
    week, the backdrop the per-community topic profiles sit against. Weeks with
    no videos are NaN rather than zero, so a gap reads as "no data" instead of
    "no interest".
    """
    window_days = config.WINDOW_DAYS if window_days is None else window_days
    rows = []
    for ws in snapshot_dates:
        we = ws + timedelta(days=window_days - 1)
        vids = videos_df[(videos_df['published_at'] >= ws)
                         & (videos_df['published_at'] <= we)]['video_id']
        vecs = [model.video_topic_of[v] for v in vids if v in model.video_topic_of]
        rows.append(np.mean(vecs, axis=0) if vecs else np.full(model.n_topics, np.nan))
    return np.array(rows)


def _heatmap_grid(ax, n_cols, n_rows):
    """Draw cell-boundary gridlines instead of the inherited through-cell grid.

    seaborn's `whitegrid` theme draws gridlines at each axes' *major* ticks --
    here the cell centres, which puts a line straight through every annotated
    value. Turn that off and redraw the grid on minor ticks offset by half a
    cell, so the lines fall between cells.
    """
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', bottom=True, left=True)


def plot_dc_topic_heatmap(matrix, dc_labels, model, keyword_count=9, name=None, save=True):
    """DC x topic heatmap with the topic keywords as a legend inset.

    The x-axis carries bare topic ids to keep the grid readable; the legend on
    the right is what actually says what each id means, so the figure stands on
    its own without a separate printed topic list.
    """
    n_topics = model.n_topics
    heatmap_w = 0.5 * n_topics + 2
    legend_w = 4.0

    fig, ax = plt.subplots(figsize=(heatmap_w + legend_w, 0.3 * len(dc_labels) + 2))
    im = ax.imshow(matrix, aspect='auto', cmap='viridis')

    # Per-cell values, with text colour picked against imshow's own
    # normalisation so digits stay legible at both ends of the colormap.
    norm = im.norm
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                    color='white' if norm(val) < 0.6 else 'black')

    ax.set_xticks(range(n_topics))
    ax.set_xticklabels(range(n_topics), fontsize=9)
    ax.set_xlabel('Topic ID', fontsize=9)
    ax.set_yticks(range(len(dc_labels)))
    ax.set_yticklabels(dc_labels, fontsize=9)
    _heatmap_grid(ax, n_topics, len(dc_labels))

    cbar = fig.colorbar(im, ax=ax, label='Mean topic weight', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    ax.legend(handles=[
        Line2D([], [], color='none',
               label=f"{k}: {', '.join(model.topic_words[k][:keyword_count])}")
        for k in range(n_topics)
    ], loc='lower left', bbox_to_anchor=(1.33, 0), fontsize=7, frameon=True,
        handlelength=0, handletextpad=0, title='Topic keywords', title_fontsize=8)

    fig.tight_layout()
    if save:
        save_figure(fig, name or f'dc_topic_heatmap_{config.START_DATE}_{config.END_DATE}')
    return fig


def plot_topic_prevalence(matrix, snapshot_dates, model, name=None, save=True):
    """Mean topic weight per snapshot across all channels, with event markers."""
    n_topics = model.n_topics
    topic_colors = [plt.cm.tab20(i) for i in range(n_topics)]

    fig, ax = plt.subplots(figsize=(14, 6))
    for k in range(n_topics):
        ax.plot(snapshot_dates, matrix[:, k], color=topic_colors[k],
                linewidth=1.3, label=f'Topic {k}')

    ax.set_ylabel('Mean topic weight')
    format_month_axis(ax)
    add_political_events(ax)

    ax.legend(handles=[
        Line2D([0], [0], color=topic_colors[k], linewidth=1.6, label=f'Topic {k}')
        for k in range(n_topics)
    ] + [
        Line2D([0], [0], color=EVENT_COLOR, linestyle='--', linewidth=1, alpha=0.5,
               label='political event')
    ], loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, frameon=True)

    fig.tight_layout()
    if save:
        save_figure(fig, name or f'topic_prevalence_{config.START_DATE}_{config.END_DATE}')
    return fig
