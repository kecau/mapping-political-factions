"""Run the whole analysis headless and write every figure and table.

    python -m scripts.run_all_figures
    python -m scripts.run_all_figures --skip-topics     # skip the LDA stage

The call sequence here is the same one `replication.ipynb` walks through; the
notebook exists to explain each step, this script exists to regenerate
everything in one go.

Run with `-m` from the project root, not `python scripts/run_all_figures.py`
directly -- this module uses relative imports to reach its sibling modules.
"""

import argparse
import time

import matplotlib

matplotlib.use('Agg')   # no display in a headless run

from . import (cache, channels, cohesion, communities, config, core_nodes,
               data, db, network, polarity, style, topics, tracking)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-topics', action='store_true',
                        help='skip the LDA topic model and the two topic figures')
    parser.add_argument('--skip-polarity', action='store_true',
                        help='skip the polarity figure (its queries are the slowest step)')
    parser.add_argument('--env', metavar='PATH', default=None)
    args = parser.parse_args(argv)

    started = time.time()
    style.apply_style()
    connection = db.connect(args.env)

    try:
        # ── 1. Channels and corpus summary ──────────────────────────────────
        print('\n[1/9] Channel set and data summary')
        channel_info_map = data.load_channel_info(connection, channels.CHANNEL_IDS)
        channel_ids = list(channel_info_map)
        print(f'  {len(channel_ids)} channels loaded')
        summary_df = data.data_summary(connection, channel_info_map)
        print(data.format_summary(summary_df))
        style.save_table(summary_df.reset_index(), 'data_summary')

        # ── 2. Snapshots ────────────────────────────────────────────────────
        print('\n[2/9] Building snapshots')
        daily_authors = cache.get_daily_authors(channel_ids, config.START_DATE,
                                                config.END_DATE, connection)
        snapshot_dates, snapshots = network.build_snapshots(daily_authors, channel_ids)

        # ── 3. Community detection and tracking ─────────────────────────────
        print('\n[3/9] Detecting and tracking communities')
        snapshot_communities = communities.detect_all(snapshots)
        dcs, events = tracking.track(snapshot_communities)
        style.save_table(tracking.event_log(events, snapshot_dates, dcs, channel_info_map),
                         'event_log')

        # ── 4. Timeline and event graph ─────────────────────────────────────
        print('\n[4/9] Timeline and event graph')
        tracking.plot_timeline(dcs, events, snapshot_dates, channel_info_map)
        tracking.plot_event_graph(dcs, events, snapshot_dates, channel_info_map)

        # ── 5. Daily comment volume ─────────────────────────────────────────
        print('\n[5/9] Daily comment volume')
        data.plot_daily_volume(data.daily_comment_volume(connection, channel_info_map))

        # ── 6. Cohesion ─────────────────────────────────────────────────────
        print('\n[6/9] Cohesion (density and conductance)')
        cohesion_df, dc_cohesion_summary_df, wing_cohesion_df = cohesion.cohesion_tables(
            dcs, snapshots, snapshot_dates, channel_info_map)
        style.save_table(dc_cohesion_summary_df, 'dc_cohesion_summary')
        cohesion.plot_cohesion(cohesion_df, dc_cohesion_summary_df, wing_cohesion_df)

        # ── 7. Core nodes ───────────────────────────────────────────────────
        print('\n[7/9] Core and broker channels')
        _, dc_core_nodes_df = core_nodes.core_nodes_table(
            dcs, snapshots, snapshot_dates, channel_info_map)
        style.save_table(dc_core_nodes_df, 'dc_core_nodes')
        core_nodes.plot_core_nodes(dc_core_nodes_df, dc_cohesion_summary_df, dcs,
                                   channel_info_map)

        # ── 8. Polarity ─────────────────────────────────────────────────────
        if args.skip_polarity:
            print('\n[8/9] Polarity -- skipped')
        else:
            print('\n[8/9] Polarity distributions')
            bipartite_scores = polarity.compute_bipartite_polarity(connection, channel_info_map)
            pairs = polarity.resolve_dc_pairs(dcs, dc_cohesion_summary_df, channel_info_map)
            plottable, _ = polarity.polarity_for_pairs(connection, dcs, pairs,
                                                       snapshot_dates, channel_info_map)
            polarity.plot_polarity_distributions(bipartite_scores, plottable)

        # ── 9. Topics ───────────────────────────────────────────────────────
        if args.skip_topics:
            print('\n[9/9] Topics -- skipped')
        else:
            print('\n[9/9] Global LDA topic model')
            videos_df = cache.get_videos(channel_ids, config.START_DATE,
                                         config.END_DATE, connection)
            print(f'  {len(videos_df):,} videos loaded')
            model = topics.fit_global_lda(videos_df)
            matrix, labels = topics.dc_topic_matrix(dcs, videos_df, snapshot_dates,
                                                    model, channel_info_map)
            topics.plot_dc_topic_heatmap(matrix, labels, model)
            topics.plot_topic_prevalence(
                topics.topic_prevalence_matrix(snapshot_dates, videos_df, model),
                snapshot_dates, model)
    finally:
        connection.close()

    print(f'\nDone in {time.time() - started:.0f}s.')
    print(f'Figures -> {config.FIGURE_DIR}')
    print(f'Tables  -> {config.TABLE_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
