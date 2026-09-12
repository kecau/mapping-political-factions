"""Replication package for the dynamic-community study of Korean political YouTube.

Every runnable piece of this replication -- data access, the tracking
algorithm, and the figures built on top of it -- lives here as one flat
package. A figure's plotting code sits directly in the module that computes
its inputs (e.g. `cohesion.py` has both `cohesion_tables()` and
`plot_wing_cohesion()`); there is no separate figures/ hierarchy to track down.

Pipeline, in the order `run_all_figures.py` and `replication.ipynb` run it:

    db.connect()                    open the MySQL connection
    data.load_channel_info()        the study's channel set and their wings
    data.data_summary()             corpus size, overall and per wing
    cache.get_daily_authors()       per-day commenter sets (Parquet-cached)
    network.build_snapshots()       weekly channel-channel overlap graphs
    communities.detect_all()        Louvain communities per snapshot
    tracking.track()                Greene et al. matching -> dynamic communities
    tracking.plot_timeline()        figure: DC lifelines over the study period
    tracking.plot_event_graph()     figure: the same output, compressed for print
    data.plot_daily_volume()        figure: daily comment volume by wing
    cohesion / core_nodes / polarity / topics
                                    computation + their own figures

Every parameter lives in `config`; nothing is hardcoded in the analysis modules.

Run any script directly with `python -m scripts.<module>` from the project
root (not `python scripts/<module>.py` -- these use relative imports).
"""

__version__ = '1.0.0'
