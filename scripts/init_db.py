"""Create the analysis schema and, optionally, load a released data bundle.

    python -m scripts.init_db --create
    python -m scripts.init_db --create --load data/
    python -m scripts.init_db --check

`--load` expects `channels.csv`, `videos.csv` and `comments.csv` in the given
directory, with headers matching the column names in `db.TABLE_COLUMNS`. Rows
are inserted with `INSERT IGNORE`, so an interrupted load can be resumed by
re-running the same command.

Run with `-m` from the project root, not `python scripts/init_db.py` directly
-- this module uses relative imports to reach its sibling modules.
"""

import argparse

from . import db


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--create', action='store_true',
                        help='create the three analysis tables if they do not exist')
    parser.add_argument('--load', metavar='DATA_DIR',
                        help='bulk-load channels.csv / videos.csv / comments.csv from DATA_DIR')
    parser.add_argument('--check', action='store_true',
                        help='report row counts for each table and exit')
    parser.add_argument('--env', metavar='PATH', default=None,
                        help='path to the .env file (default: <project root>/.env)')
    args = parser.parse_args(argv)

    if not (args.create or args.load or args.check):
        parser.error('nothing to do -- pass --create, --load and/or --check')

    connection = db.connect(args.env)
    try:
        if args.create:
            print('Creating schema...')
            db.create_schema(connection)

        if args.load:
            print(f'Loading CSVs from {args.load}...')
            db.load_csv_dir(connection, args.load)

        if args.check or args.create or args.load:
            print('Row counts:')
            with connection.cursor() as cur:
                for table in db.SCHEMA_DDL:
                    try:
                        cur.execute(f'SELECT COUNT(*) AS n FROM {table}')
                        print(f'  {table:<20} {cur.fetchone()["n"]:>15,}')
                    except Exception as exc:
                        print(f'  {table:<20} unavailable ({exc})')
    finally:
        connection.close()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
