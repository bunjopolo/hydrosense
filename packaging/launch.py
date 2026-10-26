"""PyInstaller entry point; preserve error messages when double-clicked."""
import multiprocessing
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        from hydrosense.desktop import main
        main()
    except Exception as exc:
        print(f'\nHydroSense could not start: {exc}', file=sys.stderr, flush=True)
        if len(sys.argv) == 1:
            print('If another HydroSense is open, use its browser window or close it first.', flush=True)
            input('Press Enter to close…')
        raise SystemExit(1)
