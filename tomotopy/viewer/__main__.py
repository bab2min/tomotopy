import tomotopy as tp
from tomotopy.viewer import open_viewer

def main(args):
    model = tp.load_model(args.model)
    open_viewer(model, args.host, args.port, args.root_path, args.browser_title, args.model + '.json', read_only=args.read_only)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--root-path', default='/')
    parser.add_argument('--browser-title')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('-p', '--port', type=int, default=9999)
    parser.add_argument('-r', '--read-only', action='store_true')
    main(parser.parse_args())
