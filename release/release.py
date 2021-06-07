import argparse
import logging

from api_proxy import api_proxy

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('release tools')
    
    subparsers = parser.add_subparsers(title='actions', dest='action')
    
    register_command = subparsers.add_parser('register')
    register_command.add_argument('--model_name', type=str, required=True)

    update_command = subparsers.add_parser('update')
    update_command.add_argument('--model_name', type=str, required=True)
    update_command.add_argument('--model_base_path', type=str, required=True)
    update_command.add_argument('--model_version', type=int, required=True)

    delete_command = subparsers.add_parser('delete')
    delete_command.add_argument('--model_name', type=str, required=True)
    delete_command.add_argument('--model_version', type=int, required=True)

    subparsers.add_parser('list')
    
    return parser

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = build_parser()
    args = vars(parser.parse_args())
    
    action = args.pop('action')
    print(api_proxy(action, args))
    