import os
from bokeh.command.bootstrap import main

def webserver():
    # Get the port from the environment variable (use 8080 as default)
    port = os.getenv('PORT', '8080')

    # Use bokeh serve to serve the app
    main(["bokeh", "serve", "zipfexplorer", "--port", port, "--allow-websocket-origin", os.getenv('ALLOW_ORIGIN', 'zipfexplorer-zipf9.rahtiapp.fi'), "--address", "0.0.0.0", "--use-xheaders"])

if __name__ == '__main__':
    webserver()

