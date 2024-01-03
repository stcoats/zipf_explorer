from bokeh.command.bootstrap import main

main(["bokeh", "serve", "--show", "./zipfexplorer/main.py"])

import os
from bokeh.command.bootstrap import main

def webserver():
    # Get the path to the directory containing your Bokeh application
    # Not mandatory, but clearer
    app_directory = os.path.abspath("zipfexplorer")

    # Use bokeh serve to serve the app
    main(["bokeh", "serve", "zipfexplorer", "--port", "0", "--allow-websocket-origin", "zipfexplorer-zipf9.rahtiapp.fi", "--address", "0.0.0.0", "--use-xheaders"])

if __name__ == '__main__':
    webserver()
