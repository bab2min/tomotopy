'''
..versionadded:: 0.13.0

`tomotopy.viewer` is a module for visualizing the tomotopy's topic model using a web browser. It provides a simple way to visualize the topic model and its results interactively.
But it is not recommended to use it in a production web service, because it uses python's built-in `http.server` module to serve.

The following is a simple example of how to use the viewer:
::

    import tomotopy as tp
    mdl = tp.load_model('a_trained_model.bin')
    tp.viewer.open_viewer(mdl, port=9999)
    # open http://localhost:9999 in your web browser

Or you can run the viewer from the command line:
::

    python -m tomotopy.viewer a_trained_model.bin --host localhost --port 9999
    # open http://localhost:9999 in your web browser
    
For more details, please refer to the `tomotopy.viewer.viewer_server.open_viewer` function.
'''

from .viewer_server import open_viewer
