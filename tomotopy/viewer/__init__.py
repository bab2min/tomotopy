'''
..versionadded:: 0.13.0

`tomotopy.viewer` is a module for visualizing the tomotopy's topic model using a web browser. It provides a simple way to visualize the topic model and its results interactively.
But it is not recommended to use it in a production web service, because it uses python's built-in `http.server` module to serve.
'''

from .viewer_server import open_viewer
