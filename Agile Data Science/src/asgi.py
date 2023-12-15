from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
import os
import subprocess

def run_streamlit():
    cmd = "streamlit run app.py --server.port 8501"
    subprocess.Popen(cmd, shell=True)

routes = [
    Mount("/", app=StaticFiles(directory="public"), name="public"),
]

middleware = [
    Middleware(HTTPSRedirectMiddleware)
]

application = Starlette(debug=True, routes=routes, middleware=middleware, on_startup=[run_streamlit])
