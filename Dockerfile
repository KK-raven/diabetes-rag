FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# JupyterLabの設定ディレクトリを作成
RUN mkdir -p /root/.local/share/jupyter/lab/user-settings/@jupyterlab/apputils-extension

# ダークモードをデフォルトに設定
RUN echo '{"theme": "JupyterLab Dark"}' > /root/.local/share/jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]