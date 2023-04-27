FROM python:3.10
ADD app.py .
ADD requirements.txt .
ADD Top100-US.csv .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]