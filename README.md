# Serving

Model Serving Benchmark

Metric	gRPC (TF) / REST (TF) / Flask / Flask+gunicorn

AVG (s)	0.983 / 1.624 / 9.088 / 9.394

STD (s)	0.011 / 0.021 / 0.378 / 0.348




$ docker build -t flask_serving .
$ docker run -it --rm -p 8080:80 flask_serving
